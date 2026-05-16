package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"imageclust/internal/clip"
	"imageclust/internal/models"
	"imageclust/internal/ollama"
	"imageclust/internal/utils"
	"imageclust/internal/workflow"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
)

const (
	sessionTTL      = 1 * time.Hour
	cleanupInterval = 10 * time.Minute
)

// Handlers holds shared dependencies injected at startup.
type Handlers struct {
	clipModel    *clip.Model
	ollamaClient *ollama.Client
	store        *sessionStore
}

// New returns a Handlers instance wired with the given model and Ollama client.
func New(clipModel *clip.Model, ollamaClient *ollama.Client) *Handlers {
	s := &sessionStore{sessions: make(map[string]sessionEntry)}
	go s.cleanupExpiredSessions()
	return &Handlers{
		clipModel:    clipModel,
		ollamaClient: ollamaClient,
		store:        s,
	}
}

// --- Session store --------------------------------------------------------

type sessionEntry struct {
	tempDir   string
	createdAt time.Time
}

type sessionStore struct {
	sessions map[string]sessionEntry
	mu       sync.RWMutex
}

func (s *sessionStore) cleanupExpiredSessions() {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()
	for range ticker.C {
		s.mu.Lock()
		for id, entry := range s.sessions {
			if time.Since(entry.createdAt) > sessionTTL {
				os.RemoveAll(entry.tempDir)
				delete(s.sessions, id)
				log.Printf("handlers: cleaned up expired session %s", id)
			}
		}
		s.mu.Unlock()
	}
}

func (s *sessionStore) set(id, dir string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[id] = sessionEntry{tempDir: dir, createdAt: time.Now()}
}

func (s *sessionStore) get(id string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.sessions[id]
	return e.tempDir, ok
}

func generateSessionID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

// --- Middleware -----------------------------------------------------------

// EnableCORS adds permissive CORS headers. Restrict origins in production.
func EnableCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// --- Cluster handler ------------------------------------------------------

// clusterResponse is the JSON shape returned after successful clustering.
type clusterResponse struct {
	Status    string           `json:"status"`
	SessionID string           `json:"sessionId"`
	Clusters  []clusterPayload `json:"clusters"`
}

type clusterPayload struct {
	ID           string   `json:"id"`
	Title        string   `json:"title"`
	CatchyPhrase string   `json:"catchy_phrase"`
	Images       []string `json:"images"`
}

// ClusterAndGenerate handles POST /api/cluster.
func (h *Handlers) ClusterAndGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		respondWithError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	if err := r.ParseMultipartForm(32 << 20); err != nil {
		respondWithError(w, http.StatusBadRequest, "failed to parse form data")
		return
	}

	sessionID, err := generateSessionID()
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "failed to generate session ID")
		return
	}

	tempDir, err := os.MkdirTemp("", "imageclust_*")
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "failed to create temp directory")
		return
	}
	h.store.set(sessionID, tempDir)

	uploadedImages, uploadErrors := readUploadedImages(r)
	if len(uploadedImages) == 0 {
		msg := "no valid images uploaded"
		if len(uploadErrors) > 0 {
			msg += ": " + strings.Join(uploadErrors, "; ")
		}
		respondWithError(w, http.StatusBadRequest, msg)
		return
	}
	if len(uploadErrors) > 0 {
		log.Printf("handlers: %d files failed during upload: %s", len(uploadErrors), strings.Join(uploadErrors, "; "))
	}

	minClusterSize, maxClusterSize := parseClusterSizes(r, 3, 6)

	ic := workflow.NewImageCluster(minClusterSize, maxClusterSize, tempDir, h.clipModel, h.ollamaClient)
	clusterDetails, err := ic.Run(uploadedImages)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	respondWithJSON(w, http.StatusOK, buildClusterResponse(sessionID, clusterDetails))
}

func readUploadedImages(r *http.Request) ([]models.UploadedImage, []string) {
	var images []models.UploadedImage
	var errs []string
	for _, fh := range r.MultipartForm.File["images"] {
		f, err := fh.Open()
		if err != nil {
			errs = append(errs, fmt.Sprintf("open %s: %v", fh.Filename, err))
			continue
		}
		data, err := io.ReadAll(f)
		f.Close()
		if err != nil {
			errs = append(errs, fmt.Sprintf("read %s: %v", fh.Filename, err))
			continue
		}
		images = append(images, models.UploadedImage{
			Filename: utils.SanitizeFilename(fh.Filename),
			Data:     data,
		})
	}
	return images, errs
}

func parseClusterSizes(r *http.Request, defaultMin, defaultMax int) (int, int) {
	min, max := defaultMin, defaultMax
	if v, err := strconv.Atoi(r.FormValue("minClusterSize")); err == nil && v >= 2 {
		min = v
	}
	if v, err := strconv.Atoi(r.FormValue("maxClusterSize")); err == nil && v >= min {
		max = v
	}
	return min, max
}

func buildClusterResponse(sessionID string, details map[string]models.ClusterDetails) clusterResponse {
	payloads := make([]clusterPayload, 0, len(details))
	for id, d := range details {
		payloads = append(payloads, clusterPayload{
			ID:           id,
			Title:        d.Title,
			CatchyPhrase: d.CatchyPhrase,
			Images:       d.Images,
		})
	}
	return clusterResponse{
		Status:    "success",
		SessionID: sessionID,
		Clusters:  payloads,
	}
}

// --- Image handler --------------------------------------------------------

// ServeImage handles GET /api/image/{imageName}?session=<id>.
func (h *Handlers) ServeImage(w http.ResponseWriter, r *http.Request) {
	sessionID := r.URL.Query().Get("session")
	if sessionID == "" {
		http.Error(w, "missing session parameter", http.StatusBadRequest)
		return
	}
	tempDir, ok := h.store.get(sessionID)
	if !ok {
		http.Error(w, "invalid or expired session", http.StatusNotFound)
		return
	}

	imageName := utils.SanitizeFilename(mux.Vars(r)["imageName"])
	imagePath := filepath.Join(tempDir, "images", imageName)
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		http.Error(w, "image not found", http.StatusNotFound)
		return
	}

	switch strings.ToLower(filepath.Ext(imageName)) {
	case ".png":
		w.Header().Set("Content-Type", "image/png")
	case ".gif":
		w.Header().Set("Content-Type", "image/gif")
	case ".webp":
		w.Header().Set("Content-Type", "image/webp")
	default:
		w.Header().Set("Content-Type", "image/jpeg")
	}
	http.ServeFile(w, r, imagePath)
}

// --- SPA handler ----------------------------------------------------------

// SpaHandler serves the React frontend, falling back to index.html for
// unknown paths (client-side routing).
type SpaHandler struct {
	StaticPath string
	IndexPath  string
}

func (h SpaHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := filepath.Join(h.StaticPath, r.URL.Path)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		http.ServeFile(w, r, filepath.Join(h.StaticPath, h.IndexPath))
		return
	}
	http.FileServer(http.Dir(h.StaticPath)).ServeHTTP(w, r)
}

// --- Helpers --------------------------------------------------------------

func respondWithError(w http.ResponseWriter, code int, message string) {
	respondWithJSON(w, code, map[string]interface{}{"success": false, "error": message})
}

func respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	b, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, "failed to marshal response", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(b)
}

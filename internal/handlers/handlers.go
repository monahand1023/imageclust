package handlers

import (
	"archive/zip"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"imageclust/internal/clip"
	"imageclust/internal/models"
	"imageclust/internal/ollama"
	"imageclust/internal/utils"
	"imageclust/internal/workflow"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
)

const (
	sessionTTL      = 1 * time.Hour
	cleanupInterval = 10 * time.Minute
	maxUploadBytes  = 32 << 20 // 32 MB
	maxImageCount   = 200
)

// Handlers holds shared dependencies injected at startup.
type Handlers struct {
	clipModel    *clip.Model
	ollamaClient *ollama.Client
	modelPath    string
	store        *sessionStore
}

// New returns a Handlers instance wired with the given model and Ollama
// client. modelPath is used by the health check. Call Close to stop the
// background session cleanup.
func New(clipModel *clip.Model, ollamaClient *ollama.Client, modelPath string) *Handlers {
	s := newSessionStore(sessionTTL, cleanupInterval)
	go s.cleanupExpiredSessions()
	return &Handlers{
		clipModel:    clipModel,
		ollamaClient: ollamaClient,
		modelPath:    modelPath,
		store:        s,
	}
}

// Close stops the background session-cleanup goroutine.
func (h *Handlers) Close() {
	h.store.stop()
}

// --- Health handler -------------------------------------------------------

const version = "1.0.0"

type healthResponse struct {
	Status  string       `json:"status"`
	Version string       `json:"version"`
	Checks  healthChecks `json:"checks"`
}

type healthChecks struct {
	Ollama      string `json:"ollama"`
	ModelLoaded bool   `json:"model_loaded"`
}

// HealthHandler handles GET /health. It returns a JSON summary of service
// readiness: whether Ollama is reachable and whether the CLIP model file exists.
// Returns 200 when all checks pass, 503 when any check fails.
func (h *Handlers) HealthHandler(w http.ResponseWriter, r *http.Request) {
	checks := healthChecks{}
	allOK := true

	// Check Ollama reachability.
	if err := h.ollamaClient.Ping(r.Context()); err != nil {
		checks.Ollama = "unavailable"
		allOK = false
	} else {
		checks.Ollama = "ok"
	}

	// Check CLIP model path exists (file or directory).
	if h.modelPath != "" {
		if _, err := os.Stat(h.modelPath); err == nil {
			checks.ModelLoaded = true
		} else {
			checks.ModelLoaded = false
			allOK = false
		}
	} else {
		// No path configured — treat as unavailable.
		checks.ModelLoaded = false
		allOK = false
	}

	status := "ok"
	httpCode := http.StatusOK
	if !allOK {
		status = "degraded"
		httpCode = http.StatusServiceUnavailable
	}

	respondWithJSON(w, httpCode, healthResponse{
		Status:  status,
		Version: version,
		Checks:  checks,
	})
}

// --- Session store --------------------------------------------------------

// exportData is the per-session clustering outcome kept for ZIP export.
type exportData struct {
	clusters    []clusterPayload
	unclustered []string
}

type sessionEntry struct {
	tempDir   string
	createdAt time.Time
	result    *exportData // nil until a clustering run completes
}

type sessionStore struct {
	sessions map[string]sessionEntry
	mu       sync.RWMutex
	ttl      time.Duration
	interval time.Duration
	done     chan struct{}
	stopOnce sync.Once
}

func newSessionStore(ttl, interval time.Duration) *sessionStore {
	return &sessionStore{
		sessions: make(map[string]sessionEntry),
		ttl:      ttl,
		interval: interval,
		done:     make(chan struct{}),
	}
}

func (s *sessionStore) cleanupExpiredSessions() {
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
		}
		s.mu.Lock()
		var toDelete []string
		for id, entry := range s.sessions {
			if time.Since(entry.createdAt) > s.ttl {
				toDelete = append(toDelete, id)
			}
		}
		for _, id := range toDelete {
			entry := s.sessions[id]
			if err := os.RemoveAll(entry.tempDir); err != nil {
				log.Printf("handlers: failed to cleanup temp dir %s: %v", entry.tempDir, err)
			}
			delete(s.sessions, id)
			log.Printf("handlers: cleaned up expired session %s", id)
		}
		s.mu.Unlock()
	}
}

// stop halts the cleanup goroutine. Safe to call multiple times.
func (s *sessionStore) stop() {
	s.stopOnce.Do(func() { close(s.done) })
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

// setResult attaches the clustering outcome to an existing session.
func (s *sessionStore) setResult(id string, result *exportData) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.sessions[id]; ok {
		e.result = result
		s.sessions[id] = e
	}
}

func (s *sessionStore) getResult(id string) (string, *exportData, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.sessions[id]
	if !ok || e.result == nil {
		return "", nil, false
	}
	return e.tempDir, e.result, true
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
	Status      string                `json:"status"`
	SessionID   string                `json:"sessionId"`
	Clusters    []clusterPayload      `json:"clusters"`
	Unclustered []string              `json:"unclustered,omitempty"`
	Skipped     []models.SkippedImage `json:"skipped,omitempty"`
}

type clusterPayload struct {
	ID           string   `json:"id"`
	Title        string   `json:"title"`
	CatchyPhrase string   `json:"catchy_phrase"`
	Images       []string `json:"images"`
}

// ClusterAndGenerate handles POST /api/cluster. Method restriction is
// enforced by the router.
func (h *Handlers) ClusterAndGenerate(w http.ResponseWriter, r *http.Request) {
	// ParseMultipartForm's argument only bounds memory use; MaxBytesReader
	// actually caps the request body.
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			respondWithError(w, http.StatusRequestEntityTooLarge,
				fmt.Sprintf("upload exceeds %d MB limit", maxUploadBytes>>20))
			return
		}
		respondWithError(w, http.StatusBadRequest, "failed to parse form data")
		return
	}

	if n := len(r.MultipartForm.File["images"]); n > maxImageCount {
		respondWithError(w, http.StatusBadRequest,
			fmt.Sprintf("too many images: %d (max %d)", n, maxImageCount))
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
	result, err := ic.Run(r.Context(), uploadedImages)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	resp := buildClusterResponse(sessionID, result)
	h.store.setResult(sessionID, &exportData{
		clusters:    resp.Clusters,
		unclustered: resp.Unclustered,
	})
	respondWithJSON(w, http.StatusOK, resp)
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

func buildClusterResponse(sessionID string, result *workflow.Result) clusterResponse {
	payloads := make([]clusterPayload, 0, len(result.Clusters))
	for id, d := range result.Clusters {
		payloads = append(payloads, clusterPayload{
			ID:           id,
			Title:        d.Title,
			CatchyPhrase: d.CatchyPhrase,
			Images:       d.Images,
		})
	}
	// Map iteration order is random; sort by the numeric suffix of
	// "Cluster-N" so the UI gets a stable ordering.
	sort.Slice(payloads, func(i, j int) bool {
		return clusterNum(payloads[i].ID) < clusterNum(payloads[j].ID)
	})
	return clusterResponse{
		Status:      "success",
		SessionID:   sessionID,
		Clusters:    payloads,
		Unclustered: result.Unclustered,
		Skipped:     result.Skipped,
	}
}

// clusterNum extracts N from "Cluster-N"; unparseable IDs sort last.
func clusterNum(id string) int {
	n, err := strconv.Atoi(strings.TrimPrefix(id, "Cluster-"))
	if err != nil {
		return int(^uint(0) >> 1)
	}
	return n
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
	// Guard against path traversal: SanitizeFilename allows '.' so ".." survives.
	if imageName == "" || !filepath.IsLocal(imageName) {
		http.Error(w, "invalid image path", http.StatusBadRequest)
		return
	}
	imagePath := filepath.Join(tempDir, "images", imageName)

	if _, err := os.Stat(imagePath); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			http.Error(w, "image not found", http.StatusNotFound)
		} else {
			http.Error(w, "internal error", http.StatusInternalServerError)
		}
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

// --- Export handler ---------------------------------------------------------

// ExportZip handles GET /api/export?session=<id>. It streams a ZIP archive
// with one folder per cluster (named "NN Title") plus an "unclustered" folder.
func (h *Handlers) ExportZip(w http.ResponseWriter, r *http.Request) {
	sessionID := r.URL.Query().Get("session")
	if sessionID == "" {
		http.Error(w, "missing session parameter", http.StatusBadRequest)
		return
	}
	tempDir, data, ok := h.store.getResult(sessionID)
	if !ok {
		http.Error(w, "invalid or expired session", http.StatusNotFound)
		return
	}
	imagesDir := filepath.Join(tempDir, "images")

	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="imageclust-clusters.zip"`)

	zw := zip.NewWriter(w)
	addFile := func(zipPath, filename string) {
		if !filepath.IsLocal(filename) {
			return
		}
		src, err := os.Open(filepath.Join(imagesDir, filename))
		if err != nil {
			log.Printf("handlers: export: skipping %s: %v", filename, err)
			return
		}
		defer src.Close()
		dst, err := zw.Create(zipPath)
		if err != nil {
			log.Printf("handlers: export: create %s: %v", zipPath, err)
			return
		}
		if _, err := io.Copy(dst, src); err != nil {
			log.Printf("handlers: export: copy %s: %v", zipPath, err)
		}
	}

	for i, c := range data.clusters {
		folder := fmt.Sprintf("%02d %s", i+1, zipFolderName(c.Title))
		for _, img := range c.Images {
			addFile(folder+"/"+img, img)
		}
	}
	for _, img := range data.unclustered {
		addFile("unclustered/"+img, img)
	}

	if err := zw.Close(); err != nil {
		log.Printf("handlers: export: close zip: %v", err)
	}
}

// zipFolderName makes a cluster title safe as a ZIP directory name.
func zipFolderName(title string) string {
	title = strings.Map(func(r rune) rune {
		switch r {
		case '/', '\\', ':', 0:
			return '_'
		}
		return r
	}, strings.TrimSpace(title))
	if title == "" {
		return "cluster"
	}
	return title
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
	if _, err := os.Stat(path); err != nil {
		http.ServeFile(w, r, filepath.Join(h.StaticPath, h.IndexPath))
		return
	}
	http.FileServer(http.Dir(h.StaticPath)).ServeHTTP(w, r)
}

// --- Helpers --------------------------------------------------------------

func respondWithError(w http.ResponseWriter, code int, message string) {
	respondWithJSON(w, code, map[string]any{"status": "error", "error": message})
}

func respondWithJSON(w http.ResponseWriter, code int, payload any) {
	b, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, "failed to marshal response", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(b)
}

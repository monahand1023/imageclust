package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"imageclust/internal/models"
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
	"imageclust/internal/utils"
	"imageclust/internal/workflow"
)

const (
	// sessionTTL is how long sessions are kept before cleanup
	sessionTTL = 1 * time.Hour
	// cleanupInterval is how often the cleanup routine runs
	cleanupInterval = 10 * time.Minute
)

// SpaHandler implements the http.Handler interface for serving a Single Page Application
type SpaHandler struct {
	StaticPath string
	IndexPath  string
}

// sessionEntry holds session data with creation time for TTL-based cleanup
type sessionEntry struct {
	tempDir   string
	createdAt time.Time
}

// sessionStore manages temp directories per session to avoid race conditions
type sessionStore struct {
	sessions map[string]sessionEntry
	mutex    sync.RWMutex
}

var store = &sessionStore{
	sessions: make(map[string]sessionEntry),
}

func init() {
	go store.cleanupExpiredSessions()
}

// cleanupExpiredSessions periodically removes expired sessions and their temp directories
func (s *sessionStore) cleanupExpiredSessions() {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		s.mutex.Lock()
		now := time.Now()
		for sessionID, entry := range s.sessions {
			if now.Sub(entry.createdAt) > sessionTTL {
				// Clean up temp directory
				if err := os.RemoveAll(entry.tempDir); err != nil {
					log.Printf("Failed to clean up temp directory %s: %v", entry.tempDir, err)
				} else {
					log.Printf("Cleaned up expired session %s", sessionID)
				}
				delete(s.sessions, sessionID)
			}
		}
		s.mutex.Unlock()
	}
}

// generateSessionID creates a unique session identifier
func generateSessionID() (string, error) {
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return hex.EncodeToString(bytes), nil
}

// SetSession stores a temp directory for a session
func (s *sessionStore) SetSession(sessionID, tempDir string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.sessions[sessionID] = sessionEntry{
		tempDir:   tempDir,
		createdAt: time.Now(),
	}
}

// GetSession retrieves the temp directory for a session
func (s *sessionStore) GetSession(sessionID string) (string, bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	entry, ok := s.sessions[sessionID]
	return entry.tempDir, ok
}

// DeleteSession removes a session and cleans up its temp directory
func (s *sessionStore) DeleteSession(sessionID string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if entry, ok := s.sessions[sessionID]; ok {
		os.RemoveAll(entry.tempDir)
	}
	delete(s.sessions, sessionID)
}

// EnableCORS adds the necessary headers to allow cross-origin requests
func EnableCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// ClusterAndGenerateHandler processes uploaded images and generates clusters
func ClusterAndGenerateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		respondWithError(w, http.StatusBadRequest, "Failed to parse form data")
		return
	}

	// Generate unique session ID for this request
	sessionID, err := generateSessionID()
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to generate session ID")
		return
	}

	tempDir, err := os.MkdirTemp("", "imagecluster_*")
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to create temporary directory")
		return
	}

	// Store the session -> tempDir mapping
	store.SetSession(sessionID, tempDir)

	uploadedImages := []models.UploadedImage{}
	var uploadErrors []string
	files := r.MultipartForm.File["images"]
	for _, fileHeader := range files {
		file, err := fileHeader.Open()
		if err != nil {
			log.Printf("Failed to open file %s: %v", fileHeader.Filename, err)
			uploadErrors = append(uploadErrors, fmt.Sprintf("failed to open %s", fileHeader.Filename))
			continue
		}

		data, err := io.ReadAll(file)
		file.Close() // Close immediately after reading instead of deferring in loop
		if err != nil {
			log.Printf("Failed to read file %s: %v", fileHeader.Filename, err)
			uploadErrors = append(uploadErrors, fmt.Sprintf("failed to read %s", fileHeader.Filename))
			continue
		}

		sanitizedFilename := utils.SanitizeFilename(fileHeader.Filename)
		uploadedImages = append(uploadedImages, models.UploadedImage{
			Filename: sanitizedFilename,
			Data:     data,
		})
	}

	if len(uploadedImages) == 0 {
		errorMsg := "No valid images uploaded"
		if len(uploadErrors) > 0 {
			errorMsg = fmt.Sprintf("No valid images uploaded. Errors: %s", strings.Join(uploadErrors, "; "))
		}
		respondWithError(w, http.StatusBadRequest, errorMsg)
		return
	}

	// Log partial failures if some images failed but others succeeded
	if len(uploadErrors) > 0 {
		log.Printf("Warning: %d of %d files failed to upload: %s",
			len(uploadErrors), len(files), strings.Join(uploadErrors, "; "))
	}

	// Parse cluster size settings from form data
	minClusterSize := 3 // default
	maxClusterSize := 6 // default
	if minStr := r.FormValue("minClusterSize"); minStr != "" {
		if val, err := strconv.Atoi(minStr); err == nil && val >= 2 {
			minClusterSize = val
		}
	}
	if maxStr := r.FormValue("maxClusterSize"); maxStr != "" {
		if val, err := strconv.Atoi(maxStr); err == nil && val >= minClusterSize {
			maxClusterSize = val
		}
	}

	imagecluster, err := workflow.NewImageCluster(minClusterSize, maxClusterSize, tempDir)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to initialize application")
		return
	}

	_, _, err = imagecluster.Run(uploadedImages)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"sessionId": sessionID,
		"filePath":  filepath.Join(tempDir, "clusters.html"),
	})
}

// ViewHandler serves the generated HTML file at /view
// Accepts session ID via query parameter: /api/view?session=<sessionID>
func ViewHandler(w http.ResponseWriter, r *http.Request) {
	sessionID := r.URL.Query().Get("session")
	if sessionID == "" {
		http.Error(w, "Missing session parameter", http.StatusBadRequest)
		return
	}

	tempDir, ok := store.GetSession(sessionID)
	if !ok {
		http.Error(w, "Invalid or expired session", http.StatusNotFound)
		return
	}

	htmlFilePath := filepath.Join(tempDir, "clusters.html")
	if _, err := os.Stat(htmlFilePath); os.IsNotExist(err) {
		http.Error(w, "No HTML file available", http.StatusNotFound)
		return
	}

	http.ServeFile(w, r, htmlFilePath)
}

// ImageHandler serves images from the temporary directory
// Accepts session ID via query parameter: /api/image/{imageName}?session=<sessionID>
func ImageHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	imageName := utils.SanitizeFilename(vars["imageName"])

	sessionID := r.URL.Query().Get("session")
	if sessionID == "" {
		http.Error(w, "Missing session parameter", http.StatusBadRequest)
		return
	}

	tempDir, ok := store.GetSession(sessionID)
	if !ok {
		http.Error(w, "Invalid or expired session", http.StatusNotFound)
		return
	}

	imagesDir := filepath.Join(tempDir, "images")
	imagePath := filepath.Join(imagesDir, imageName)

	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		log.Printf("Image not found: %s", imagePath)
		http.Error(w, "Image not found", http.StatusNotFound)
		return
	}

	ext := strings.ToLower(filepath.Ext(imageName))
	contentType := "image/jpeg"
	switch ext {
	case ".png":
		contentType = "image/png"
	case ".gif":
		contentType = "image/gif"
	case ".webp":
		contentType = "image/webp"
	}
	w.Header().Set("Content-Type", contentType)

	http.ServeFile(w, r, imagePath)
}

// respondWithError sends an error response in JSON format.
func respondWithError(w http.ResponseWriter, code int, message string) {
	respondWithJSON(w, code, map[string]interface{}{
		"success": false,
		"error":   message,
	})
}

// respondWithJSON sends a response in JSON format.
func respondWithJSON(w http.ResponseWriter, code int, payload map[string]interface{}) {
	response, err := json.Marshal(payload)
	if err != nil {
		log.Printf("Error marshaling response JSON: %v", err)
		http.Error(w, "Failed to marshal response", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(response)
}

// ServeHTTP handles all requests by attempting to serve static files first,
// and falling back to serving index.html for any non-file routes
func (h SpaHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Serve /view route
	if r.URL.Path == "/view" {
		ViewHandler(w, r)
		return
	}

	// Handle all other routes
	path := filepath.Join(h.StaticPath, r.URL.Path)
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		http.ServeFile(w, r, filepath.Join(h.StaticPath, h.IndexPath))
		return
	}
	http.FileServer(http.Dir(h.StaticPath)).ServeHTTP(w, r)
}

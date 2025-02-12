package handlers

import (
	"encoding/json"
	"imageclust/internal/models"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/gorilla/mux"
	"imageclust/internal/config"
	"imageclust/internal/utils"
	"imageclust/internal/workflow"
)

// SpaHandler implements the http.Handler interface for serving a Single Page Application
type SpaHandler struct {
	StaticPath string
	IndexPath  string
}

// Handler encapsulates the necessary credentials and dependencies.
type Handler struct {
	ProfileID string
	AuthToken string
}

// NewHandler initializes and returns a new Handler instance.
func NewHandler() *Handler {
	return &Handler{}
}

// Global variables to manage the current temp directory
var (
	currentTempDir string
	tempDirMutex   sync.RWMutex
)

// SetTempDir sets the current temp directory in a thread-safe way.
func SetTempDir(dir string) {
	tempDirMutex.Lock()
	defer tempDirMutex.Unlock()
	currentTempDir = dir
}

// GetTempDir gets the current temp directory in a thread-safe way.
func GetTempDir() string {
	tempDirMutex.RLock()
	defer tempDirMutex.RUnlock()
	return currentTempDir
}

// EnableCORS adds the necessary headers to allow cross-origin requests
func EnableCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Allow requests from any origin
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		// Handle preflight requests for CORS (OPTIONS method)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (h *Handler) ClusterAndGenerateHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("Route /cluster_and_generate was called")

	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(32 << 20) // 32MB max memory
	if err != nil {
		log.Printf("Error parsing multipart form: %v", err)
		http.Error(w, "Failed to parse multipart form data", http.StatusBadRequest)
		return
	}

	// Extract configurations from form fields
	appConfig, err := config.ExtractConfigurations(r)
	if err != nil {
		log.Printf("Error in configurations: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Override cluster sizes with hardcoded values
	appConfig.MinClusterSize = 3 // Hardcoded minimum cluster size
	appConfig.MaxClusterSize = 6 // Hardcoded maximum cluster size

	log.Printf("Using hardcoded cluster sizes - Min: %d, Max: %d", appConfig.MinClusterSize, appConfig.MaxClusterSize)

	// Set the Handler's ProfileID and AuthToken for use in PublishHandler
	h.ProfileID = appConfig.ProfileID
	h.AuthToken = appConfig.AuthToken

	// Create a temporary directory
	tempDir, err := os.MkdirTemp("", "imagecluster_*")
	if err != nil {
		log.Printf("Failed to create temporary directory: %v", err)
		http.Error(w, "Failed to create temporary directory.", http.StatusInternalServerError)
		return
	}
	log.Printf("Temporary directory created at: %s", tempDir)

	// Set the temp directory globally for image serving
	SetTempDir(tempDir)

	// Process uploaded images
	uploadedImages := []models.UploadedImage{}
	files := r.MultipartForm.File["images"]
	for _, fileHeader := range files {
		file, err := fileHeader.Open()
		if err != nil {
			log.Printf("Error opening uploaded file: %v", err)
			continue
		}
		defer file.Close()

		data, err := io.ReadAll(file)
		if err != nil {
			log.Printf("Error reading uploaded file: %v", err)
			continue
		}

		sanitizedFilename := utils.SanitizeFilename(fileHeader.Filename)
		uploadedImages = append(uploadedImages, models.UploadedImage{
			Filename: sanitizedFilename,
			Data:     data,
		})
	}

	if len(uploadedImages) == 0 {
		http.Error(w, "No valid images uploaded", http.StatusBadRequest)
		return
	}

	// Initialize imagecluster
	productSetter, err := workflow.NewImageCluster(
		appConfig.MinClusterSize,
		appConfig.MaxClusterSize,
		tempDir,
	)
	if err != nil {
		log.Printf("Failed to initialize ProductSetter: %v", err)
		http.Error(w, "Failed to initialize application.", http.StatusInternalServerError)
		return
	}

	// Run the main workflow
	_, htmlFilePath, err := productSetter.Run(uploadedImages)
	if err != nil {
		log.Printf("Error during ProductSetter run: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Log the location of the generated HTML file
	log.Printf("HTML file generated at: %s", htmlFilePath)

	// Redirect the client to the /view endpoint to display the HTML
	http.Redirect(w, r, "/view", http.StatusSeeOther)
}

// ViewHandler serves the generated HTML file at /view
func (h *Handler) ViewHandler(w http.ResponseWriter, r *http.Request) {
	tempDir := GetTempDir()
	if tempDir == "" {
		http.Error(w, "No HTML file available", http.StatusNotFound)
		return
	}
	htmlFilePath := filepath.Join(tempDir, "clustered_fashion_items.html")
	http.ServeFile(w, r, htmlFilePath)
}

// ImageHandler serves images directly from tempDir/images/
func (h *Handler) ImageHandler(w http.ResponseWriter, r *http.Request) {
	tempDir := GetTempDir()
	if tempDir == "" {
		http.Error(w, "No images available", http.StatusNotFound)
		return
	}
	vars := mux.Vars(r)
	imageName := vars["imageName"]

	// Construct the path to the images subdirectory
	imagesDir := filepath.Join(tempDir, "images")
	imagePath := filepath.Join(imagesDir, imageName)

	// Check if the image exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		http.Error(w, "Image not found", http.StatusNotFound)
		return
	}

	// Serve the image file
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
	// Get the absolute path to prevent directory traversal
	path, err := filepath.Abs(r.URL.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Prepend the static file path
	path = filepath.Join(h.StaticPath, path)

	// Check if the file exists
	_, err = os.Stat(path)
	if os.IsNotExist(err) {
		// File doesn't exist, serve index.html
		indexPath := filepath.Join(h.StaticPath, h.IndexPath)
		http.ServeFile(w, r, indexPath)
		return
	} else if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Serve the static file
	http.FileServer(http.Dir(h.StaticPath)).ServeHTTP(w, r)
}

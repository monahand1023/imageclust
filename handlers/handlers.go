// Package handlers/handlers.go
package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/gorilla/mux"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"ProductSetter/productsetter"
	"ProductSetter/utils"
)

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

// PublishRequest represents the expected JSON payload for the publish endpoint.
type PublishRequest struct {
	ClusterID           string   `json:"cluster_id"`
	Title               string   `json:"title"`
	Description         string   `json:"description"`
	ProductReferenceIDs []string `json:"product_reference_ids"`
}

// PublishResponse represents the JSON response structure.
type PublishResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

// APIURL is the endpoint for publishing clusters.
// Replace this with the actual API URL as needed.
const APIURL = "https://qa-api-gateway.rewardstyle.com/api/pub/v1/shops/create_shop_product_collection"

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

// ClusterAndGenerateHandler handles the clustering and response generation.
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
	appConfig, err := productsetter.ExtractConfigurations(r)
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
	tempDir, err := os.MkdirTemp("", "productsetter_*")
	if err != nil {
		log.Printf("Failed to create temporary directory: %v", err)
		http.Error(w, "Failed to create temporary directory.", http.StatusInternalServerError)
		return
	}
	log.Printf("Temporary directory created at: %s", tempDir)

	// Set the temp directory globally for image serving
	SetTempDir(tempDir)

	// Initialize ProductSetter
	productSetter, err := productsetter.NewProductSetter(
		appConfig.ProfileID,
		appConfig.AuthToken,
		appConfig.NumberOfDaysLimit,
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
	clusterDetails, _, err := productSetter.Run() // Ignoring htmlOutputPath to fix the unused variable error
	if err != nil {
		log.Printf("Error during ProductSetter run: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Generate the HTML output and save it to tempDir
	htmlFilePath, err := utils.GenerateHTMLOutput(clusterDetails, tempDir)
	if err != nil {
		log.Printf("Error during HTML generation: %v", err)
		http.Error(w, "Failed to generate HTML output.", http.StatusInternalServerError)
		return
	}

	// Log the location of the generated HTML file
	log.Printf("HTML file generated at: %s", htmlFilePath)

	// Redirect the client to the /view endpoint to display the HTML
	http.Redirect(w, r, "/view", http.StatusSeeOther)
}

// PublishHandler handles the publishing of a cluster.
func (h *Handler) PublishHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("Route /publish was called")

	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// Parse JSON body
	var req PublishRequest
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("Error reading request body: %v", err)
		respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}
	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {

		}
	}(r.Body)

	err = json.Unmarshal(body, &req)
	if err != nil {
		log.Printf("Error unmarshaling JSON: %v", err)
		respondWithError(w, http.StatusBadRequest, "Invalid JSON format")
		return
	}

	if req.ClusterID == "" || req.Title == "" || req.Description == "" || len(req.ProductReferenceIDs) == 0 {
		respondWithError(w, http.StatusBadRequest, "Missing required fields in request")
		return
	}

	// Retrieve profileID and authToken from Handler's fields
	profileID := h.ProfileID
	authToken := h.AuthToken
	log.Printf("Retrieved from Handler - PROFILE_ID: %s, AUTH_TOKEN: %s", profileID, authToken)

	if profileID == "" || authToken == "" {
		respondWithError(w, http.StatusBadRequest, "ProfileID or AuthToken not set. Please invoke /cluster_and_generate first.")
		return
	}

	// Construct the payload for the publish API
	payload := map[string]interface{}{
		"add_product_reference_ids": req.ProductReferenceIDs,
		"subtype":                   "",
		"title":                     req.Title,
		"description":               req.Description,
		"attributes":                map[string]interface{}{}, // Empty JSON object
		"profile_id":                profileID,
	}

	// Marshal payload to JSON for logging
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Printf("Error marshaling payload: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to prepare payload")
		return
	}

	// Log the equivalent curl command
	curlCommand := fmt.Sprintf(`
curl -X POST "%s" \
     -H "Authorization: Bearer %s" \
     -H "Content-Type: application/json" \
     -d '%s'
`, APIURL, authToken, string(payloadBytes))
	log.Println("Equivalent curl command:\n" + curlCommand)

	// Create the HTTP request
	reqPublish, err := http.NewRequest("POST", APIURL, bytes.NewBuffer(payloadBytes))
	if err != nil {
		log.Printf("Error creating HTTP request: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to create request to external API")
		return
	}

	// Set headers
	reqPublish.Header.Set("Authorization", "Bearer "+authToken)
	reqPublish.Header.Set("Content-Type", "application/json")

	// Create the HTTP client with a timeout
	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	// Perform the HTTP POST request
	resp, err := client.Do(reqPublish)
	if err != nil {
		log.Printf("Error performing HTTP request: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to perform request to external API")
		return
	}
	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {

		}
	}(resp.Body)

	// Read the response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Error reading external API response: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to read response from external API")
		return
	}

	// Handle the response based on status code
	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusCreated {
		log.Printf("Shop product collection created successfully for cluster: %s", req.Title)
		response := PublishResponse{
			Success: true,
			Message: "Shop product collection created successfully",
		}
		respondWithJSON(w, http.StatusOK, response)
	} else {
		log.Printf("Failed to create shop product collection. Status code: %d", resp.StatusCode)
		log.Printf("Response: %s", string(respBody))
		response := PublishResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to create shop product collection. Status code: %d", resp.StatusCode),
		}
		respondWithJSON(w, resp.StatusCode, response)
	}
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
	respondWithJSON(w, code, PublishResponse{
		Success: false,
		Error:   message,
	})
}

// respondWithJSON sends a response in JSON format.
func respondWithJSON(w http.ResponseWriter, code int, payload PublishResponse) {
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

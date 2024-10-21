// handlers/handlers.go
package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"ProductSetter/productsetter"
)

// ClusterAndGenerateHandler handles the clustering and response generation.
func ClusterAndGenerateHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("Route /cluster_and_generate was called")

	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// Parse the form data
	err := r.ParseMultipartForm(32 << 20) // 32MB max memory
	if err != nil {
		log.Printf("Error parsing multipart form: %v", err)
		http.Error(w, "Failed to parse multipart form data", http.StatusBadRequest)
		return
	}

	// Extract configurations from form fields using productsetter package
	appConfig, err := productsetter.ExtractConfigurations(r)
	if err != nil {
		log.Printf("Error in configurations: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Create a temporary directory
	tempDir, err := os.MkdirTemp("", "productsetter_*")
	if err != nil {
		log.Printf("Failed to create temporary directory: %v", err)
		http.Error(w, "Failed to create temporary directory.", http.StatusInternalServerError)
		return
	}
	defer os.RemoveAll(tempDir) // Clean up after processing

	log.Printf("Temporary directory created at: %s", tempDir)

	// Initialize ProductSetter using productsetter package
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

	// Run the main workflow and get clusters and HTML path
	clusters, htmlOutputPath, err := productSetter.Run()
	if err != nil {
		log.Printf("Error during ProductSetter run: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Prepare ClusterDetails for HTML generation (if needed)
	// Depending on how Run is structured, this might already be done within Run
	// If Run already calls PrepareClusterDetails, you can skip this step
	// Otherwise, ensure clusterDetails are prepared here
	// clusterDetails := productSetter.PrepareClusterDetails(clusters, productSetter.FetchProductDetails())

	// Prepare the response payload
	responsePayload := map[string]interface{}{
		"clusters":  clusters,       // Adjust based on what Run returns
		"html_path": htmlOutputPath, // Adjust based on what Run returns
	}

	// Send the response
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(responsePayload)
	if err != nil {
		log.Printf("Failed to encode response: %v", err)
		http.Error(w, "Failed to encode response.", http.StatusInternalServerError)
		return
	}
}

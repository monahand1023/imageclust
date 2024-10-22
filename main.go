package main

import (
	"log"
	"net/http"
	"path/filepath"
	"sync"

	"ProductSetter/handlers"
	"github.com/gorilla/mux"
)

var (
	currentTempDir string
	tempDirMutex   sync.RWMutex
)

func main() {
	// Initialize the router using Gorilla Mux
	router := mux.NewRouter()

	// Apply the CORS middleware
	router.Use(handlers.EnableCORS)

	// Initialize the handler
	h := handlers.NewHandler()

	// Register handlers for various routes
	router.HandleFunc("/cluster_and_generate", h.ClusterAndGenerateHandler).Methods("POST")
	router.HandleFunc("/publish", h.PublishHandler).Methods("POST")
	router.HandleFunc("/", htmlFormHandler).Methods("GET") // Serve the HTML form on the root path
	router.HandleFunc("/view", h.ViewHandler).Methods("GET")
	router.HandleFunc("/image/{imageName}", h.ImageHandler).Methods("GET")

	// Serve images from the current temporary directory
	router.PathPrefix("/images/").Handler(http.HandlerFunc(serveImages))

	// Define the server address and port
	serverAddress := ":8080"

	// Log the server start
	log.Printf("Starting server on %s", serverAddress)

	// Start the HTTP server
	err := http.ListenAndServe(serverAddress, router)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// serveImages dynamically serves images from the current temp directory.
func serveImages(w http.ResponseWriter, r *http.Request) {
	tempDirMutex.RLock()
	defer tempDirMutex.RUnlock()

	if currentTempDir == "" {
		http.Error(w, "No image directory available", http.StatusNotFound)
		return
	}

	imagePath := r.URL.Path[len("/images/"):]
	fullPath := filepath.Join(currentTempDir, "images", imagePath)
	http.ServeFile(w, r, fullPath)
}

// htmlFormHandler serves the HTML form on the root path.
func htmlFormHandler(w http.ResponseWriter, r *http.Request) {
	form := `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Product Clustering</title>
</head>
<body>
    <h1>Cluster Products</h1>
    <form action="/cluster_and_generate" method="post" enctype="multipart/form-data">
        <label for="profile_id">Profile ID:</label><br>
        <input type="text" id="profile_id" name="profile_id" required><br><br>

        <label for="auth_token">Auth Token:</label><br>
        <input type="text" id="auth_token" name="auth_token" required><br><br>

        <label for="number_of_days_limit">Number of Days Limit:</label><br>
        <input type="number" id="number_of_days_limit" name="number_of_days_limit" min="1" value="30" required><br><br>

        <input type="submit" value="Cluster and Generate">
    </form>
</body>
</html>
	`
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(form))
}

package main

import (
	"log"
	"net/http"
	"path/filepath"
	"sync"

	"github.com/gorilla/mux"
	"imageclust/internal/handlers"
)

var (
	currentTempDir string
	tempDirMutex   sync.RWMutex
)

func main() {
	router := mux.NewRouter()
	router.Use(handlers.EnableCORS)

	h := handlers.NewHandler()

	// API routes
	router.HandleFunc("/api/cluster", h.ClusterAndGenerateHandler).Methods("POST")
	router.HandleFunc("/api/image/{imageName}", h.ImageHandler).Methods("GET")

	// Serve frontend
	spa := handlers.SpaHandler{StaticPath: "frontend/build", IndexPath: "index.html"}
	router.PathPrefix("/").Handler(spa)

	serverAddress := ":8080"
	log.Printf("Starting server on %s", serverAddress)

	err := http.ListenAndServe(serverAddress, router)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

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

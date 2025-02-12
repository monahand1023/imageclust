package main

import (
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"imageclust/internal/handlers"
)

func main() {
	router := mux.NewRouter()
	router.Use(handlers.EnableCORS)

	// API routes
	router.HandleFunc("/api/cluster", handlers.ClusterAndGenerateHandler).Methods("POST")
	router.HandleFunc("/api/image/{imageName}", handlers.ImageHandler).Methods("GET")
	router.HandleFunc("/view", handlers.ViewHandler).Methods("GET")

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

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
	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.HandleFunc("/cluster", handlers.ClusterAndGenerateHandler).Methods("POST")
	apiRouter.HandleFunc("/image/{imageName:.*}", handlers.ImageHandler).Methods("GET")

	// View route
	router.HandleFunc("/view", handlers.ViewHandler).Methods("GET")

	// Serve frontend static files
	router.PathPrefix("/").Handler(http.FileServer(http.Dir("frontend/build")))

	serverAddress := ":8080"
	log.Printf("Starting server on %s", serverAddress)
	log.Printf("View results at http://localhost%s/view", serverAddress)

	err := http.ListenAndServe(serverAddress, router)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

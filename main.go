package main

import (
	"github.com/gorilla/mux"
	"imageclust/internal/handlers"
	"log"
	"net/http"
)

func main() {
	router := mux.NewRouter()
	router.Use(handlers.EnableCORS)

	// API routes
	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.HandleFunc("/cluster", handlers.ClusterAndGenerateHandler).Methods("POST")
	apiRouter.HandleFunc("/image/{imageName:.*}", handlers.ImageHandler).Methods("GET")
	apiRouter.HandleFunc("/view", handlers.ViewHandler).Methods("GET")

	// Serve static files
	spa := handlers.SpaHandler{StaticPath: "frontend/build", IndexPath: "index.html"}
	router.PathPrefix("/").Handler(spa)

	serverAddress := ":8080"
	log.Printf("Starting server on %s", serverAddress)
	err := http.ListenAndServe(serverAddress, router)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

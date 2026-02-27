package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"imageclust/internal/handlers"
)

func main() {
	router := mux.NewRouter()
	router.Use(handlers.EnableCORS)

	// Health check endpoint
	router.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}).Methods("GET")

	// API routes
	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.HandleFunc("/cluster", handlers.ClusterAndGenerateHandler).Methods("POST")
	apiRouter.HandleFunc("/image/{imageName:.*}", handlers.ImageHandler).Methods("GET")
	apiRouter.HandleFunc("/view", handlers.ViewHandler).Methods("GET")

	// Serve static files
	spa := handlers.SpaHandler{StaticPath: "frontend/build", IndexPath: "index.html"}
	router.PathPrefix("/").Handler(spa)

	serverAddress := ":8080"
	srv := &http.Server{
		Addr:         serverAddress,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in a goroutine
	go func() {
		log.Printf("Starting server on %s", serverAddress)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal for graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited gracefully")
}

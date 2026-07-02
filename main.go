package main

import (
	"context"
	"errors"
	"imageclust/internal/clip"
	"imageclust/internal/handlers"
	"imageclust/internal/ollama"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/gorilla/mux"
)

func main() {
	// --- ONNX Runtime initialization ---
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		if runtime.GOOS == "darwin" {
			libPath = "/opt/homebrew/lib/libonnxruntime.dylib"
		}
		// On Linux the library is resolved from LD_LIBRARY_PATH; pass empty string.
	}
	if err := clip.InitONNXRuntime(libPath); err != nil {
		log.Fatalf("failed to initialize ONNX Runtime: %v", err)
	}

	// --- CLIP model ---
	modelPath := os.Getenv("CLIP_MODEL_PATH")
	if modelPath == "" {
		modelPath = "models/clip-vit-large-patch14/vision_model.onnx"
	}
	log.Printf("Loading CLIP model from %s", modelPath)
	clipModel, err := clip.LoadModel(modelPath)
	if err != nil {
		log.Fatalf("failed to load CLIP model: %v\nRun scripts/download_model.sh to download it.", err)
	}
	defer clipModel.Close()
	log.Printf("CLIP model loaded — embedding dim: %d", clipModel.EmbeddingDim)

	// --- Ollama client ---
	ollamaClient := ollama.NewClient(os.Getenv("OLLAMA_HOST"), os.Getenv("OLLAMA_MODEL"))
	log.Printf("Ollama client ready")

	// --- HTTP routing ---
	h := handlers.New(clipModel, ollamaClient, modelPath)
	defer h.Close()

	router := mux.NewRouter()
	router.Use(handlers.EnableCORS)

	router.HandleFunc("/health", h.HealthHandler).Methods(http.MethodGet)

	api := router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/cluster", h.ClusterAndGenerate).Methods(http.MethodPost, http.MethodOptions)
	api.HandleFunc("/image/{imageName:.*}", h.ServeImage).Methods(http.MethodGet)
	api.HandleFunc("/export", h.ExportZip).Methods(http.MethodGet)

	spa := handlers.SpaHandler{StaticPath: "frontend/dist", IndexPath: "index.html"}
	router.PathPrefix("/").Handler(spa)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	server := &http.Server{
		Addr:    ":" + port,
		Handler: router,
		// The cluster endpoint runs CLIP + Ollama inference and can
		// legitimately take minutes, so no WriteTimeout is set.
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}

	// Shut down gracefully on SIGINT/SIGTERM so deferred cleanup
	// (model Close, session cleanup stop) actually runs.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		log.Printf("Server listening on :%s", port)
		errCh <- server.ListenAndServe()
	}()

	select {
	case err := <-errCh:
		log.Fatalf("server error: %v", err)
	case <-ctx.Done():
		log.Println("shutting down…")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
			log.Printf("shutdown error: %v", err)
		}
	}
}

package main

import (
	"imageclust/internal/clip"
	"imageclust/internal/handlers"
	"imageclust/internal/ollama"
	"log"
	"net/http"
	"os"
	"runtime"

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
	ollamaHost := os.Getenv("OLLAMA_HOST")
	ollamaModel := os.Getenv("OLLAMA_MODEL")
	ollamaClient, err := ollama.NewClient(ollamaHost, ollamaModel)
	if err != nil {
		log.Fatalf("failed to create Ollama client: %v", err)
	}
	log.Printf("Ollama client ready")

	// --- HTTP routing ---
	h := handlers.NewWithModelPath(clipModel, ollamaClient, modelPath)

	router := mux.NewRouter()
	router.Use(handlers.EnableCORS)

	router.HandleFunc("/health", h.HealthHandler).Methods(http.MethodGet)

	api := router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/cluster", h.ClusterAndGenerate).Methods(http.MethodPost, http.MethodOptions)
	api.HandleFunc("/image/{imageName:.*}", h.ServeImage).Methods(http.MethodGet)

	spa := handlers.SpaHandler{StaticPath: "frontend/dist", IndexPath: "index.html"}
	router.PathPrefix("/").Handler(spa)

	addr := ":8080"
	log.Printf("Server listening on %s", addr)
	if err := http.ListenAndServe(addr, router); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

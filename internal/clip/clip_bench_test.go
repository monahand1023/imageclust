package clip

import (
	"os"
	"testing"
)

// BenchmarkEmbed measures CLIP inference time per image.
// Requires CLIP_MODEL_PATH and ONNXRUNTIME_LIB_PATH env vars (or their defaults).
// Run with: go test -bench=BenchmarkEmbed -benchtime=5x ./internal/clip/
func BenchmarkEmbed(b *testing.B) {
	modelPath := os.Getenv("CLIP_MODEL_PATH")
	if modelPath == "" {
		modelPath = "../../models/clip-vit-large-patch14/vision_model.onnx"
	}
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("CLIP model not found at %s — set CLIP_MODEL_PATH to run this benchmark", modelPath)
	}

	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		libPath = "/opt/homebrew/lib/libonnxruntime.dylib"
	}
	if err := InitONNXRuntime(libPath); err != nil {
		b.Fatalf("InitONNXRuntime: %v", err)
	}

	model, err := LoadModel(modelPath)
	if err != nil {
		b.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()

	// Use a real test image from the standard test data if present, else a temp file.
	imgPath := os.Getenv("BENCH_IMAGE_PATH")
	if imgPath == "" {
		imgPath = "/tmp/imageclust-test/img_1.jpg"
	}
	if _, err := os.Stat(imgPath); err != nil {
		b.Skipf("test image not found at %s — set BENCH_IMAGE_PATH to run this benchmark", imgPath)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		emb, err := model.Embed(imgPath)
		if err != nil {
			b.Fatalf("Embed: %v", err)
		}
		if len(emb) != embeddingDim {
			b.Fatalf("unexpected embedding dim: %d", len(emb))
		}
	}
}

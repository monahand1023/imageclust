#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/models/clip-vit-large-patch14"
MODEL_FILE="$MODEL_DIR/vision_model.onnx"
MODEL_URL="https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main/onnx/vision_model.onnx"

echo "==> Downloading CLIP ViT-L/14 vision encoder ONNX (~350MB)"
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
  echo "    Model already exists at $MODEL_FILE — skipping download."
else
  curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
  echo "    Saved to $MODEL_FILE"
fi

echo ""
echo "==> Prerequisites checklist:"
echo ""
echo "  1. ONNX Runtime (required for inference):"
echo "       brew install onnxruntime"
echo "     Default library path: /opt/homebrew/lib/libonnxruntime.dylib"
echo "     Override with: export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.dylib"
echo ""
echo "  2. Ollama (required for cluster title generation):"
echo "       brew install ollama"
echo "       ollama serve   # keep running in a separate terminal"
echo "       ollama pull llama3.2-vision:11b"
echo "     Default endpoint: http://localhost:11434"
echo "     Override with: export OLLAMA_HOST=http://localhost:11434"
echo "     Override model: export OLLAMA_MODEL=llama3.2-vision:11b"
echo ""
echo "  3. Build and run:"
echo "       go build -o imageclust ."
echo "       ./imageclust"
echo ""
echo "Done."

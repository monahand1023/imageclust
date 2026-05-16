---
name: project-architecture
description: imageclust pipeline architecture after the CLIP/Ollama migration
metadata:
  type: project
---

# imageclust architecture (post-migration)

**Why:** Replaced AWS Rekognition + ResNet50 ONNX (GoCV/CPU) + Claude via Bedrock with local stack for better quality and no cloud costs.

**Pipeline:**
1. Upload images → Go HTTP server (gorilla/mux)
2. CLIP ViT-L/14 ONNX (yalue/onnxruntime_go) generates 1024-dim L2-normalized embeddings
   - Model loaded once at startup from `models/clip-vit-large-patch14/vision_model.onnx`
   - Pure Go image preprocessing (disintegration/imaging — no GoCV)
   - `internal/clip/clip.go` — AdvancedSession with pre-allocated tensors, mutex-serialized
3. Ward hierarchical clustering with min/max size constraints (unchanged — `internal/clustering/`)
4. Ollama REST API (direct net/http, no SDK) → llama3.2-vision:11b generates title + catchy phrase per cluster
   - Sends up to 3 representative images (closest to centroid) per cluster
   - `internal/ollama/ollama.go`
5. API returns JSON `{status, sessionId, clusters:[{id, title, catchy_phrase, images:[]}]}`
6. React frontend renders clusters inline (no server-generated HTML)
   - `frontend/src/components/ClusterResults.jsx` — grid of clusters with image thumbnails
   - Images served via `GET /api/image/{name}?session=X`

**Key files:**
- `internal/clip/clip.go` — CLIP inference
- `internal/ollama/ollama.go` — Ollama REST client
- `internal/workflow/workflow.go` — orchestration
- `internal/handlers/handlers.go` — HTTP layer
- `main.go` — startup: loads model, creates Ollama client, wires router
- `scripts/download_model.sh` — downloads vision_model.onnx from HuggingFace

**Environment variables:**
- `ONNXRUNTIME_LIB_PATH` — default `/opt/homebrew/lib/libonnxruntime.dylib` on macOS
- `CLIP_MODEL_PATH` — default `models/clip-vit-large-patch14/vision_model.onnx`
- `OLLAMA_HOST` — default `http://localhost:11434`
- `OLLAMA_MODEL` — default `llama3.2-vision:11b`

**How to apply:** When working on this codebase, model/Ollama setup is a prerequisite. Direct users to `scripts/download_model.sh` and `brew install onnxruntime` before building.

# imageclust

Semantic image clustering that runs entirely on your local machine. Upload a collection of photos, get back labeled groups organized by what they're *about* — not just visual similarity.

Clusters 20 images in ~50 seconds on an M4 Mac Mini (no GPU, no cloud).

---

## How it works

```
Upload images
    ↓
CLIP ViT-L/14 ONNX  →  768-dim semantic embeddings per image
    ↓
Ward hierarchical clustering  →  min/max size-constrained groups
    ↓
Ollama vision LLM  →  title + catchy phrase per cluster (3 rep images sent)
    ↓
JSON API  →  React frontend renders inline
```

**Why CLIP over ResNet/DINOv2:** CLIP's contrastive text-image training produces a semantically-organized embedding space — images of the same *concept* cluster together even if they look different visually. ResNet produces visual similarity; DINOv2 is good for re-identification. CLIP is the right choice for "what is this about" clustering.

**Why Ward over HDBSCAN:** Ward guarantees every image lands in a cluster and supports hard min/max size constraints. HDBSCAN produces noise points (unassigned images) and can't enforce a max cluster size.

---

## Prerequisites

**macOS:**
```bash
brew install onnxruntime ollama
ollama pull llava:7b          # 4.7 GB vision model
bash scripts/download_model.sh # ~1.2 GB CLIP model
```

**Linux:** Download ONNX Runtime from [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases) (v1.20.1, `linux-x64` or `linux-aarch64`), extract the `.so`, set `ONNXRUNTIME_LIB_PATH`. Then install Ollama and run the model download script.

---

## Running

```bash
go build -o imageclust .
OLLAMA_MODEL=llava:7b ./imageclust
# open http://localhost:8080
```

Environment variables (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `ONNXRUNTIME_LIB_PATH` | `/opt/homebrew/lib/libonnxruntime.dylib` | Path to ORT shared library |
| `CLIP_MODEL_PATH` | `models/clip-vit-large-patch14/vision_model.onnx` | CLIP ONNX model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2-vision:11b` | Vision-capable model name |

---

## Docker

The Dockerfile builds a self-contained image with the Go server and React frontend. Ollama must run on the host (or another container) — the default `OLLAMA_HOST` is `http://host.docker.internal:11434`.

```bash
docker build -t imageclust .
docker run -p 8080:8080 \
  -v /path/to/models:/app/models \
  imageclust
```

The CLIP model (~1.2 GB) is mounted at runtime via the volume. To bake it in instead, uncomment the `COPY models/` line in the Dockerfile.

Cross-platform builds with `--platform` work correctly (arm64 → `aarch64`, amd64 → `x64` ORT release naming).

---

## Benchmarks

Hardware: Apple M4 Pro, 14-core, 64 GB RAM. CPU inference only (no GPU/CoreML EP).

### CLIP embedding — `go test -bench=BenchmarkEmbed ./internal/clip/`

| | |
|---|---|
| Time per image | ~432 ms |
| Throughput | ~2.3 images/sec |
| Memory per call | ~3.7 MB |

Inference is serialized (one ORT session, mutex-protected). Preprocessing (decode → resize → NCHW normalization) runs in parallel across the worker pool; the ORT session is the bottleneck.

### Ward clustering — `go test -bench=. ./internal/clustering/`

| Images | Time | Memory |
|--------|------|--------|
| 10 | 0.15 ms | 0.3 MB |
| 20 | 0.54 ms | 1.2 MB |
| 50 | 3.5 ms | 7.4 MB |
| 100 | 14 ms | 29 MB |
| 200 | 55 ms | 115 MB |

O(n²) distance matrix. Negligible relative to CLIP and Ollama.

### End-to-end HTTP pipeline

| Images | Clusters | Total time | CLIP share | Ollama share |
|--------|----------|-----------|-----------|-------------|
| 10 | 2 | ~23 s | ~4 s | ~19 s |
| 20 | 4 | ~51 s | ~9 s | ~42 s |

**Bottleneck is Ollama** (~10 s/cluster, sequential). CLIP is ~17% of total time for 20 images. To speed things up: run a smaller vision model (`llava:7b` is already fast; `moondream` is faster but lower quality), or parallelize cluster title generation.

---

## Project structure

```
internal/
  clip/       — CLIP ViT-L/14 ONNX inference (AdvancedSession, mutex-serialized)
  ollama/     — Direct HTTP client for Ollama /api/generate (no SDK)
  workflow/   — Pipeline orchestration: embed → cluster → title
  clustering/ — Ward hierarchical clustering with min/max size constraints
  handlers/   — HTTP layer: multipart upload, JSON API, session store
  models/     — Shared types (UploadedImage, ClusterDetails)
  utils/      — Filename sanitization
frontend/
  src/components/
    ImageUploadForm.jsx  — Upload form with drag-and-drop
    ClusterResults.jsx   — Inline cluster grid renderer
scripts/
  download_model.sh  — Fetch CLIP ONNX from HuggingFace
  benchmark.sh       — End-to-end pipeline timing script
```

---

## API

**POST /api/cluster** — multipart form

| Field | Type | Description |
|-------|------|-------------|
| `images` | file (multiple) | Image files to cluster |
| `minClusterSize` | int | Minimum images per cluster (default 3) |
| `maxClusterSize` | int | Maximum images per cluster (default 6) |

Response:
```json
{
  "status": "success",
  "sessionId": "abc123",
  "clusters": [
    {
      "id": "Cluster-0",
      "title": "Serene rural sunset",
      "catchy_phrase": "Nature's canvas of tranquility",
      "images": ["img_0.jpg", "img_3.jpg", "img_7.jpg"]
    }
  ]
}
```

**GET /api/image/{filename}?session=\<sessionId\>** — serves an uploaded image. Sessions expire after 1 hour.

---

## License

MIT

# ---- Stage 1: Build React frontend ----
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Build Go backend ----
FROM golang:1.25-bookworm AS backend-builder
WORKDIR /app

# Download ONNX Runtime 1.26.0 (matches yalue/onnxruntime_go v1.30.1 requirement)
# ORT release names differ from Docker's TARGETARCH: amd64→x64, arm64→aarch64
ARG ONNXRUNTIME_VERSION=1.26.0
ARG TARGETOS=linux
ARG TARGETARCH
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && \
    case "${TARGETARCH}" in \
      "arm64") ORT_ARCH=aarch64 ;; \
      *)       ORT_ARCH=x64 ;; \
    esac && \
    curl -fsSL \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${TARGETOS}-${ORT_ARCH}-${ONNXRUNTIME_VERSION}.tgz" \
      -o /tmp/ort.tgz && \
    tar -xzf /tmp/ort.tgz -C /tmp && \
    cp /tmp/onnxruntime-${TARGETOS}-${ORT_ARCH}-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/local/lib/ && \
    ln -s /usr/local/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/local/lib/libonnxruntime.so && \
    ldconfig && \
    rm -rf /tmp/ort.tgz /tmp/onnxruntime-*

ENV ONNXRUNTIME_LIB_PATH=/usr/local/lib/libonnxruntime.so
ENV CGO_ENABLED=1

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o imageclust .

# ---- Stage 3: Runtime image ----
FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# ONNX Runtime shared library — Docker flattens symlinks on COPY, so
# copy only the versioned file and recreate the unversioned symlink.
ARG ONNXRUNTIME_VERSION=1.26.0
COPY --from=backend-builder /usr/local/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/local/lib/
RUN ln -sf /usr/local/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/local/lib/libonnxruntime.so && \
    ldconfig

# Application binary and frontend
COPY --from=backend-builder /app/imageclust ./
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# CLIP model — mount at runtime via volume or bake in at build time:
#   docker run -v /path/to/models:/app/models imageclust
# Or build with:
#   COPY models/ ./models/
VOLUME ["/app/models"]

ENV ONNXRUNTIME_LIB_PATH=/usr/local/lib/libonnxruntime.so
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8080
CMD ["./imageclust"]

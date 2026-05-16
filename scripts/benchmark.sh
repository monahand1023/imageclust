#!/usr/bin/env bash
# benchmark.sh — measures imageclust pipeline throughput at different image counts.
# Usage: bash scripts/benchmark.sh [server_url]
# The server must already be running.
set -euo pipefail

SERVER="${1:-http://localhost:8080}"
IMAGE_DIR="${BENCH_IMAGE_DIR:-/tmp/imageclust-test}"
MIN_SIZE=3
MAX_SIZE=8

check_deps() {
  for cmd in curl python3 bc; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd not found"; exit 1; }
  done
}

download_images() {
  local count=$1
  mkdir -p "$IMAGE_DIR"
  local existing
  existing=$(ls "$IMAGE_DIR"/img_*.jpg 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$existing" -ge "$count" ]]; then return; fi
  echo "  Downloading $count images from picsum.photos..."
  for i in $(seq 1 "$count"); do
    [[ -f "$IMAGE_DIR/img_${i}.jpg" ]] && continue
    curl -sL "https://picsum.photos/seed/${i}/640/480" -o "$IMAGE_DIR/img_${i}.jpg" &
  done
  wait
  echo "  Done."
}

wait_for_server() {
  echo -n "Waiting for server at $SERVER..."
  for i in $(seq 1 20); do
    if curl -s -o /dev/null -w "%{http_code}" "$SERVER/" | grep -q "200"; then
      echo " ready."
      return
    fi
    sleep 1
    echo -n "."
  done
  echo ""
  echo "ERROR: server not reachable at $SERVER"
  exit 1
}

run_benchmark() {
  local count=$1
  local images=()
  for i in $(seq 1 "$count"); do
    images+=("-F" "images=@$IMAGE_DIR/img_${i}.jpg")
  done

  local start end elapsed status
  start=$(python3 -c "import time; print(time.time())")
  local response
  response=$(curl -s -X POST "$SERVER/api/cluster" \
    -F "minClusterSize=$MIN_SIZE" \
    -F "maxClusterSize=$MAX_SIZE" \
    "${images[@]}")
  end=$(python3 -c "import time; print(time.time())")

  elapsed=$(python3 -c "print(f'{${end} - ${start}:.1f}')")
  status=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','error'))" 2>/dev/null || echo "error")
  local clusters
  clusters=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('clusters',[])))" 2>/dev/null || echo "?")
  local per_image
  per_image=$(python3 -c "print(f'{(${end} - ${start}) / ${count}:.2f}')")

  printf "  %-6s images → %ss total  (%ss/image)  %s clusters  [%s]\n" \
    "$count" "$elapsed" "$per_image" "$clusters" "$status"
}

# --- main ---

check_deps
wait_for_server

echo ""
echo "=== imageclust pipeline benchmark ==="
echo "Server: $SERVER"
echo "Images: $IMAGE_DIR"
echo ""

echo "Go unit benchmarks (clustering, no model required):"
go test -bench=. -benchmem -count=1 ./internal/clustering/ 2>/dev/null | grep -E "^Benchmark|^ok" | \
  awk '/^Benchmark/{printf "  %-35s %s ns/op\n", $1, $3}' || echo "  (run from repo root)"
echo ""

echo "Go CLIP benchmark (requires model):"
ONNXRUNTIME_LIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
  go test -bench=BenchmarkEmbed -benchmem -benchtime=5x -count=1 ./internal/clip/ 2>/dev/null | \
  grep -E "^Benchmark|Skipping|skip" | head -5 | \
  awk '/^Benchmark/{printf "  %-35s %s ns/op\n", $1, $3}' || true
echo ""

echo "End-to-end HTTP pipeline timings:"
download_images 50
for n in 10 20 50; do
  run_benchmark "$n"
done

echo ""
echo "Done."

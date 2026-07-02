package workflow

import "sync"

// embeddingCache is a process-wide, content-addressed cache of CLIP
// embeddings keyed by SHA-256 of the image bytes. Re-running the pipeline
// with the same images (the common loop when tuning cluster sizes) skips
// inference entirely. Cached slices are shared — callers must not mutate them.
type embeddingCache struct {
	mu      sync.Mutex
	entries map[string][]float32
	order   []string // insertion order for FIFO eviction
	cap     int
}

func newEmbeddingCache(capacity int) *embeddingCache {
	return &embeddingCache{
		entries: make(map[string][]float32, capacity),
		cap:     capacity,
	}
}

func (c *embeddingCache) get(key string) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	emb, ok := c.entries[key]
	return emb, ok
}

func (c *embeddingCache) put(key string, emb []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.entries[key]; exists {
		return
	}
	for len(c.entries) >= c.cap && len(c.order) > 0 {
		oldest := c.order[0]
		c.order = c.order[1:]
		delete(c.entries, oldest)
	}
	c.entries[key] = emb
	c.order = append(c.order, key)
}

// embCache holds ~768 floats × 4 B ≈ 3 KB per entry → ~1.5 MB at capacity.
var embCache = newEmbeddingCache(512)

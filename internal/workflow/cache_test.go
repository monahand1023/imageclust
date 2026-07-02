package workflow

import (
	"imageclust/internal/models"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// --- embeddingCache ------------------------------------------------------

func TestEmbeddingCache_PutGet(t *testing.T) {
	c := newEmbeddingCache(4)
	c.put("k1", []float32{1, 2})

	got, ok := c.get("k1")
	if !ok {
		t.Fatal("expected k1 to be cached")
	}
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Errorf("got %v, want [1 2]", got)
	}

	if _, ok := c.get("missing"); ok {
		t.Error("expected miss for unknown key")
	}
}

func TestEmbeddingCache_EvictsOldestAtCapacity(t *testing.T) {
	c := newEmbeddingCache(2)
	c.put("a", []float32{1})
	c.put("b", []float32{2})
	c.put("c", []float32{3}) // capacity 2 → "a" (oldest) must go

	if _, ok := c.get("a"); ok {
		t.Error("a should have been evicted")
	}
	if _, ok := c.get("b"); !ok {
		t.Error("b should still be cached")
	}
	if _, ok := c.get("c"); !ok {
		t.Error("c should be cached")
	}
}

// --- generateEmbeddings cache integration --------------------------------

type countingEmbedder struct {
	mu    sync.Mutex
	calls int
}

func (c *countingEmbedder) Embed(string) ([]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	return []float32{1, 0, 0, 0}, nil
}

func (c *countingEmbedder) Close() {}

func (c *countingEmbedder) callCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

func TestGenerateEmbeddings_CachesByContentHash(t *testing.T) {
	dir := t.TempDir()
	imageDir := filepath.Join(dir, "images")
	if err := os.MkdirAll(imageDir, 0755); err != nil {
		t.Fatal(err)
	}

	// Distinct content per file; prefixed with the test name so the
	// process-wide cache can't collide with other tests.
	uploads := []models.UploadedImage{
		{Filename: "a.png", Data: []byte(t.Name() + "-content-a")},
		{Filename: "b.png", Data: []byte(t.Name() + "-content-b")},
	}

	embedder := &countingEmbedder{}
	ic := &ImageCluster{TempDir: dir, ClipModel: embedder}

	items, err := ic.saveImages(uploads, imageDir)
	if err != nil {
		t.Fatalf("saveImages: %v", err)
	}

	if _, err := ic.generateEmbeddings(items); err != nil {
		t.Fatalf("first generateEmbeddings: %v", err)
	}
	if got := embedder.callCount(); got != 2 {
		t.Fatalf("first run: Embed called %d times, want 2", got)
	}

	// Same content again — every embedding must come from the cache.
	if _, err := ic.generateEmbeddings(items); err != nil {
		t.Fatalf("second generateEmbeddings: %v", err)
	}
	if got := embedder.callCount(); got != 2 {
		t.Errorf("second run: Embed called %d times total, want 2 (cache hit)", got)
	}
}

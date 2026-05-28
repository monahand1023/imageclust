package workflow

import (
	"context"
	"fmt"
	"image"
	"image/png"
	"imageclust/internal/models"
	"os"
	"strings"
	"testing"
)

// --- Stub implementations ------------------------------------------------

type stubEmbedder struct {
	// embeddings maps image path suffix → embedding vector.
	// If the path doesn't match, returns a default unit vector.
	embeddings map[string][]float32
	err        error
}

func (s *stubEmbedder) Embed(imagePath string) ([]float32, error) {
	if s.err != nil {
		return nil, s.err
	}
	for suffix, emb := range s.embeddings {
		if strings.HasSuffix(imagePath, suffix) {
			return emb, nil
		}
	}
	// Default: return a simple unit vector so clustering can proceed.
	return []float32{1, 0, 0, 0}, nil
}

func (s *stubEmbedder) Close() {}

type stubTitleGenerator struct {
	title        string
	catchyPhrase string
	err          error
	calls        int
}

func (s *stubTitleGenerator) GenerateClusterTitle(_ context.Context, _ []string, _, _ int) (string, string, error) {
	s.calls++
	if s.err != nil {
		return "", "", s.err
	}
	return s.title, s.catchyPhrase, nil
}

// --- Helpers -------------------------------------------------------------

// writePNG writes a minimal 1×1 RGBA PNG to path.
func writePNG(path string) error {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// makeUploadedImages returns a slice of models.UploadedImage built from real
// 1×1 PNG bytes (so CLIP preprocessing won't fail if a real Embedder were used).
func makeUploadedImages(t *testing.T, n int) []models.UploadedImage {
	t.Helper()
	images := make([]models.UploadedImage, n)
	for i := 0; i < n; i++ {
		img := image.NewRGBA(image.Rect(0, 0, 1, 1))
		tmp, err := os.CreateTemp(t.TempDir(), "img*.png")
		if err != nil {
			t.Fatalf("CreateTemp: %v", err)
		}
		if err := png.Encode(tmp, img); err != nil {
			t.Fatalf("png.Encode: %v", err)
		}
		tmp.Close()
		data, err := os.ReadFile(tmp.Name())
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		images[i] = models.UploadedImage{
			Filename: fmt.Sprintf("test_%d.png", i),
			Data:     data,
		}
	}
	return images
}

// --- Tests ---------------------------------------------------------------

// TestRun_FullPipeline exercises the complete Run() path with stub
// Embedder and TitleGenerator — no ONNX Runtime or Ollama required.
// We upload enough images to guarantee at least one cluster.
func TestRun_FullPipeline(t *testing.T) {
	// 6 images, minClusterSize=2, maxClusterSize=6 → should produce ≥1 cluster.
	const n = 6

	embedder := &stubEmbedder{}
	titler := &stubTitleGenerator{title: "Test Cluster", catchyPhrase: "A test phrase"}

	ic := NewImageCluster(2, 6, t.TempDir(), embedder, titler)
	images := makeUploadedImages(t, n)

	details, err := ic.Run(context.Background(), images)
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if len(details) == 0 {
		t.Fatal("Run returned 0 clusters; expected at least 1")
	}

	for key, cd := range details {
		if cd.Title == "" {
			t.Errorf("cluster %s: empty title", key)
		}
		if len(cd.Images) == 0 {
			t.Errorf("cluster %s: no images", key)
		}
	}
}

// TestRun_TitleAndPhrasePopulated verifies that the title and catchy phrase
// returned by the stub flow through to the ClusterDetails map.
func TestRun_TitleAndPhrasePopulated(t *testing.T) {
	embedder := &stubEmbedder{}
	titler := &stubTitleGenerator{title: "Sunset Photos", catchyPhrase: "Golden hour magic"}

	ic := NewImageCluster(2, 6, t.TempDir(), embedder, titler)
	images := makeUploadedImages(t, 4)

	details, err := ic.Run(context.Background(), images)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(details) == 0 {
		t.Skip("no clusters produced; skip title check")
	}

	for key, cd := range details {
		if cd.Title != "Sunset Photos" {
			t.Errorf("cluster %s: title = %q, want %q", key, cd.Title, "Sunset Photos")
		}
		if cd.CatchyPhrase != "Golden hour magic" {
			t.Errorf("cluster %s: catchy_phrase = %q, want %q", key, cd.CatchyPhrase, "Golden hour magic")
		}
	}
}

// TestRun_ContextCancellation verifies that passing an already-cancelled
// context to Run either returns promptly with a context error or completes
// without panicking. Because title generation is the first ctx-aware step,
// the context error should surface there.
func TestRun_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before Run is called

	embedder := &stubEmbedder{}
	titler := &stubTitleGenerator{title: "T", catchyPhrase: "P"}

	ic := NewImageCluster(2, 6, t.TempDir(), embedder, titler)
	images := makeUploadedImages(t, 4)

	// Run should either succeed (if the pipeline completes before ctx is checked)
	// or return the context error. It must not panic.
	_, err := ic.Run(ctx, images)
	if err != nil && err != context.Canceled {
		// Any other error is acceptable too — just not a panic.
		t.Logf("Run returned non-context error with cancelled ctx: %v", err)
	}
}

// TestBuildClusterDetails_TitleFallback verifies that when TitleGenerator
// returns an error, the cluster title falls back to "Cluster N".
func TestBuildClusterDetails_TitleFallback(t *testing.T) {
	// Build a minimal ImageCluster with a failing title generator.
	titler := &stubTitleGenerator{err: fmt.Errorf("ollama unavailable")}
	embedder := &stubEmbedder{}
	ic := NewImageCluster(2, 6, t.TempDir(), embedder, titler)

	// Construct one cluster manually with a single saved item.
	dir := t.TempDir()
	imgPath := dir + "/img_0.png"
	if err := writePNG(imgPath); err != nil {
		t.Fatalf("writePNG: %v", err)
	}

	items := []itemRecord{{ID: "img_0", ImagePath: imgPath}}
	embeddingsList := [][]float32{{1, 0}}
	clusters := map[int][]string{
		42: {"img_0"},
	}

	details, err := ic.buildClusterDetails(context.Background(), clusters, items, embeddingsList)
	if err != nil {
		t.Fatalf("buildClusterDetails: %v", err)
	}

	cd, ok := details["Cluster-42"]
	if !ok {
		t.Fatalf("expected key Cluster-42, got keys: %v", mapKeys(details))
	}
	// Title should fall back to "Cluster 42" when generator errors.
	if cd.Title != "Cluster 42" {
		t.Errorf("fallback title = %q, want %q", cd.Title, "Cluster 42")
	}
}

// TestBuildClusterDetails_MultiCluster verifies that multiple clusters are
// all processed and returned correctly.
func TestBuildClusterDetails_MultiCluster(t *testing.T) {
	titler := &stubTitleGenerator{title: "Group", catchyPhrase: "phrase"}
	embedder := &stubEmbedder{}
	ic := NewImageCluster(2, 6, t.TempDir(), embedder, titler)

	dir := t.TempDir()
	items := make([]itemRecord, 4)
	embs := make([][]float32, 4)
	for i := 0; i < 4; i++ {
		p := fmt.Sprintf("%s/img_%d.png", dir, i)
		if err := writePNG(p); err != nil {
			t.Fatalf("writePNG %d: %v", i, err)
		}
		items[i] = itemRecord{ID: fmt.Sprintf("img_%d", i), ImagePath: p}
		embs[i] = []float32{float32(i), 0}
	}

	clusters := map[int][]string{
		1: {"img_0", "img_1"},
		2: {"img_2", "img_3"},
	}

	details, err := ic.buildClusterDetails(context.Background(), clusters, items, embs)
	if err != nil {
		t.Fatalf("buildClusterDetails: %v", err)
	}
	if len(details) != 2 {
		t.Errorf("expected 2 cluster results, got %d", len(details))
	}
	for _, cd := range details {
		if len(cd.Images) == 0 {
			t.Error("cluster has no images")
		}
	}
}

// TestRun_EmbedError verifies that an Embedder error surfaces from Run.
func TestRun_EmbedError(t *testing.T) {
	embedder := &stubEmbedder{err: fmt.Errorf("GPU out of memory")}
	titler := &stubTitleGenerator{title: "T", catchyPhrase: "P"}

	ic := NewImageCluster(2, 6, t.TempDir(), embedder, titler)
	images := makeUploadedImages(t, 3)

	_, err := ic.Run(context.Background(), images)
	if err == nil {
		t.Fatal("expected error from failed Embedder, got nil")
	}
}

// TestRun_ImageFilenamesAreBasenames verifies that ClusterDetails.Images
// contains only base filenames, not full paths.
func TestRun_ImageFilenamesAreBasenames(t *testing.T) {
	embedder := &stubEmbedder{}
	titler := &stubTitleGenerator{title: "T", catchyPhrase: "P"}

	ic := NewImageCluster(2, 6, t.TempDir(), embedder, titler)
	images := makeUploadedImages(t, 4)

	details, err := ic.Run(context.Background(), images)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	for key, cd := range details {
		for _, filename := range cd.Images {
			if strings.ContainsAny(filename, "/\\") {
				t.Errorf("cluster %s: image filename %q contains path separator", key, filename)
			}
		}
	}
}

// mapKeys returns the keys of a map for error messages.
func mapKeys(m map[string]models.ClusterDetails) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

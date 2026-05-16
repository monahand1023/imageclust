package workflow

import (
	"fmt"
	"imageclust/internal/models"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestMaxWorkers(t *testing.T) {
	if maxWorkers != runtime.NumCPU() {
		t.Errorf("maxWorkers = %d, want %d", maxWorkers, runtime.NumCPU())
	}
	if maxWorkers < 1 {
		t.Errorf("maxWorkers must be ≥ 1, got %d", maxWorkers)
	}
}

func TestEmbJobStruct(t *testing.T) {
	item := itemRecord{ID: "img_0", ImagePath: "/tmp/a.jpg"}
	job := embJob{index: 2, item: item}
	if job.index != 2 || job.item.ID != "img_0" {
		t.Errorf("embJob fields not set correctly: %+v", job)
	}
}

func TestEmbResultStruct(t *testing.T) {
	res := embResult{
		index:     1,
		itemID:    "img_1",
		embedding: []float32{0.1, 0.2, 0.3},
	}
	if res.index != 1 || res.itemID != "img_1" || len(res.embedding) != 3 {
		t.Errorf("embResult fields wrong: %+v", res)
	}
}

// --- saveImages --------------------------------------------------------

func TestSaveImages_IndexBasedNaming(t *testing.T) {
	dir := t.TempDir()
	imageDir := filepath.Join(dir, "images")
	if err := os.MkdirAll(imageDir, 0755); err != nil {
		t.Fatal(err)
	}

	ic := &ImageCluster{TempDir: dir}
	images := []models.UploadedImage{
		{Filename: "photo.JPG", Data: []byte("data1")},
		{Filename: "snapshot.png", Data: []byte("data2")},
		{Filename: "noext", Data: []byte("data3")},
	}

	items, err := ic.saveImages(images, imageDir)
	if err != nil {
		t.Fatalf("saveImages: %v", err)
	}
	if len(items) != 3 {
		t.Fatalf("len(items) = %d, want 3", len(items))
	}

	for i, item := range items {
		wantID := fmt.Sprintf("img_%d", i)
		if item.ID != wantID {
			t.Errorf("item[%d].ID = %q, want %q", i, item.ID, wantID)
		}
	}

	// Extensions are lowercased and preserved; missing ext defaults to .jpg.
	wantSuffixes := []string{"img_0.jpg", "img_1.png", "img_2.jpg"}
	for i, want := range wantSuffixes {
		if !strings.HasSuffix(items[i].ImagePath, want) {
			t.Errorf("item[%d].ImagePath = %q, want suffix %q", i, items[i].ImagePath, want)
		}
	}

	// All files must exist on disk with the correct content.
	wantData := [][]byte{[]byte("data1"), []byte("data2"), []byte("data3")}
	for i, item := range items {
		got, err := os.ReadFile(item.ImagePath)
		if err != nil {
			t.Errorf("item[%d]: ReadFile: %v", i, err)
			continue
		}
		if string(got) != string(wantData[i]) {
			t.Errorf("item[%d]: content = %q, want %q", i, got, wantData[i])
		}
	}
}

func TestSaveImages_CollisionAvoidance(t *testing.T) {
	// Two files with identical names must not overwrite each other.
	dir := t.TempDir()
	imageDir := filepath.Join(dir, "images")
	os.MkdirAll(imageDir, 0755)

	ic := &ImageCluster{TempDir: dir}
	images := []models.UploadedImage{
		{Filename: "same.jpg", Data: []byte("first")},
		{Filename: "same.jpg", Data: []byte("second")},
	}

	items, err := ic.saveImages(images, imageDir)
	if err != nil {
		t.Fatalf("saveImages: %v", err)
	}

	got0, _ := os.ReadFile(items[0].ImagePath)
	got1, _ := os.ReadFile(items[1].ImagePath)
	if string(got0) != "first" || string(got1) != "second" {
		t.Errorf("collision: got %q and %q, want first and second", got0, got1)
	}
}

// --- selectRepresentatives --------------------------------------------

func TestSelectRepresentatives_FewerThanK(t *testing.T) {
	paths := []string{"a.jpg", "b.jpg"}
	embs := [][]float32{{1, 0}, {0, 1}}
	got := selectRepresentatives(paths, embs, 5)
	if len(got) != 2 {
		t.Errorf("expected 2 (all), got %d", len(got))
	}
}

func TestSelectRepresentatives_Empty(t *testing.T) {
	got := selectRepresentatives(nil, nil, 3)
	if got != nil {
		t.Errorf("expected nil for empty input, got %v", got)
	}
}

func TestSelectRepresentatives_EmptyEmbeddings(t *testing.T) {
	// No embeddings → falls back to first k paths.
	paths := []string{"a.jpg", "b.jpg", "c.jpg"}
	got := selectRepresentatives(paths, nil, 2)
	if len(got) != 2 {
		t.Errorf("empty embs: got %d, want 2", len(got))
	}
}

func TestSelectRepresentatives_PicksClosest(t *testing.T) {
	// Three images; the first two are aligned with [1,0], the third is
	// perpendicular. With k=2 we expect the first two selected.
	paths := []string{"a.jpg", "b.jpg", "c.jpg"}
	embs := [][]float32{
		{1, 0},     // aligned with centroid direction
		{0.9, 0.1}, // close
		{0, 1},     // perpendicular — should be excluded
	}
	got := selectRepresentatives(paths, embs, 2)
	if len(got) != 2 {
		t.Errorf("expected 2 representatives, got %d", len(got))
	}
	for _, p := range got {
		if p == "c.jpg" {
			t.Error("c.jpg should not be selected as representative")
		}
	}
}

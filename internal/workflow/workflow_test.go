package workflow

import (
	"runtime"
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

func TestSelectRepresentatives_PicksClosest(t *testing.T) {
	// Three images; the first two are aligned with [1,0], the third is
	// perpendicular. With k=2 we expect the first two selected.
	paths := []string{"a.jpg", "b.jpg", "c.jpg"}
	embs := [][]float32{
		{1, 0},  // aligned with centroid direction
		{0.9, 0.1},
		{0, 1},  // perpendicular — should be excluded
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

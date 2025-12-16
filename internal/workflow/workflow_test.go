package workflow

import (
	"runtime"
	"testing"
)

func TestMaxWorkersConfiguration(t *testing.T) {
	// maxWorkers should be set to number of CPUs
	expected := runtime.NumCPU()
	if maxWorkers != expected {
		t.Errorf("maxWorkers = %d, want %d (runtime.NumCPU())", maxWorkers, expected)
	}
}

func TestMaxWorkersPositive(t *testing.T) {
	if maxWorkers < 1 {
		t.Errorf("maxWorkers should be at least 1, got %d", maxWorkers)
	}
}

func TestEmbeddingJobStruct(t *testing.T) {
	job := embeddingJob{
		index: 5,
		item: ItemDetails{
			ID:        "test-id",
			ImagePath: "/path/to/image.jpg",
			Labels:    []string{"label1", "label2"},
		},
	}

	if job.index != 5 {
		t.Errorf("expected index 5, got %d", job.index)
	}
	if job.item.ID != "test-id" {
		t.Errorf("expected ID 'test-id', got %s", job.item.ID)
	}
}

func TestEmbeddingResultStruct(t *testing.T) {
	result := embeddingResult{
		index:     3,
		embedding: []float32{0.1, 0.2, 0.3},
		itemID:    "item-3",
		err:       nil,
	}

	if result.index != 3 {
		t.Errorf("expected index 3, got %d", result.index)
	}
	if len(result.embedding) != 3 {
		t.Errorf("expected 3 embeddings, got %d", len(result.embedding))
	}
	if result.itemID != "item-3" {
		t.Errorf("expected itemID 'item-3', got %s", result.itemID)
	}
	if result.err != nil {
		t.Errorf("expected nil error, got %v", result.err)
	}
}

func TestItemDetailsStruct(t *testing.T) {
	item := ItemDetails{
		ID:        "img_1",
		ImagePath: "/tmp/images/test.jpg",
		Labels:    []string{"Person", "Clothing", "Fashion"},
	}

	if item.ID != "img_1" {
		t.Errorf("expected ID 'img_1', got %s", item.ID)
	}
	if item.ImagePath != "/tmp/images/test.jpg" {
		t.Errorf("expected ImagePath '/tmp/images/test.jpg', got %s", item.ImagePath)
	}
	if len(item.Labels) != 3 {
		t.Errorf("expected 3 labels, got %d", len(item.Labels))
	}
}

func TestMakeItemMap(t *testing.T) {
	items := []ItemDetails{
		{ID: "a", ImagePath: "/path/a.jpg", Labels: []string{"l1"}},
		{ID: "b", ImagePath: "/path/b.jpg", Labels: []string{"l2"}},
		{ID: "c", ImagePath: "/path/c.jpg", Labels: []string{"l3"}},
	}

	itemMap := makeItemMap(items)

	if len(itemMap) != 3 {
		t.Errorf("expected 3 items in map, got %d", len(itemMap))
	}

	if item, ok := itemMap["a"]; !ok {
		t.Error("expected item 'a' in map")
	} else if item.ImagePath != "/path/a.jpg" {
		t.Errorf("expected ImagePath '/path/a.jpg', got %s", item.ImagePath)
	}

	if _, ok := itemMap["nonexistent"]; ok {
		t.Error("expected 'nonexistent' to not be in map")
	}
}

func TestFormatLabels(t *testing.T) {
	labelsSet := map[string]struct{}{
		"Person":   {},
		"Clothing": {},
	}

	result := formatLabels(labelsSet)

	// Since map iteration order is not guaranteed, check both labels exist
	if len(result) == 0 {
		t.Error("expected non-empty result")
	}

	// Check that it contains expected labels (order may vary)
	containsPerson := false
	containsClothing := false
	if result == "Person, Clothing" || result == "Clothing, Person" {
		containsPerson = true
		containsClothing = true
	}

	if !containsPerson || !containsClothing {
		t.Errorf("expected result to contain both labels, got: %s", result)
	}
}

func TestGetItemIDs(t *testing.T) {
	items := []ItemDetails{
		{ID: "id1"},
		{ID: "id2"},
		{ID: "id3"},
	}

	ids := getItemIDs(items)

	if len(ids) != 3 {
		t.Errorf("expected 3 IDs, got %d", len(ids))
	}

	for i, id := range ids {
		expected := items[i].ID
		if id != expected {
			t.Errorf("expected ID at index %d to be %s, got %s", i, expected, id)
		}
	}
}

package embeddings

import (
	"reflect"
	"testing"
)

func TestGenerateLabelVector_SingleLabel(t *testing.T) {
	labelSet := map[string]int{
		"Person":   0,
		"Clothing": 1,
		"Fashion":  2,
	}
	labels := []string{"Person"}

	result := GenerateLabelVector(labels, labelSet)

	if len(result) != 3 {
		t.Errorf("expected vector length 3, got %d", len(result))
	}
	if result[0] != 1.0 {
		t.Errorf("expected result[0] = 1.0, got %f", result[0])
	}
	if result[1] != 0.0 {
		t.Errorf("expected result[1] = 0.0, got %f", result[1])
	}
	if result[2] != 0.0 {
		t.Errorf("expected result[2] = 0.0, got %f", result[2])
	}
}

func TestGenerateLabelVector_MultipleLabels(t *testing.T) {
	labelSet := map[string]int{
		"Person":   0,
		"Clothing": 1,
		"Fashion":  2,
	}
	labels := []string{"Person", "Fashion"}

	result := GenerateLabelVector(labels, labelSet)

	if len(result) != 3 {
		t.Errorf("expected vector length 3, got %d", len(result))
	}
	if result[0] != 1.0 {
		t.Errorf("expected result[0] = 1.0, got %f", result[0])
	}
	if result[1] != 0.0 {
		t.Errorf("expected result[1] = 0.0, got %f", result[1])
	}
	if result[2] != 1.0 {
		t.Errorf("expected result[2] = 1.0, got %f", result[2])
	}
}

func TestGenerateLabelVector_EmptyLabels(t *testing.T) {
	labelSet := map[string]int{
		"Person":   0,
		"Clothing": 1,
	}
	labels := []string{}

	result := GenerateLabelVector(labels, labelSet)

	if len(result) != 2 {
		t.Errorf("expected vector length 2, got %d", len(result))
	}
	for i, v := range result {
		if v != 0.0 {
			t.Errorf("expected result[%d] = 0.0, got %f", i, v)
		}
	}
}

func TestGenerateLabelVector_UnknownLabel(t *testing.T) {
	labelSet := map[string]int{
		"Person":   0,
		"Clothing": 1,
	}
	labels := []string{"Unknown"}

	result := GenerateLabelVector(labels, labelSet)

	if len(result) != 2 {
		t.Errorf("expected vector length 2, got %d", len(result))
	}
	// All zeros since "Unknown" is not in the labelSet
	for i, v := range result {
		if v != 0.0 {
			t.Errorf("expected result[%d] = 0.0, got %f", i, v)
		}
	}
}

func TestGenerateLabelVector_EmptyLabelSet(t *testing.T) {
	labelSet := map[string]int{}
	labels := []string{"Person"}

	result := GenerateLabelVector(labels, labelSet)

	if len(result) != 0 {
		t.Errorf("expected empty vector, got length %d", len(result))
	}
}

func TestCombineEmbeddings_Basic(t *testing.T) {
	embedding := []float32{1.0, 2.0, 3.0}
	labelVector := []float32{0.0, 1.0}

	result := CombineEmbeddings(embedding, labelVector)

	expected := []float32{1.0, 2.0, 3.0, 0.0, 1.0}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("CombineEmbeddings result = %v, want %v", result, expected)
	}
}

func TestCombineEmbeddings_EmptyEmbedding(t *testing.T) {
	embedding := []float32{}
	labelVector := []float32{1.0, 0.0}

	result := CombineEmbeddings(embedding, labelVector)

	expected := []float32{1.0, 0.0}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("CombineEmbeddings result = %v, want %v", result, expected)
	}
}

func TestCombineEmbeddings_EmptyLabelVector(t *testing.T) {
	embedding := []float32{1.0, 2.0}
	labelVector := []float32{}

	result := CombineEmbeddings(embedding, labelVector)

	expected := []float32{1.0, 2.0}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("CombineEmbeddings result = %v, want %v", result, expected)
	}
}

func TestCombineEmbeddings_BothEmpty(t *testing.T) {
	embedding := []float32{}
	labelVector := []float32{}

	result := CombineEmbeddings(embedding, labelVector)

	if len(result) != 0 {
		t.Errorf("expected empty result, got length %d", len(result))
	}
}

func TestCombineEmbeddings_Length(t *testing.T) {
	embedding := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	labelVector := []float32{0.0, 1.0, 0.0}

	result := CombineEmbeddings(embedding, labelVector)

	expectedLen := len(embedding) + len(labelVector)
	if len(result) != expectedLen {
		t.Errorf("expected result length %d, got %d", expectedLen, len(result))
	}
}

func TestAppContext_Initialization(t *testing.T) {
	appCtx := &AppContext{
		ImageDir:      "/tmp/images",
		CacheDir:      "/tmp/cache",
		LabelSet:      make(map[string]int),
		LabelsMapping: make(map[string][]string),
	}

	if appCtx.ImageDir != "/tmp/images" {
		t.Errorf("ImageDir = %s, want '/tmp/images'", appCtx.ImageDir)
	}
	if appCtx.CacheDir != "/tmp/cache" {
		t.Errorf("CacheDir = %s, want '/tmp/cache'", appCtx.CacheDir)
	}
	if appCtx.LabelSet == nil {
		t.Error("LabelSet should not be nil")
	}
	if appCtx.LabelsMapping == nil {
		t.Error("LabelsMapping should not be nil")
	}
}

func TestAppContext_ZeroValue(t *testing.T) {
	var appCtx AppContext

	if appCtx.ImageDir != "" {
		t.Errorf("zero value ImageDir should be empty, got '%s'", appCtx.ImageDir)
	}
	if appCtx.CacheDir != "" {
		t.Errorf("zero value CacheDir should be empty, got '%s'", appCtx.CacheDir)
	}
	if appCtx.LabelSet != nil {
		t.Error("zero value LabelSet should be nil")
	}
	if appCtx.LabelsMapping != nil {
		t.Error("zero value LabelsMapping should be nil")
	}
}

func TestGenerateLabelVector_AllLabels(t *testing.T) {
	labelSet := map[string]int{
		"A": 0,
		"B": 1,
		"C": 2,
	}
	labels := []string{"A", "B", "C"}

	result := GenerateLabelVector(labels, labelSet)

	for i, v := range result {
		if v != 1.0 {
			t.Errorf("expected result[%d] = 1.0, got %f", i, v)
		}
	}
}

func TestGenerateLabelVector_DuplicateLabels(t *testing.T) {
	labelSet := map[string]int{
		"Person": 0,
	}
	labels := []string{"Person", "Person", "Person"}

	result := GenerateLabelVector(labels, labelSet)

	if len(result) != 1 {
		t.Errorf("expected vector length 1, got %d", len(result))
	}
	// Should still be 1.0 even with duplicates
	if result[0] != 1.0 {
		t.Errorf("expected result[0] = 1.0, got %f", result[0])
	}
}

package rekognition

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

func TestMaxImageSizeConstant(t *testing.T) {
	expected := 5 * 1024 * 1024 // 5MB
	if MaxImageSize != expected {
		t.Errorf("MaxImageSize = %d, want %d", MaxImageSize, expected)
	}
}

func TestRekognitionService_GetCacheFilePath(t *testing.T) {
	rs := &RekognitionService{
		CacheDir: "/tmp/cache",
	}

	tests := []struct {
		name      string
		imagePath string
		expected  string
	}{
		{
			name:      "simple image name",
			imagePath: "/images/test.jpg",
			expected:  "/tmp/cache/test.jpg_labels.json",
		},
		{
			name:      "nested path",
			imagePath: "/path/to/nested/image.png",
			expected:  "/tmp/cache/image.png_labels.json",
		},
		{
			name:      "no extension",
			imagePath: "/images/image",
			expected:  "/tmp/cache/image_labels.json",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := rs.getCacheFilePath(tt.imagePath)
			if result != tt.expected {
				t.Errorf("getCacheFilePath(%s) = %s, want %s", tt.imagePath, result, tt.expected)
			}
		})
	}
}

func TestRekognitionService_LoadLabelsFromCache_FileNotExists(t *testing.T) {
	rs := &RekognitionService{
		CacheDir: "/tmp/nonexistent",
	}

	_, err := rs.loadLabelsFromCache("/tmp/nonexistent/missing_labels.json")
	if err == nil {
		t.Error("expected error for non-existent cache file")
	}
}

func TestRekognitionService_StoreAndLoadLabelsFromCache(t *testing.T) {
	// Create a temp directory for cache
	tempDir, err := os.MkdirTemp("", "rekognition_test")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	rs := &RekognitionService{
		CacheDir: tempDir,
	}

	// Create test labels
	labels := []types.Label{
		{
			Name:       aws.String("Person"),
			Confidence: aws.Float32(95.5),
		},
		{
			Name:       aws.String("Clothing"),
			Confidence: aws.Float32(88.2),
		},
	}

	cacheFilePath := filepath.Join(tempDir, "test_image.jpg_labels.json")

	// Store labels
	err = rs.storeLabelsInCache(cacheFilePath, labels)
	if err != nil {
		t.Fatalf("storeLabelsInCache failed: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(cacheFilePath); os.IsNotExist(err) {
		t.Error("cache file was not created")
	}

	// Load labels back
	loadedLabels, err := rs.loadLabelsFromCache(cacheFilePath)
	if err != nil {
		t.Fatalf("loadLabelsFromCache failed: %v", err)
	}

	// Verify loaded labels
	if len(loadedLabels) != 2 {
		t.Errorf("expected 2 labels, got %d", len(loadedLabels))
	}

	if loadedLabels[0].Name == nil || *loadedLabels[0].Name != "Person" {
		t.Errorf("first label should be 'Person'")
	}
	if loadedLabels[1].Name == nil || *loadedLabels[1].Name != "Clothing" {
		t.Errorf("second label should be 'Clothing'")
	}
}

func TestRekognitionService_LoadLabelsFromCache_InvalidJSON(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "rekognition_test")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	rs := &RekognitionService{
		CacheDir: tempDir,
	}

	// Write invalid JSON to cache file
	cacheFilePath := filepath.Join(tempDir, "invalid_labels.json")
	err = os.WriteFile(cacheFilePath, []byte("not valid json"), 0644)
	if err != nil {
		t.Fatalf("failed to write invalid cache file: %v", err)
	}

	// Try to load - should fail
	_, err = rs.loadLabelsFromCache(cacheFilePath)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestRekognitionService_StoreLabelsInCache_EmptyLabels(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "rekognition_test")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	rs := &RekognitionService{
		CacheDir: tempDir,
	}

	labels := []types.Label{}
	cacheFilePath := filepath.Join(tempDir, "empty_labels.json")

	err = rs.storeLabelsInCache(cacheFilePath, labels)
	if err != nil {
		t.Fatalf("storeLabelsInCache failed for empty labels: %v", err)
	}

	// Load back and verify
	loadedLabels, err := rs.loadLabelsFromCache(cacheFilePath)
	if err != nil {
		t.Fatalf("loadLabelsFromCache failed: %v", err)
	}

	if len(loadedLabels) != 0 {
		t.Errorf("expected 0 labels, got %d", len(loadedLabels))
	}
}

func TestRekognitionServiceStruct(t *testing.T) {
	rs := &RekognitionService{
		Client:   nil,
		CacheDir: "/tmp/cache",
	}

	if rs.CacheDir != "/tmp/cache" {
		t.Errorf("CacheDir = %s, want '/tmp/cache'", rs.CacheDir)
	}
}

func TestLabelJSON_Marshaling(t *testing.T) {
	label := types.Label{
		Name:       aws.String("Fashion"),
		Confidence: aws.Float32(92.5),
	}

	data, err := json.Marshal(label)
	if err != nil {
		t.Fatalf("failed to marshal label: %v", err)
	}

	var unmarshaled types.Label
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("failed to unmarshal label: %v", err)
	}

	if *unmarshaled.Name != "Fashion" {
		t.Errorf("unmarshaled Name = %s, want 'Fashion'", *unmarshaled.Name)
	}
}

func TestRekognitionService_CacheFilePathConsistency(t *testing.T) {
	rs := &RekognitionService{
		CacheDir: "/cache",
	}

	imagePath := "/images/photo.jpg"

	// Calling multiple times should return the same path
	path1 := rs.getCacheFilePath(imagePath)
	path2 := rs.getCacheFilePath(imagePath)

	if path1 != path2 {
		t.Errorf("getCacheFilePath returned inconsistent results: %s vs %s", path1, path2)
	}
}

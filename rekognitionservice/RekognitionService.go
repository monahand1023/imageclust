// RekognitionService.go
package rekognitionservice

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

// RekognitionService interacts with AWS Rekognition to detect labels in images.
type RekognitionService struct {
	Client   *rekognition.Client
	CacheDir string // Directory for storing cached labels
}

// NewRekognitionService initializes the Rekognition client and cache directory.
// Parameters:
// - region: AWS region (e.g., "us-east-1").
// - cacheDir: Directory path where cached labels will be stored.
func NewRekognitionService(region, cacheDir string) (*RekognitionService, error) {
	// Load AWS configuration
	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("unable to load AWS SDK config: %v", err)
	}

	// Initialize Rekognition client
	client := rekognition.NewFromConfig(cfg)

	// Ensure the cache directory exists
	err = os.MkdirAll(cacheDir, 0755)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %v", err)
	}

	return &RekognitionService{
		Client:   client,
		CacheDir: cacheDir,
	}, nil
}

// DetectLabels detects labels from an image stored at the specified path using AWS Rekognition.
// It checks for cached results before calling the Rekognition API.
// Parameters:
// - imagePath: Full path to the image file.
// - maxLabels: Maximum number of labels to return.
// - minConfidence: Minimum confidence level for labels.
// Returns:
// - A slice of detected labels.
// - An error if detection fails.
func (rs *RekognitionService) DetectLabels(imagePath string, maxLabels int32, minConfidence float32) ([]types.Label, error) {
	// Generate cache file path based on the image name
	cacheFilePath := rs.getCacheFilePath(imagePath)

	// Check if the cache file exists
	if labels, err := rs.loadLabelsFromCache(cacheFilePath); err == nil {
		// Labels found in cache
		return labels, nil
	}

	// If no cache, proceed to call Rekognition API
	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read image file '%s': %v", imagePath, err)
	}

	input := &rekognition.DetectLabelsInput{
		Image: &types.Image{
			Bytes: imageBytes,
		},
		MaxLabels:     aws.Int32(maxLabels),
		MinConfidence: aws.Float32(minConfidence),
	}

	result, err := rs.Client.DetectLabels(context.TODO(), input)
	if err != nil {
		return nil, fmt.Errorf("failed to detect labels for image '%s': %v", imagePath, err)
	}

	// Cache the detected labels
	if err := rs.storeLabelsInCache(cacheFilePath, result.Labels); err != nil {
		fmt.Printf("Warning: failed to cache labels for '%s': %v\n", imagePath, err)
	}

	return result.Labels, nil
}

// getCacheFilePath generates the path for the cache file based on the image name.
func (rs *RekognitionService) getCacheFilePath(imagePath string) string {
	// Create a unique file name for the cache based on the image file name
	fileName := filepath.Base(imagePath) + "_labels.json"
	return filepath.Join(rs.CacheDir, fileName)
}

// loadLabelsFromCache attempts to load labels from a cached JSON file.
// Returns the labels if successful, otherwise returns an error.
func (rs *RekognitionService) loadLabelsFromCache(cacheFilePath string) ([]types.Label, error) {
	// Check if cache file exists
	if _, err := os.Stat(cacheFilePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("cache file does not exist: %s", cacheFilePath)
	}

	// Read the cached file
	cacheData, err := os.ReadFile(cacheFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read cache file '%s': %v", cacheFilePath, err)
	}

	// Parse the cached JSON file
	var labels []types.Label
	if err := json.Unmarshal(cacheData, &labels); err != nil {
		return nil, fmt.Errorf("failed to unmarshal cache file '%s': %v", cacheFilePath, err)
	}

	return labels, nil
}

// storeLabelsInCache stores the detected labels in a JSON file in the cache directory.
func (rs *RekognitionService) storeLabelsInCache(cacheFilePath string, labels []types.Label) error {
	// Convert labels to JSON
	cacheData, err := json.Marshal(labels)
	if err != nil {
		return fmt.Errorf("failed to marshal labels for cache file '%s': %v", cacheFilePath, err)
	}

	// Write the JSON data to a file
	err = os.WriteFile(cacheFilePath, cacheData, 0644)
	if err != nil {
		return fmt.Errorf("failed to write cache file '%s': %v", cacheFilePath, err)
	}

	return nil
}

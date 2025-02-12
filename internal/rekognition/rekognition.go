// Package rekognition
package rekognition

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"gocv.io/x/gocv"
	"image"
	"log"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

const MaxImageSize = 5 * 1024 * 1024 // 5MB in bytes

// RekognitionService interacts with AWS Rekognition to detect labels in images.
type RekognitionService struct {
	Client   *rekognition.Client
	CacheDir string // Directory for storing cached labels
}

// NewRekognitionService initializes the Rekognition client and cache directory.
// Parameters:
// - region: AWS region (e.g., "us-west-2").
// - cacheDir: Directory path where cached labels will be stored.
func NewRekognitionService(region, cacheDir string) (*RekognitionService, error) {
	var cfg aws.Config
	var err error

	if os.Getenv("DEV_MODE") == "true" {
		// Load static credentials from environment variables for development
		accessKey := os.Getenv("AWS_ACCESS_KEY_ID")
		secretKey := os.Getenv("AWS_SECRET_ACCESS_KEY")

		if accessKey == "" || secretKey == "" {
			return nil, fmt.Errorf("AWS credentials not found in environment variables")
		}

		cfg, err = config.LoadDefaultConfig(context.TODO(),
			config.WithRegion(region),
			config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
				accessKey,
				secretKey,
				"",
			)),
		)
	} else {
		// Load default AWS configuration for production
		cfg, err = config.LoadDefaultConfig(context.TODO(),
			config.WithRegion(region),
		)
	}

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
// DetectLabels detects labels from an image stored at the specified path using AWS Rekognition.
func (rs *RekognitionService) DetectLabels(imagePath string, maxLabels int32, minConfidence float32) ([]types.Label, error) {
	// Generate cache file path based on the image name
	cacheFilePath := rs.getCacheFilePath(imagePath)

	// Check if the cache file exists
	if labels, err := rs.loadLabelsFromCache(cacheFilePath); err == nil {
		return labels, nil
	}

	// If no cache, resize if needed and proceed to call Rekognition API
	imageBytes, err := resizeImageIfNeeded(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to process image file '%s': %v", imagePath, err)
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

// resizeImageIfNeeded resizes the image if it's larger than MaxImageSize
func resizeImageIfNeeded(imagePath string) ([]byte, error) {
	// Read the file
	fileInfo, err := os.Stat(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %v", err)
	}

	// If file is under size limit, just read and return it
	if fileInfo.Size() <= MaxImageSize {
		return os.ReadFile(imagePath)
	}

	log.Printf("Image %s is too large (%d bytes), resizing...", imagePath, fileInfo.Size())

	// Read image using gocv
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		return nil, fmt.Errorf("failed to read image for resizing")
	}
	defer img.Close()

	// Calculate new dimensions while maintaining aspect ratio
	originalSize := img.Size()
	ratio := float64(originalSize[1]) / float64(originalSize[0])

	// Start with a reasonable max dimension (e.g., 2048 pixels)
	var newWidth, newHeight int
	maxDimension := 2048
	if originalSize[0] > originalSize[1] {
		newWidth = maxDimension
		newHeight = int(float64(maxDimension) * ratio)
	} else {
		newHeight = maxDimension
		newWidth = int(float64(maxDimension) / ratio)
	}

	// Create a new mat for the resized image
	resized := gocv.NewMat()
	defer resized.Close()

	// Resize the image
	gocv.Resize(img, &resized, image.Point{X: newWidth, Y: newHeight}, 0, 0, gocv.InterpolationLinear)

	// Create a temporary file for the resized image
	tempFile, err := os.CreateTemp("", "resize_*.jpg")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %v", err)
	}
	tempPath := tempFile.Name()
	tempFile.Close()
	defer os.Remove(tempPath)

	// Write the resized image
	success := gocv.IMWrite(tempPath, resized)
	if !success {
		return nil, fmt.Errorf("failed to write resized image")
	}

	// Read the resized file
	resizedData, err := os.ReadFile(tempPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read resized image: %v", err)
	}

	// If still too large, try again with more aggressive resizing
	if len(resizedData) > MaxImageSize {
		log.Printf("Image still too large after initial resize (%d bytes), reducing dimensions further", len(resizedData))

		// Try with smaller dimensions
		newWidth = newWidth / 2
		newHeight = newHeight / 2
		gocv.Resize(img, &resized, image.Point{X: newWidth, Y: newHeight}, 0, 0, gocv.InterpolationLinear)

		success = gocv.IMWrite(tempPath, resized)
		if !success {
			return nil, fmt.Errorf("failed to write resized image with reduced dimensions")
		}

		resizedData, err = os.ReadFile(tempPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read resized image: %v", err)
		}
	}

	log.Printf("Successfully resized image from %d bytes to %d bytes", fileInfo.Size(), len(resizedData))
	return resizedData, nil
}

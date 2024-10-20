// embeddings/embeddings.go
package embeddings

import (
	"ProductSetter/rekognitionservice"
	"encoding/json"
	"fmt"
	"image"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"gocv.io/x/gocv"
)

// AppContext holds application-wide shared resources
type AppContext struct {
	ImageDir      string              // Directory for image files
	CacheDir      string              // Cache directory for storing embeddings
	LabelSet      map[string]int      // Set of all possible labels for encoding
	Mutex         sync.Mutex          // To handle concurrent access to shared resources
	LabelsMapping map[string][]string // Map of productRefID -> labels
	Net           gocv.Net            // OpenCV DNN network
}

// LoadPretrainedModel loads the pre-trained ResNet50 model using GoCV
func LoadPretrainedModel(modelPath string) (gocv.Net, error) {
	// Read the network using the ONNX model
	net := gocv.ReadNet(modelPath, "")
	if net.Empty() {
		return net, fmt.Errorf("failed to load ResNet50 model from: %s", modelPath)
	}

	// Set preferable backend and target to CPU
	net.SetPreferableBackend(gocv.NetBackendDefault)
	net.SetPreferableTarget(gocv.NetTargetCPU)

	return net, nil
}

// PreprocessImage resizes and normalizes the image to match ResNet50 input requirements
func PreprocessImage(imagePath string) (gocv.Mat, error) {
	// Load the image using GoCV
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		return img, fmt.Errorf("failed to read image: %s", imagePath)
	}
	defer img.Close()

	// Resize to 224x224 (standard for ResNet50)
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, image.Pt(224, 224), 0, 0, gocv.InterpolationLinear)

	// Convert image to RGB
	gocv.CvtColor(resized, &resized, gocv.ColorBGRToRGB)

	// Create a blob from the image
	// Parameters: scale factor, size, mean subtraction, swap RB channels, crop
	blob := gocv.BlobFromImage(resized, 1.0/255.0, image.Pt(224, 224), gocv.NewScalar(0, 0, 0, 0), false, false)

	return blob, nil
}

// GetImageEmbedding generates an image embedding using ResNet50
func GetImageEmbedding(appCtx *AppContext, imagePath string) ([]float32, error) {
	// Preprocess the image to create a blob
	blob, err := PreprocessImage(imagePath)
	if err != nil {
		return nil, err
	}
	defer blob.Close()

	// Set the input to the network
	appCtx.Net.SetInput(blob, "")

	// Forward pass to get the output from the desired layer
	// For ResNet50, we can use the 'avg_pool' layer for embeddings
	embeddingMat := appCtx.Net.Forward("avg_pool")
	if embeddingMat.Empty() {
		return nil, fmt.Errorf("failed to generate embedding for image: %s", imagePath)
	}
	defer embeddingMat.Close()

	// Extract the data as a float32 slice
	embedding, err := embeddingMat.DataPtrFloat32()
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve embedding data: %v", err)
	}

	// Verify that the embedding is not empty
	if len(embedding) == 0 {
		return nil, fmt.Errorf("embedding is empty for image: %s", imagePath)
	}

	return embedding, nil
}

// CreateProductEmbedding generates an embedding for a product using both image embeddings and Rekognition labels
func CreateProductEmbedding(productRefID string, appCtx *AppContext, rekognitionSvc *rekognitionservice.RekognitionService) ([]float64, error) {
	// Check if the combined embedding is already cached
	combinedEmbedding, found, err := CheckCache(productRefID, appCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to check cache: %v", err)
	}
	if found {
		fmt.Printf("Using cached combined embedding for product %s\n", productRefID)
		return combinedEmbedding, nil
	}

	// Get the image path
	imagePath := filepath.Join(appCtx.ImageDir, productRefID+".jpg")

	// 1. Generate an image embedding using the pre-trained model
	imageEmbedding, err := GetImageEmbedding(appCtx, imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to generate image embedding for product %s: %v", productRefID, err)
	}

	// 2. Detect labels using Rekognition (cached)
	labels, err := rekognitionSvc.DetectLabels(imagePath, 10, 80)
	if err != nil {
		return nil, fmt.Errorf("failed to detect labels for product %s: %v", productRefID, err)
	}

	// Store labels in LabelsMapping
	appCtx.Mutex.Lock()
	labelNames := []string{}
	for _, label := range labels {
		labelNames = append(labelNames, *label.Name)
	}
	appCtx.LabelsMapping[productRefID] = labelNames
	appCtx.Mutex.Unlock()

	// 3. Convert detected labels to a vector (e.g., one-hot encoding)
	labelVector := GenerateLabelVector(labelNames, appCtx.LabelSet)

	// 4. Combine the image embedding and label vector
	combinedEmbedding = CombineEmbeddings(imageEmbedding, labelVector)

	// 5. Cache the combined embedding
	err = StoreInCache(productRefID, combinedEmbedding, appCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to store embedding in cache: %v", err)
	}

	return combinedEmbedding, nil
}

// GenerateLabelVector converts labels into a one-hot encoded vector based on the full label set
func GenerateLabelVector(labels []string, labelSet map[string]int) []float64 {
	labelVector := make([]float64, len(labelSet))
	for _, label := range labels {
		if idx, exists := labelSet[label]; exists {
			labelVector[idx] = 1.0
		}
	}
	return labelVector
}

// CombineEmbeddings merges the image embedding and label vector into a single embedding
func CombineEmbeddings(imageEmbedding []float32, labelVector []float64) []float64 {
	combined := make([]float64, len(imageEmbedding)+len(labelVector))
	for i, val := range imageEmbedding {
		combined[i] = float64(val)
	}
	copy(combined[len(imageEmbedding):], labelVector)
	return combined
}

// CreateEmbeddingsForAllProducts creates embeddings for all products using both the pre-trained model and Rekognition labels
func CreateEmbeddingsForAllProducts(productRefIDs []string, appCtx *AppContext, rekognitionSvc *rekognitionservice.RekognitionService) ([][]float64, error) {
	embeddingsList := [][]float64{}
	for _, productRefID := range productRefIDs {
		combinedEmbedding, err := CreateProductEmbedding(productRefID, appCtx, rekognitionSvc)
		if err != nil {
			return nil, fmt.Errorf("failed to create embedding for product %s: %v", productRefID, err)
		}
		embeddingsList = append(embeddingsList, combinedEmbedding)
	}
	return embeddingsList, nil
}

// CheckCache retrieves a combined embedding from the cache if it exists
func CheckCache(productRefID string, appCtx *AppContext) ([]float64, bool, error) {
	cacheFilePath := filepath.Join(appCtx.CacheDir, productRefID+"_embedding.json")

	// Check if cache file exists
	if _, err := os.Stat(cacheFilePath); os.IsNotExist(err) {
		// Cache does not exist
		return nil, false, nil
	}

	// Read cached embedding
	cacheData, err := ioutil.ReadFile(cacheFilePath)
	if err != nil {
		return nil, false, fmt.Errorf("failed to read cache file: %v", err)
	}

	// Parse cached embedding
	var embedding []float64
	if err := json.Unmarshal(cacheData, &embedding); err != nil {
		return nil, false, fmt.Errorf("failed to parse cache file: %v", err)
	}

	return embedding, true, nil
}

// StoreInCache saves a combined embedding to the cache
func StoreInCache(productRefID string, embedding []float64, appCtx *AppContext) error {
	cacheFilePath := filepath.Join(appCtx.CacheDir, productRefID+"_embedding.json")

	// Convert embedding to JSON
	cacheData, err := json.Marshal(embedding)
	if err != nil {
		return fmt.Errorf("failed to marshal embedding to JSON: %v", err)
	}

	// Write to cache file
	if err := ioutil.WriteFile(cacheFilePath, cacheData, 0644); err != nil {
		return fmt.Errorf("failed to write cache file: %v", err)
	}

	return nil
}

// BuildLabelSet constructs a set of all possible labels from the dataset
func BuildLabelSet(productRefIDs []string, rekognitionSvc *rekognitionservice.RekognitionService, appCtx *AppContext) error {
	labelSet := make(map[string]int)
	index := 0

	for _, productRefID := range productRefIDs {
		imagePath := filepath.Join(appCtx.ImageDir, productRefID+".jpg")

		// Detect labels (cached)
		labels, err := rekognitionSvc.DetectLabels(imagePath, 10, 80)
		if err != nil {
			return fmt.Errorf("failed to detect labels for product %s: %v", productRefID, err)
		}

		// Collect labels into the label set
		for _, label := range labels {
			labelName := *label.Name
			if _, exists := labelSet[labelName]; !exists {
				labelSet[labelName] = index
				index++
			}
		}
	}

	// Assign the built label set to the app context
	appCtx.LabelSet = labelSet
	return nil
}

// Package embeddings/embeddings.go
package embeddings

import (
	"fmt"
	"image"
	"imageclust/rekognitionservice"
	"log"
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
	Net           gocv.Net            // OpenCV DNN network for ResNet50
	NetMutex      sync.Mutex
}

// LoadPretrainedModelONNX loads the pre-trained ResNet50 model in ONNX format using GoCV
func LoadPretrainedModelONNX(modelPath string) (gocv.Net, error) {
	// Read the network using the ResNet50 ONNX model
	net := gocv.ReadNetFromONNX(modelPath)
	if net.Empty() {
		return net, fmt.Errorf("failed to load ResNet50 ONNX model from: %s", modelPath)
	}

	// Set preferable backend and target to CPU
	err := net.SetPreferableBackend(gocv.NetBackendDefault)
	if err != nil {
		return gocv.Net{}, err
	}
	net.SetPreferableTarget(gocv.NetTargetCPU)

	return net, nil
}

// PreprocessImage resizes and normalizes the image to match ResNet50 input requirements
func PreprocessImage(imagePath string) (gocv.Mat, error) {
	log.Printf("Preprocessing image: %s", imagePath)

	// Load the image using GoCV
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		return gocv.NewMat(), fmt.Errorf("failed to read image: %s. The image file might be corrupt or unreadable", imagePath)
	}
	defer func(img *gocv.Mat) {
		err := img.Close()
		if err != nil {
		}
	}(&img)

	// Resize to 224x224 (standard for ResNet50)
	resized := gocv.NewMat()
	defer func(resized *gocv.Mat) {
		err := resized.Close()
		if err != nil {

		}
	}(&resized)

	gocv.Resize(img, &resized, image.Pt(224, 224), 0, 0, gocv.InterpolationLinear)
	if resized.Empty() {
		return gocv.NewMat(), fmt.Errorf("failed to resize image: %s. There might be an issue with the image content", imagePath)
	}

	// Convert image to RGB
	rgb := gocv.NewMat()
	defer func(rgb *gocv.Mat) {
		err := rgb.Close()
		if err != nil {
		}
	}(&rgb)

	gocv.CvtColor(resized, &rgb, gocv.ColorBGRToRGB)
	if rgb.Empty() {
		return gocv.NewMat(), fmt.Errorf("failed to convert image to RGB: %s. Image data might be invalid", imagePath)
	}

	// Create a blob from the image
	blob := gocv.NewMat()
	defer func(blob *gocv.Mat) {
		err := blob.Close()
		if err != nil {

		}
	}(&blob)

	blob = gocv.BlobFromImage(rgb, 1.0/255.0, image.Pt(224, 224), gocv.NewScalar(0, 0, 0, 0), false, false)
	if blob.Empty() {
		return gocv.NewMat(), fmt.Errorf("failed to create blob from image: %s. Blob generation failed", imagePath)
	}

	// Check the shape of the blob
	blobSize := blob.Size()
	if len(blobSize) != 4 || blobSize[0] != 1 || blobSize[1] != 3 || blobSize[2] != 224 || blobSize[3] != 224 {
		return gocv.NewMat(), fmt.Errorf("invalid blob shape for image %s: expected (1, 3, 224, 224), got %v", imagePath, blobSize)
	}

	// Return a clone of the blob to ensure it's not closed prematurely
	finalBlob := blob.Clone()

	if finalBlob.Empty() {
		return gocv.NewMat(), fmt.Errorf("final blob is empty after processing image: %s. This might indicate a deeper issue with image preprocessing", imagePath)
	}

	log.Printf("Successfully preprocessed image: %s", imagePath)
	return finalBlob, nil
}

// GetImageEmbedding generates an image embedding using ResNet50
func GetImageEmbedding(appCtx *AppContext, imagePath string) ([]float32, error) {
	// Preprocess the image to create a blob
	blob, err := PreprocessImage(imagePath)
	if err != nil {
		return nil, err
	}
	defer func(blob *gocv.Mat) {
		err := blob.Close()
		if err != nil {

		}
	}(&blob)

	// Lock the Net object
	appCtx.NetMutex.Lock()
	defer appCtx.NetMutex.Unlock()

	// Set the input to the network
	appCtx.Net.SetInput(blob, "")

	// Forward pass to get the output from the desired layer
	outputLayer := "resnetv17_dense0_fwd"
	embeddingMat := appCtx.Net.Forward(outputLayer)
	if embeddingMat.Empty() {
		return nil, fmt.Errorf("failed to generate embedding for image: %s", imagePath)
	}
	defer func(embeddingMat *gocv.Mat) {
		err := embeddingMat.Close()
		if err != nil {
		}
	}(&embeddingMat)

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

// CreateProductEmbedding generates an embedding for a product using ResNet50 and Rekognition labels

// GenerateLabelVector converts labels into a one-hot encoded vector based on the full label set
func GenerateLabelVector(labels []string, labelSet map[string]int) []float32 {
	labelVector := make([]float32, len(labelSet))
	for _, label := range labels {
		if idx, exists := labelSet[label]; exists {
			labelVector[idx] = 1.0
		}
	}
	return labelVector
}

// CombineEmbeddings merges the image embedding and label vector into a single embedding
func CombineEmbeddings(embedding []float32, labelVector []float32) []float32 {
	// Combine the two vectors
	combined := make([]float32, len(embedding)+len(labelVector))
	copy(combined, embedding)
	copy(combined[len(embedding):], labelVector)
	return combined
}

// BuildLabelSet constructs a set of all possible labels from the dataset
func BuildLabelSet(productRefIDs []string, rekognitionSvc *rekognitionservice.RekognitionService, appCtx *AppContext) error {
	log.Println("Building label set from product images")
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
	log.Printf("Label set built with %d unique labels", len(labelSet))
	return nil
}

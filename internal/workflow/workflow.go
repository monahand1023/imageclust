package workflow

import (
	"fmt"
	"imageclust/internal/ai"
	"imageclust/internal/clustering"
	"imageclust/internal/embeddings"
	"imageclust/internal/models"
	"imageclust/internal/rekognition"
	"imageclust/internal/utils"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type ImageCluster struct {
	TempDir         string
	RekognitionSvc  *rekognition.RekognitionService
	EmbeddingsModel *embeddings.AppContext
	MinClusterSize  int
	MaxClusterSize  int
	Mutex           sync.Mutex
}

func NewImageCluster(minClusterSize, maxClusterSize int, tempDir string) (*ImageCluster, error) {
	log.Printf("Initializing ImageCluster with min=%d, max=%d clusters", minClusterSize, maxClusterSize)

	appCtx := &embeddings.AppContext{
		ImageDir:      filepath.Join(tempDir, "images"),
		CacheDir:      filepath.Join(tempDir, "cache"),
		LabelSet:      make(map[string]int),
		LabelsMapping: make(map[string][]string),
	}

	log.Printf("Creating directories at %s and %s", appCtx.ImageDir, appCtx.CacheDir)

	rekogSvc, err := rekognition.NewRekognitionService("us-east-1", appCtx.CacheDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize RekognitionService: %v", err)
	}

	modelPath := "resnet50-v1-7.onnx"
	log.Printf("Loading ResNet50 model from %s", modelPath)
	net, err := embeddings.LoadPretrainedModelONNX(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load ResNet50 ONNX model: %v", err)
	}

	appCtx.Net = net

	return &ImageCluster{
		TempDir:         tempDir,
		RekognitionSvc:  rekogSvc,
		EmbeddingsModel: appCtx,
		MinClusterSize:  minClusterSize,
		MaxClusterSize:  maxClusterSize,
	}, nil
}

func (ic *ImageCluster) Run(uploadedImages []models.UploadedImage) (map[string]models.ClusterDetails, string, error) {
	startTime := time.Now()
	log.Println("Starting ImageCluster run...")

	err := os.MkdirAll(ic.EmbeddingsModel.ImageDir, 0755)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create image directory: %v", err)
	}

	err = os.MkdirAll(ic.EmbeddingsModel.CacheDir, 0755)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create cache directory: %v", err)
	}

	log.Printf("Processing %d uploaded images", len(uploadedImages))
	productDetails := make([]models.CombinedProductDetails, len(uploadedImages))
	productRefIDs := make([]string, len(uploadedImages))

	for i, img := range uploadedImages {
		imagePath := filepath.Join(ic.EmbeddingsModel.ImageDir, img.Filename)
		err := os.WriteFile(imagePath, img.Data, 0644)
		if err != nil {
			return nil, "", fmt.Errorf("failed to save uploaded image %s: %v", img.Filename, err)
		}
		log.Printf("Saved image %s to %s", img.Filename, imagePath)

		labels, err := ic.RekognitionSvc.DetectLabels(imagePath, 10, 75.0)
		if err != nil {
			return nil, "", fmt.Errorf("failed to detect labels for %s: %v", img.Filename, err)
		}

		labelNames := make([]string, len(labels))
		for j, label := range labels {
			labelNames[j] = *label.Name
		}
		log.Printf("Detected %d labels for image %s", len(labelNames), img.Filename)

		productRefIDs[i] = fmt.Sprintf("img_%d", i)
		productDetails[i] = models.CombinedProductDetails{
			ProductReferenceID: productRefIDs[i],
			ImagePath:          imagePath,
			Labels:             labelNames,
		}
	}

	// Build label set using the actual files in the directory
	log.Println("Building label set from detected labels")
	err = embeddings.BuildLabelSet(productRefIDs, ic.RekognitionSvc, ic.EmbeddingsModel)
	if err != nil {
		return nil, "", fmt.Errorf("failed to build label set: %v", err)
	}

	log.Println("Creating embeddings for all images")
	embeddingsList, productReferenceIDs, err := ic.CreateEmbeddingsForAllProducts(productDetails)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create embeddings: %v", err)
	}
	log.Printf("Created embeddings for %d images", len(embeddingsList))

	log.Println("Performing clustering")
	clusters, success := clustering.PerformClusteringWithConstraints(
		embeddingsList,
		productReferenceIDs,
		ic.MinClusterSize,
		ic.MaxClusterSize,
	)
	if !success {
		return nil, "", fmt.Errorf("clustering failed due to constraints")
	}
	log.Printf("Formed %d clusters", len(clusters))

	log.Println("Preparing cluster details")
	clusterDetails := ic.PrepareClusterDetails(clusters, productDetails)

	log.Println("Generating HTML output")
	htmlOutputPath, err := utils.GenerateHTMLOutput(clusterDetails, ic.TempDir)
	if err != nil {
		return nil, "", fmt.Errorf("failed to generate HTML output: %v", err)
	}
	log.Printf("Generated HTML output at: %s", htmlOutputPath)

	log.Printf("Total execution time: %v", time.Since(startTime))
	return clusterDetails, htmlOutputPath, nil
}

func (ic *ImageCluster) CreateEmbeddingsForAllProducts(productDetails []models.CombinedProductDetails) ([][]float32, []string, error) {
	embeddingsList := make([][]float32, len(productDetails))
	productReferenceIDs := make([]string, len(productDetails))
	var mu sync.Mutex
	var wg sync.WaitGroup
	errChan := make(chan error, len(productDetails))

	log.Printf("Creating embeddings for %d products concurrently", len(productDetails))

	for i, product := range productDetails {
		wg.Add(1)
		go func(idx int, pd models.CombinedProductDetails) {
			defer wg.Done()

			log.Printf("Generating embedding for product %s", pd.ProductReferenceID)
			imageEmbedding, err := embeddings.GetImageEmbedding(ic.EmbeddingsModel, pd.ImagePath)
			if err != nil {
				errChan <- fmt.Errorf("failed to generate image embedding for %s: %v", pd.ProductReferenceID, err)
				return
			}

			labelVector := embeddings.GenerateLabelVector(pd.Labels, ic.EmbeddingsModel.LabelSet)
			combinedEmbedding := embeddings.CombineEmbeddings(imageEmbedding, labelVector)

			mu.Lock()
			embeddingsList[idx] = combinedEmbedding
			productReferenceIDs[idx] = pd.ProductReferenceID
			mu.Unlock()

			log.Printf("Successfully created embedding for product %s", pd.ProductReferenceID)
		}(i, product)
	}

	wg.Wait()
	close(errChan)

	// Check for any errors that occurred during embedding generation
	for err := range errChan {
		if err != nil {
			log.Printf("Error during embedding generation: %v", err)
			return nil, nil, err
		}
	}

	return embeddingsList, productReferenceIDs, nil
}

func (ic *ImageCluster) PrepareClusterDetails(clusters map[int][]string, productDetails []models.CombinedProductDetails) map[string]models.ClusterDetails {
	clusterDetails := make(map[string]models.ClusterDetails)
	log.Printf("Preparing details for %d clusters", len(clusters))

	for clusterID, products := range clusters {
		clusterKey := fmt.Sprintf("Cluster-%d", clusterID)
		log.Printf("Processing %s with %d products", clusterKey, len(products))

		details := models.NewClusterDetails()
		details.ProductReferenceIDs = products

		labelsSet := make(map[string]struct{})
		var images []string

		for _, pid := range details.ProductReferenceIDs {
			product := models.ProductDetailsMap(pid, productDetails)
			if product != nil {
				for _, label := range product.Labels {
					labelsSet[label] = struct{}{}
				}
				if product.ImagePath != "" {
					imageFilename := filepath.Base(product.ImagePath)
					images = append(images, imageFilename)
				}
			}
		}

		labelsList := make([]string, 0, len(labelsSet))
		for label := range labelsSet {
			labelsList = append(labelsList, label)
		}
		aggregatedLabels := strings.Join(labelsList, ", ")
		details.Labels = aggregatedLabels
		details.Images = images

		log.Printf("Generating AI service outputs for %s", clusterKey)
		modelOutputs := ai.GenerateTitleAndCatchyPhraseMultiService(aggregatedLabels, 3)

		for _, output := range modelOutputs {
			serviceOutput := models.ServiceOutput{
				ServiceName:  output.ServiceName,
				Title:        output.Title,
				CatchyPhrase: output.CatchyPhrase,
			}
			details.SetServiceOutput(serviceOutput)

			if output.ServiceName == "Claude 3" {
				details.Title = output.Title
				details.CatchyPhrase = output.CatchyPhrase
			}
		}

		clusterDetails[clusterKey] = details
		log.Printf("Completed processing for %s", clusterKey)
	}

	return clusterDetails
}

func getProductRefIDs(productDetails []models.CombinedProductDetails) []string {
	productRefIDs := make([]string, len(productDetails))
	for i, pd := range productDetails {
		productRefIDs[i] = pd.ProductReferenceID
	}
	return productRefIDs
}

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

type ItemDetails struct {
	ID        string
	ImagePath string
	Labels    []string
}

func NewImageCluster(minClusterSize, maxClusterSize int, tempDir string) (*ImageCluster, error) {
	log.Printf("Initializing ImageCluster with min=%d, max=%d clusters", minClusterSize, maxClusterSize)

	appCtx := &embeddings.AppContext{
		ImageDir:      filepath.Join(tempDir, "images"),
		CacheDir:      filepath.Join(tempDir, "cache"),
		LabelSet:      make(map[string]int),
		LabelsMapping: make(map[string][]string),
	}

	rekogSvc, err := rekognition.NewRekognitionService("us-east-1", appCtx.CacheDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize RekognitionService: %v", err)
	}

	modelPath := "resnet50-v1-7.onnx"
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

	if err := ic.createDirectories(); err != nil {
		return nil, "", err
	}

	itemDetails, err := ic.processImages(uploadedImages)
	if err != nil {
		return nil, "", err
	}

	err = embeddings.BuildLabelSet(getItemIDs(itemDetails), ic.RekognitionSvc, ic.EmbeddingsModel)
	if err != nil {
		return nil, "", fmt.Errorf("failed to build label set: %v", err)
	}

	embeddingsList, itemIDs, err := ic.createEmbeddings(itemDetails)
	if err != nil {
		return nil, "", err
	}

	clusters, success := clustering.PerformClusteringWithConstraints(
		embeddingsList,
		itemIDs,
		ic.MinClusterSize,
		ic.MaxClusterSize,
	)
	if !success {
		return nil, "", fmt.Errorf("clustering failed")
	}

	clusterDetails := ic.prepareClusterDetails(clusters, itemDetails)

	htmlOutputPath, err := utils.GenerateHTMLOutput(clusterDetails, ic.TempDir)
	if err != nil {
		return nil, "", fmt.Errorf("failed to generate HTML output: %v", err)
	}

	log.Printf("Completed clustering in %v", time.Since(startTime))
	return clusterDetails, htmlOutputPath, nil
}

func (ic *ImageCluster) createDirectories() error {
	dirs := []string{ic.EmbeddingsModel.ImageDir, ic.EmbeddingsModel.CacheDir}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %v", dir, err)
		}
	}
	return nil
}

func (ic *ImageCluster) processImages(uploadedImages []models.UploadedImage) ([]ItemDetails, error) {
	itemDetails := make([]ItemDetails, len(uploadedImages))

	for i, img := range uploadedImages {
		imagePath := filepath.Join(ic.EmbeddingsModel.ImageDir, img.Filename)
		if err := os.WriteFile(imagePath, img.Data, 0644); err != nil {
			return nil, fmt.Errorf("failed to save image %s: %v", img.Filename, err)
		}

		labels, err := ic.RekognitionSvc.DetectLabels(imagePath, 10, 75.0)
		if err != nil {
			return nil, fmt.Errorf("failed to detect labels for %s: %v", img.Filename, err)
		}

		labelNames := make([]string, len(labels))
		for j, label := range labels {
			labelNames[j] = *label.Name
		}

		itemDetails[i] = ItemDetails{
			ID:        fmt.Sprintf("img_%d", i),
			ImagePath: imagePath,
			Labels:    labelNames,
		}
	}

	return itemDetails, nil
}

func (ic *ImageCluster) createEmbeddings(items []ItemDetails) ([][]float32, []string, error) {
	embeddingsList := make([][]float32, len(items))
	itemIDs := make([]string, len(items))
	var mu sync.Mutex
	var wg sync.WaitGroup
	errChan := make(chan error, len(items))

	for i, item := range items {
		wg.Add(1)
		go func(idx int, item ItemDetails) {
			defer wg.Done()

			imageEmbedding, err := embeddings.GetImageEmbedding(ic.EmbeddingsModel, item.ImagePath)
			if err != nil {
				errChan <- fmt.Errorf("failed to generate embedding for %s: %v", item.ID, err)
				return
			}

			labelVector := embeddings.GenerateLabelVector(item.Labels, ic.EmbeddingsModel.LabelSet)
			combinedEmbedding := embeddings.CombineEmbeddings(imageEmbedding, labelVector)

			mu.Lock()
			embeddingsList[idx] = combinedEmbedding
			itemIDs[idx] = item.ID
			mu.Unlock()
		}(i, item)
	}

	wg.Wait()
	close(errChan)

	if err := <-errChan; err != nil {
		return nil, nil, err
	}

	return embeddingsList, itemIDs, nil
}

func (ic *ImageCluster) prepareClusterDetails(clusters map[int][]string, items []ItemDetails) map[string]models.ClusterDetails {
	clusterDetails := make(map[string]models.ClusterDetails)
	itemMap := makeItemMap(items)

	for clusterID, itemIDs := range clusters {
		clusterKey := fmt.Sprintf("Cluster-%d", clusterID)
		var details models.ClusterDetails
		details = details.Init()

		labelsSet := make(map[string]struct{})
		var images []string

		for _, id := range itemIDs {
			if item, exists := itemMap[id]; exists {
				for _, label := range item.Labels {
					labelsSet[label] = struct{}{}
				}
				images = append(images, filepath.Base(item.ImagePath))
			}
		}

		details.Labels = formatLabels(labelsSet)
		details.Images = images

		modelOutputs := ai.GenerateTitleAndCatchyPhraseMultiService(details.Labels, 3)
		for _, output := range modelOutputs {
			details.SetServiceOutput(models.ServiceOutput{
				ServiceName:  output.ServiceName,
				Title:        output.Title,
				CatchyPhrase: output.CatchyPhrase,
			})

			if output.ServiceName == "Claude 3" {
				details.Title = output.Title
				details.CatchyPhrase = output.CatchyPhrase
			}
		}

		clusterDetails[clusterKey] = details
	}

	return clusterDetails
}

func makeItemMap(items []ItemDetails) map[string]ItemDetails {
	itemMap := make(map[string]ItemDetails)
	for _, item := range items {
		itemMap[item.ID] = item
	}
	return itemMap
}

func formatLabels(labelsSet map[string]struct{}) string {
	labels := make([]string, 0, len(labelsSet))
	for label := range labelsSet {
		labels = append(labels, label)
	}
	return strings.Join(labels, ", ")
}

func getItemIDs(items []ItemDetails) []string {
	ids := make([]string, len(items))
	for i, item := range items {
		ids[i] = item.ID
	}
	return ids
}

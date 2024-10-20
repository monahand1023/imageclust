package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"ProductSetter/clustering"
	"ProductSetter/embeddings"
	"ProductSetter/rekognitionservice"
	"ProductSetter/utils"
)

// ProductSetter handles fetching product details, processing images, and clustering.
type ProductSetter struct {
	ProfileID       string
	AuthToken       string
	NumberOfDays    int
	TempDir         string
	RekognitionSvc  *rekognitionservice.RekognitionService
	Client          *http.Client
	EmbeddingsModel interface{}
}

// Run executes the main logic: fetching product details, clustering, and generating HTML.
func (ps *ProductSetter) Run() error {
	startTime := time.Now()
	log.Println("Starting ProductSetter run...")

	// Ensure TempDir exists
	err := os.MkdirAll(ps.TempDir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %v", err)
	}

	// Fetch combined product details
	productDetails, err := ps.fetchProductDetails()
	if err != nil {
		return fmt.Errorf("failed to fetch product details: %v", err)
	}
	log.Printf("Fetched %d product details.", len(productDetails))

	// Create embeddings for all products
	embeddingsList, productReferenceIDs, err := ps.createEmbeddings(productDetails)
	if err != nil {
		return fmt.Errorf("failed to create embeddings: %v", err)
	}
	log.Printf("Created embeddings for %d products.", len(embeddingsList))

	// Perform clustering
	clusters, success := clustering.PerformClusteringWithConstraints(
		embeddingsList,
		productReferenceIDs,
		3,  // minSize
		10, // maxSize
	)
	if !success {
		log.Println("Clustering failed due to constraints.")
		return fmt.Errorf("clustering failed due to constraints")
	}
	log.Printf("Formed %d clusters.", len(clusters))

	// Prepare ClusterDetails for HTML generation
	clusterDetails := make(map[string]utils.ClusterDetails)
	for clusterID, products := range clusters {
		clusterIDStr := fmt.Sprintf("Cluster-%d", clusterID)
		clusterDetails[clusterIDStr] = utils.ClusterDetails{
			Title:               "", // Will be populated below
			CatchyPhrase:        "", // Will be populated below
			Labels:              "", // Will be populated below
			Images:              []string{},
			ProductReferenceIDs: products,
		}
	}

	// Generate titles and catchy phrases (use placeholders for now)
	for clusterID, details := range clusterDetails {
		title, catchyPhrase := "Sample Title", "Sample Catchy Phrase"
		details.Title = title
		details.CatchyPhrase = catchyPhrase

		// Gather labels and images
		labelsSet := make(map[string]struct{})
		for _, pid := range details.ProductReferenceIDs {
			product := productDetailsMap(pid, productDetails)
			for _, label := range product.Labels {
				labelsSet[label] = struct{}{} // No `Name` field, treat labels as plain strings
			}
		}

		labelsList := []string{}
		for label := range labelsSet {
			labelsList = append(labelsList, label)
		}
		details.Labels = strings.Join(labelsList, ", ")

		// Map product reference IDs to image filenames
		for _, pid := range details.ProductReferenceIDs {
			imagePath := productDetailsMap(pid, productDetails).ImagePath
			imageFilename := filepath.Base(imagePath)
			details.Images = append(details.Images, imageFilename)
		}

		clusterDetails[clusterID] = details
	}

	// Generate the HTML output
	htmlOutputPath, err := utils.GenerateHTMLOutput(
		clusterDetails,
		ps.TempDir,
		"localhost",
		5003,
	)
	if err != nil {
		return fmt.Errorf("failed to generate HTML output: %v", err)
	}
	log.Printf("HTML output generated successfully. Access it at: file://%s\n", htmlOutputPath)

	log.Printf("Total execution time: %v", time.Since(startTime))
	return nil
}

// fetchProductDetails retrieves and processes product details from the API.
func (ps *ProductSetter) fetchProductDetails() ([]CombinedProductDetails, error) {
	// Implementation remains mostly unchanged
	return nil, nil
}

// createEmbeddings generates embeddings for all products based on their labels.
func (ps *ProductSetter) createEmbeddings(productDetails []CombinedProductDetails) ([][]float64, []string, error) {
	var embeddingsList [][]float64
	var productReferenceIDs []string

	for _, product := range productDetails {
		// Aggregate labels into a single string
		var labels []string
		for _, label := range product.Labels {
			labels = append(labels, label) // No need to dereference label
		}
		labelsStr := strings.Join(labels, ", ")

		// Type assertion for EmbeddingsModel
		embeddingCtx, ok := ps.EmbeddingsModel.(*embeddings.AppContext)
		if !ok {
			return nil, nil, fmt.Errorf("invalid type for embeddings model")
		}

		// Create embedding using the embeddings package
		embedding, err := embeddings.CreateProductEmbedding(labelsStr, embeddingCtx, ps.RekognitionSvc)
		if err != nil {
			log.Printf("Failed to create embedding for %s: %v", product.ProductReferenceID, err)
			continue
		}

		embeddingsList = append(embeddingsList, embedding)
		productReferenceIDs = append(productReferenceIDs, product.ProductReferenceID)
	}

	if len(embeddingsList) == 0 {
		return nil, nil, fmt.Errorf("no embeddings were created")
	}

	return embeddingsList, productReferenceIDs, nil
}

// productDetailsMap retrieves a product's details by its reference ID.
func productDetailsMap(pid string, productDetails []CombinedProductDetails) CombinedProductDetails {
	for _, product := range productDetails {
		if product.ProductReferenceID == pid {
			return product
		}
	}
	return CombinedProductDetails{}
}

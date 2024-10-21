// ProductSetter/productsetter/ProductSetter.go
package productsetter

import (
	"ProductSetter/openai_utils"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"ProductSetter/clustering"
	"ProductSetter/embeddings"
	"ProductSetter/models"
	"ProductSetter/rekognitionservice"
	"ProductSetter/utils"
)

// ProductSetter holds the configuration and dependencies for the application
type ProductSetter struct {
	ProfileID         string
	AuthToken         string
	NumberOfDaysLimit int
	TempDir           string
	Client            *http.Client
	RekognitionSvc    *rekognitionservice.RekognitionService
	EmbeddingsModel   *embeddings.AppContext
	MinClusterSize    int
	MaxClusterSize    int
	Mutex             sync.Mutex
}

// NewProductSetter initializes and returns a new ProductSetter instance
func NewProductSetter(profileID, authToken string, numberOfDaysLimit int, minClusterSize, maxClusterSize int, tempDir string) (*ProductSetter, error) {
	// Initialize HTTP client with timeout
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Initialize AppContext for embeddings (excluding Net for now)
	appCtx := &embeddings.AppContext{
		ImageDir:      filepath.Join(tempDir, "images"),
		CacheDir:      filepath.Join(tempDir, "cache"),
		LabelSet:      make(map[string]int),
		LabelsMapping: make(map[string][]string),
		// Net will be assigned after loading the model
	}

	// Initialize RekognitionService with desired AWS region and cache directory
	rekogSvc, err := rekognitionservice.NewRekognitionService("us-east-1", appCtx.CacheDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize RekognitionService: %v", err)
	}

	// Load pre-trained ResNet50 model (ONNX format)
	modelPath := "resnet50-v1-7.onnx" // Assuming modelPath is based on ProfileID; adjust as needed
	net, err := embeddings.LoadPretrainedModelONNX(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load ResNet50 ONNX model: %v", err)
	}

	// Assign the loaded model to appCtx.Net
	appCtx.Net = net

	// Return the initialized ProductSetter instance
	return &ProductSetter{
		ProfileID:         profileID,
		AuthToken:         authToken,
		NumberOfDaysLimit: numberOfDaysLimit,
		TempDir:           tempDir,
		Client:            client,
		RekognitionSvc:    rekogSvc,
		EmbeddingsModel:   appCtx,
		MinClusterSize:    minClusterSize,
		MaxClusterSize:    maxClusterSize,
	}, nil
}

// Run executes the main workflow of the ProductSetter application
func (ps *ProductSetter) Run() (map[int][]string, string, error) {
	startTime := time.Now()
	log.Println("Starting ProductSetter run...")

	// Ensure necessary directories exist
	err := os.MkdirAll(ps.EmbeddingsModel.ImageDir, 0755)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create image directory: %v", err)
	}

	err = os.MkdirAll(ps.EmbeddingsModel.CacheDir, 0755)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create cache directory: %v", err)
	}

	// Fetch combined product details
	productDetails, err := ps.FetchProductDetails()
	if err != nil {
		return nil, "", fmt.Errorf("failed to fetch product details: %v", err)
	}
	log.Printf("Fetched %d product details.", len(productDetails))

	// Check if any products were fetched
	if len(productDetails) == 0 {
		log.Println("No product details fetched.")
		return nil, "", fmt.Errorf("no products to process")
	}

	// Build Label Set from all product labels
	err = embeddings.BuildLabelSet(getProductRefIDs(productDetails), ps.RekognitionSvc, ps.EmbeddingsModel)
	if err != nil {
		return nil, "", fmt.Errorf("failed to build label set: %v", err)
	}

	// Create embeddings for all products
	embeddingsList, productReferenceIDs, err := ps.CreateEmbeddingsForAllProducts(productDetails)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create embeddings: %v", err)
	}
	log.Printf("Created embeddings for %d products.", len(embeddingsList))

	// Perform clustering with specified constraints
	clusters, success := clustering.PerformClusteringWithConstraints(
		embeddingsList,
		productReferenceIDs,
		ps.MinClusterSize,
		ps.MaxClusterSize,
	)
	if !success {
		log.Println("Clustering failed due to constraints.")
		return nil, "", fmt.Errorf("clustering failed due to constraints")
	}
	log.Printf("Formed %d clusters.", len(clusters))

	// Prepare ClusterDetails for HTML generation
	clusterDetails := ps.PrepareClusterDetails(clusters, productDetails)

	// Generate the HTML output
	htmlOutputPath, err := utils.GenerateHTMLOutput(
		clusterDetails,
		ps.TempDir,
		"localhost", // Default Host, can be parameterized if needed
		5003,        // Default Port, can be parameterized if needed
	)
	if err != nil {
		return nil, "", fmt.Errorf("failed to generate HTML output: %v", err)
	}
	log.Printf("HTML output generated successfully. Access it at: file://%s\n", htmlOutputPath)

	log.Printf("Total execution time: %v", time.Since(startTime))
	return clusters, htmlOutputPath, nil
}

// CreateEmbeddingsForAllProducts generates embeddings for all products concurrently
func (ps *ProductSetter) CreateEmbeddingsForAllProducts(productDetails []models.CombinedProductDetails) ([][]float32, []string, error) {
	embeddingsList := make([][]float32, len(productDetails))
	productReferenceIDs := make([]string, len(productDetails))
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstErr error
	errChan := make(chan error, len(productDetails))

	for i, product := range productDetails {
		wg.Add(1)
		go func(idx int, pd models.CombinedProductDetails) {
			defer wg.Done()

			// Generate embedding
			embedding, err := embeddings.GetImageEmbedding(ps.EmbeddingsModel, pd.ImagePath)
			if err != nil {
				errChan <- fmt.Errorf("failed to generate embedding for %s: %v", pd.ProductReferenceID, err)
				return
			}

			// Append label vector if needed
			labelVector := embeddings.GenerateLabelVector(pd.Labels, ps.EmbeddingsModel.LabelSet)
			combinedEmbedding := embeddings.CombineEmbeddings(embedding, labelVector) // Ensure this returns []float32

			// Assign to embeddings list
			mu.Lock()
			embeddingsList[idx] = combinedEmbedding
			productReferenceIDs[idx] = pd.ProductReferenceID
			mu.Unlock()
		}(i, product)
	}

	// Wait for all goroutines to finish
	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if firstErr == nil {
			firstErr = err
		}
		log.Println(err)
	}
	if firstErr != nil {
		return nil, nil, firstErr
	}

	return embeddingsList, productReferenceIDs, nil
}

// FetchProductDetails retrieves product details from the API, downloads images, and fetches labels
func (ps *ProductSetter) FetchProductDetails() ([]models.CombinedProductDetails, error) {
	combinedProductDetailsList := make([]models.CombinedProductDetails, 0)
	var mu sync.Mutex

	// Construct the API URL
	baseURL := fmt.Sprintf("https://qa-api-gateway.rewardstyle.com/api/pub/v3/activities/products/profiles/%s?limit=20", ps.ProfileID)
	nextToken := ""

	for {
		// Append next token if present
		activitiesURL := baseURL
		if nextToken != "" {
			activitiesURL += fmt.Sprintf("&next=%s", utils.URLEncode(nextToken))
		}

		// Build the HTTP GET request
		req, err := http.NewRequest("GET", activitiesURL, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create activities request: %v", err)
		}

		// Send the request
		resp, err := ps.Client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to send activities request: %v", err)
		}

		// Ensure response body is closed
		defer resp.Body.Close()

		// Check for successful response
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("activities request failed with status code: %d", resp.StatusCode)
		}

		// Parse the JSON response
		var activitiesResp struct {
			Activities []struct {
				ProductRefID string `json:"product_ref_id"`
			} `json:"activities"`
			Meta struct {
				Next string `json:"next"`
			} `json:"meta"`
		}

		err = json.NewDecoder(resp.Body).Decode(&activitiesResp)
		if err != nil {
			return nil, fmt.Errorf("failed to decode activities response: %v", err)
		}

		// Check if there are activities to process
		if len(activitiesResp.Activities) == 0 {
			break
		}

		// Process each activity concurrently
		var wg sync.WaitGroup
		for _, activity := range activitiesResp.Activities {
			wg.Add(1)
			go func(productRefID string) {
				defer wg.Done()

				// Fetch product details
				productDetail, err := ps.fetchProductDetail(productRefID)
				if err != nil {
					log.Printf("Error fetching product detail for %s: %v", productRefID, err)
					return // Skip this product on error
				}

				// Detect labels using AWS Rekognition
				labels, err := ps.RekognitionSvc.DetectLabels(productDetail.ImagePath, 10, 75.0)
				if err != nil {
					log.Printf("Error detecting labels for %s: %v", productRefID, err)
					return // Skip this product on error
				}

				// Convert labels to []string
				labelNames := make([]string, len(labels))
				for i, label := range labels {
					labelNames[i] = *label.Name
				}
				productDetail.Labels = labelNames

				// Only append the product if no errors occurred during the fetch or label detection process
				mu.Lock()
				combinedProductDetailsList = append(combinedProductDetailsList, *productDetail)
				mu.Unlock()
			}(activity.ProductRefID)
		}

		// Wait for all goroutines to finish
		wg.Wait()

		// Check if there is a next token for pagination
		nextToken = activitiesResp.Meta.Next
		if nextToken == "" {
			break
		}
	}

	// Return the list of valid product details
	return combinedProductDetailsList, nil
}

// PrepareClusterDetails organizes cluster information for HTML generation
func (ps *ProductSetter) PrepareClusterDetails(clusters map[int][]string, productDetails []models.CombinedProductDetails) map[string]utils.ClusterDetails {
	clusterDetails := make(map[string]utils.ClusterDetails)

	for clusterID, products := range clusters {
		clusterKey := fmt.Sprintf("Cluster-%d", clusterID)
		clusterDetails[clusterKey] = utils.ClusterDetails{
			Title:               "",
			CatchyPhrase:        "",
			Labels:              "",
			Images:              []string{},
			ProductReferenceIDs: products,
		}
	}

	// Populate each cluster's details
	for clusterKey, details := range clusterDetails {
		// Aggregate labels
		labelsSet := make(map[string]struct{})
		for _, pid := range details.ProductReferenceIDs {
			product := models.ProductDetailsMap(pid, productDetails)
			if product.ProductReferenceID != "" {
				for _, label := range product.Labels {
					labelsSet[label] = struct{}{}
				}
			}
		}

		// Convert labels set to a comma-separated string
		labelsList := make([]string, 0, len(labelsSet))
		for label := range labelsSet {
			labelsList = append(labelsList, label)
		}
		aggregatedLabels := strings.Join(labelsList, ", ")
		details.Labels = aggregatedLabels

		// **Generate Title and Catchy Phrase using GPT**
		title, catchyPhrase := openai_utils.GenerateTitleAndCatchyPhrase(aggregatedLabels, 3)
		details.Title = title
		details.CatchyPhrase = catchyPhrase

		// Gather image filenames
		for _, pid := range details.ProductReferenceIDs {
			product := models.ProductDetailsMap(pid, productDetails)
			if product.ProductReferenceID != "" {
				imageFilename := filepath.Base(product.ImagePath)
				details.Images = append(details.Images, imageFilename)
			}
		}

		// Update the cluster details
		clusterDetails[clusterKey] = details
	}

	return clusterDetails
}

// fetchProductDetail retrieves detailed information for a single product and downloads its image
func (ps *ProductSetter) fetchProductDetail(productRefID string) (*models.CombinedProductDetails, error) {
	// Encode the productRefID
	encodedProductRefID := utils.URLEncode(productRefID)

	// Construct the product detail API URL
	productDetailURL := fmt.Sprintf("http://qa-rs-product-service.rslocal/v1/retailer_product_references?ids[]=%s", encodedProductRefID)

	// Build the HTTP GET request
	req, err := http.NewRequest("GET", productDetailURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create product detail request: %v", err)
	}

	// Send the request
	resp, err := ps.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send product detail request: %v", err)
	}
	defer resp.Body.Close()

	// Check for successful response
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("product detail request failed with status code: %d", resp.StatusCode)
	}

	// Parse the JSON response
	var productDetailResp struct {
		RetailerProducts      []json.RawMessage `json:"retailer_products"`
		RetailerProductImages []json.RawMessage `json:"retailer_product_images"`
	}

	err = json.NewDecoder(resp.Body).Decode(&productDetailResp)
	if err != nil {
		return nil, fmt.Errorf("failed to decode product detail response: %v", err)
	}

	// Ensure there are products and images
	if len(productDetailResp.RetailerProducts) == 0 || len(productDetailResp.RetailerProductImages) == 0 {
		return nil, fmt.Errorf("no retailer_products or retailer_product_images found for %s", productRefID)
	}

	// Extract product details
	var retailerProduct struct {
		RetailerID  string `json:"retailer_id"`
		Price       string `json:"price"` // Updated: expects a string
		Description string `json:"description"`
		Title       string `json:"title"`
		UpdatedAt   string `json:"updated_at"`
	}
	err = json.Unmarshal(productDetailResp.RetailerProducts[0], &retailerProduct)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal retailer_product: %v", err)
	}

	// Convert Price from string to float32
	priceFloat, err := strconv.ParseFloat(retailerProduct.Price, 32)
	if err != nil {
		return nil, fmt.Errorf("invalid price format for %s: %v", productRefID, err)
	}

	// Extract product image URL
	var retailerProductImage struct {
		URL string `json:"url"`
	}
	err = json.Unmarshal(productDetailResp.RetailerProductImages[0], &retailerProductImage)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal retailer_product_image: %v", err)
	}

	// Download the image
	imagePath, err := ps.downloadImage(retailerProductImage.URL, productRefID)
	if err != nil {
		return nil, fmt.Errorf("failed to download image for %s: %v", productRefID, err)
	}

	// Create CombinedProductDetails instance
	combinedProduct := models.NewCombinedProductDetails(
		productRefID,
		retailerProduct.RetailerID,
		float32(priceFloat),
		imagePath,
		retailerProduct.Description,
		retailerProduct.Title,
		retailerProduct.UpdatedAt,
	)

	return combinedProduct, nil
}

// downloadImage downloads an image from the given URL and saves it to the image directory
func (ps *ProductSetter) downloadImage(imageURL, productRefID string) (string, error) {
	if imageURL == "" {
		return "", fmt.Errorf("empty image URL for %s", productRefID)
	}

	// Ensure the images directory exists
	imagesDir := ps.EmbeddingsModel.ImageDir
	if _, err := os.Stat(imagesDir); os.IsNotExist(err) {
		err := os.MkdirAll(imagesDir, 0755) // Create the images directory if it doesn't exist
		if err != nil {
			return "", fmt.Errorf("failed to create images directory: %v", err)
		}
	}

	// Build the HTTP GET request for the image
	req, err := http.NewRequest("GET", imageURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create image download request: %v", err)
	}

	// Send the request
	resp, err := ps.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send image download request: %v", err)
	}
	defer resp.Body.Close()

	// Check for successful response
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("image download failed with status code: %d", resp.StatusCode)
	}

	// Generate a sanitized file path
	sanitizedProductID := utils.SanitizeFilename(productRefID)
	imageFilename := fmt.Sprintf("%s.jpg", sanitizedProductID)
	imagePath := filepath.Join(ps.EmbeddingsModel.ImageDir, imageFilename)

	// Create the image file
	file, err := os.Create(imagePath)
	if err != nil {
		return "", fmt.Errorf("failed to create image file: %v", err)
	}
	defer file.Close()

	// Copy the image data to the file
	_, err = io.Copy(file, resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to save image file: %v", err)
	}

	return imagePath, nil
}

// getProductRefIDs extracts product reference IDs from product details.
func getProductRefIDs(productDetails []models.CombinedProductDetails) []string {
	productRefIDs := make([]string, len(productDetails))
	for i, pd := range productDetails {
		productRefIDs[i] = pd.ProductReferenceID
	}
	return productRefIDs
}

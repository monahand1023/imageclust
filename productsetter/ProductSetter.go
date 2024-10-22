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
	modelPath := "resnet50-v1-7.onnx" // Adjust as needed
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

	// Step 1: Fetch combined product details
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

	// Step 2: Build Label Set from all product labels
	err = embeddings.BuildLabelSet(getProductRefIDs(productDetails), ps.RekognitionSvc, ps.EmbeddingsModel)
	if err != nil {
		return nil, "", fmt.Errorf("failed to build label set: %v", err)
	}

	// Step 3: Create embeddings for all products
	embeddingsList, productReferenceIDs, err := ps.CreateEmbeddingsForAllProducts(productDetails)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create embeddings: %v", err)
	}
	log.Printf("Created embeddings for %d products.", len(embeddingsList))

	// Step 4: Perform clustering with specified constraints
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

	// Step 5: Prepare ClusterDetails for GPT and HTML generation
	clusterDetails := ps.PrepareClusterDetails(clusters, productDetails)

	// Step 6: Generate the HTML output with GPT-enhanced descriptions
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

	// Log total execution time
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

			// Generate image embedding
			imageEmbedding, err := embeddings.GetImageEmbedding(ps.EmbeddingsModel, pd.ImagePath)
			if err != nil {
				errChan <- fmt.Errorf("failed to generate image embedding for %s: %v", pd.ProductReferenceID, err)
				return
			}

			// Include Labels
			labelVector := embeddings.GenerateLabelVector(pd.Labels, ps.EmbeddingsModel.LabelSet)

			// Combine image embedding and label vector
			combinedEmbedding := embeddings.CombineEmbeddings(imageEmbedding, labelVector)

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
					return
				}

				// Detect labels using AWS Rekognition
				labels, err := ps.RekognitionSvc.DetectLabels(productDetail.ImagePath, 10, 75.0)
				if err != nil {
					log.Printf("Error detecting labels for %s: %v", productRefID, err)
					return
				}

				// Convert labels to []string
				labelNames := make([]string, len(labels))
				for i, label := range labels {
					labelNames[i] = *label.Name
				}
				productDetail.Labels = labelNames

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

	return combinedProductDetailsList, nil
}

// PrepareClusterDetails aggregates cluster information and interacts with GPT
func (ps *ProductSetter) PrepareClusterDetails(clusters map[int][]string, productDetails []models.CombinedProductDetails) map[string]models.ClusterDetails {
	clusterDetails := make(map[string]models.ClusterDetails)

	for clusterID, products := range clusters {
		clusterKey := fmt.Sprintf("Cluster-%d", clusterID)
		clusterDetails[clusterKey] = models.ClusterDetails{
			ProductReferenceIDs: products,
			Images:              []string{},
			Labels:              "",
			Title:               "",
			CatchyPhrase:        "",
		}
	}

	// Populate each cluster's details
	for clusterKey, details := range clusterDetails {
		// Aggregate Labels
		labelsSet := make(map[string]struct{})
		titles := []string{}
		descriptions := []string{}
		for _, pid := range details.ProductReferenceIDs {
			product := models.ProductDetailsMap(pid, productDetails)
			if product != nil {
				for _, label := range product.Labels {
					labelsSet[label] = struct{}{}
				}
				if product.Title != "" {
					titles = append(titles, product.Title)
				}
				if product.Description != "" {
					descriptions = append(descriptions, product.Description)
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

		aggregatedTitles := utils.CleanText(strings.Join(titles, ", "))
		aggregatedDescriptions := utils.CleanText(strings.Join(descriptions, ", "))

		// Combine aggregated features for GPT
		aggregatedFeatures := fmt.Sprintf("Labels: %s. Titles: %s. Descriptions: %s.", aggregatedLabels, aggregatedTitles, aggregatedDescriptions)

		// Generate title and catchy phrase using GPT
		title, catchyPhrase := openai_utils.GenerateTitleAndCatchyPhrase(aggregatedFeatures, 3)

		// Assign GPT-generated title and catchy phrase
		details.Title = title
		details.CatchyPhrase = catchyPhrase

		// Gather image filenames
		for _, pid := range details.ProductReferenceIDs {
			product := models.ProductDetailsMap(pid, productDetails)
			if product != nil && product.ImagePath != "" {
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
	encodedProductRefID := utils.URLEncode(productRefID)

	productDetailURL := fmt.Sprintf("http://qa-rs-product-service.rslocal/v1/retailer_product_references?ids[]=%s", encodedProductRefID)

	req, err := http.NewRequest("GET", productDetailURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create product detail request: %v", err)
	}

	resp, err := ps.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send product detail request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("product detail request failed with status code: %d", resp.StatusCode)
	}

	var productDetailResp struct {
		RetailerProducts      []json.RawMessage `json:"retailer_products"`
		RetailerProductImages []json.RawMessage `json:"retailer_product_images"`
	}

	err = json.NewDecoder(resp.Body).Decode(&productDetailResp)
	if err != nil {
		return nil, fmt.Errorf("failed to decode product detail response: %v", err)
	}

	if len(productDetailResp.RetailerProducts) == 0 || len(productDetailResp.RetailerProductImages) == 0 {
		return nil, fmt.Errorf("no retailer_products or retailer_product_images found for %s", productRefID)
	}

	var retailerProduct struct {
		RetailerID  string `json:"retailer_id"`
		Price       string `json:"price"` // Expects a string
		Description string `json:"description"`
		Title       string `json:"title"`
		UpdatedAt   string `json:"updated_at"`
	}
	err = json.Unmarshal(productDetailResp.RetailerProducts[0], &retailerProduct)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal retailer_product: %v", err)
	}

	priceFloat, err := strconv.ParseFloat(retailerProduct.Price, 32)
	if err != nil {
		return nil, fmt.Errorf("invalid price format for %s: %v", productRefID, err)
	}

	var retailerProductImage struct {
		URL string `json:"url"`
	}
	err = json.Unmarshal(productDetailResp.RetailerProductImages[0], &retailerProductImage)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal retailer_product_image: %v", err)
	}

	imagePath, err := ps.downloadImage(retailerProductImage.URL, productRefID)
	if err != nil {
		return nil, fmt.Errorf("failed to download image for %s: %v", productRefID, err)
	}

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

	imagesDir := ps.EmbeddingsModel.ImageDir
	if _, err := os.Stat(imagesDir); os.IsNotExist(err) {
		err := os.MkdirAll(imagesDir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create images directory: %v", err)
		}
	}

	req, err := http.NewRequest("GET", imageURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create image download request: %v", err)
	}

	resp, err := ps.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send image download request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("image download failed with status code: %d", resp.StatusCode)
	}

	sanitizedProductID := utils.SanitizeFilename(productRefID)
	imageFilename := fmt.Sprintf("%s.jpg", sanitizedProductID)
	imagePath := filepath.Join(ps.EmbeddingsModel.ImageDir, imageFilename)

	file, err := os.Create(imagePath)
	if err != nil {
		return "", fmt.Errorf("failed to create image file: %v", err)
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to save image file: %v", err)
	}

	return imagePath, nil
}

// getProductRefIDs extracts product reference IDs from product details
func getProductRefIDs(productDetails []models.CombinedProductDetails) []string {
	productRefIDs := make([]string, len(productDetails))
	for i, pd := range productDetails {
		productRefIDs[i] = pd.ProductReferenceID
	}
	return productRefIDs
}

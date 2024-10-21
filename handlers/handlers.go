package handlers

import (
	"bytes"
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

	"ProductSetter/clustering"
	"ProductSetter/embeddings"
	"ProductSetter/openai_utils"
	"ProductSetter/rekognitionservice"
	"ProductSetter/utils"
)

// AppContext holds the application-wide context and shared resources.
type AppContext struct {
	Config               map[string]interface{}
	CurrentEmbeddings    [][]float32
	CurrentProductRefIDs []string
	Clusters             map[string]utils.ClusterDetails // Updated to use utils.ClusterDetails
	LabelsMapping        map[string][]string
	TitleMapping         map[string]string
	DescriptionMapping   map[string]string
	UpdatedAtMapping     map[string]string
	ImageDir             string
	CacheDir             string
	Mutex                sync.Mutex
	ProfileID            string
	AuthToken            string
	MinClusterSize       int
	MaxClusterSize       int
	Model                interface{}
	RekognitionSvc       *rekognitionservice.RekognitionService
	ImageFilenameMapping map[string]string
}

// NewAppContext initializes and returns a new AppContext.
func NewAppContext() *AppContext {
	return &AppContext{
		Config:               make(map[string]interface{}),
		LabelsMapping:        make(map[string][]string),
		TitleMapping:         make(map[string]string),
		DescriptionMapping:   make(map[string]string),
		UpdatedAtMapping:     make(map[string]string),
		Clusters:             make(map[string]utils.ClusterDetails),
		MinClusterSize:       3,
		MaxClusterSize:       10,
		ImageFilenameMapping: make(map[string]string),
	}
}

var appCtx = NewAppContext()

// ClusterAndGenerateHandler handles the clustering and response generation.
func ClusterAndGenerateHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("Route /cluster_and_generate was called")

	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(32 << 20) // 32MB max memory
	if err != nil {
		log.Printf("Error parsing multipart form: %v", err)
		http.Error(w, "Failed to parse multipart form data", http.StatusBadRequest)
		return
	}

	// Get 'data' field from the form
	jsonData := r.FormValue("data")
	if jsonData == "" {
		log.Println("Missing 'data' field in form data.")
		http.Error(w, "Missing 'data' field in form data", http.StatusBadRequest)
		return
	}

	// Parse JSON data
	var data map[string]interface{}
	err = json.Unmarshal([]byte(jsonData), &data)
	if err != nil {
		log.Printf("Error parsing JSON data: %v", err)
		http.Error(w, "Invalid JSON data", http.StatusBadRequest)
		return
	}

	// Extract and set configurations
	err = extractConfigurations(data, appCtx)
	if err != nil {
		log.Printf("Error in configurations: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Initialize RekognitionService with region and cache directory
	appCtx.RekognitionSvc, err = rekognitionservice.NewRekognitionService("us-east-1", appCtx.CacheDir)
	if err != nil {
		http.Error(w, "Failed to initialize RekognitionService", http.StatusInternalServerError)
		return
	}

	// Load the pre-trained model
	modelPath, ok := data["model_path"].(string)
	if !ok || modelPath == "" {
		log.Println("Missing or invalid 'model_path' in data.")
		http.Error(w, "Missing or invalid 'model_path' in data.", http.StatusBadRequest)
		return
	}
	appCtx.Model, err = embeddings.LoadPretrainedModelONNX(modelPath)
	if err != nil {
		http.Error(w, "Failed to load pre-trained model", http.StatusInternalServerError)
		return
	}

	// Handle image files
	imageFiles := r.MultipartForm.File["images"]
	if len(imageFiles) == 0 {
		log.Println("No images uploaded.")
		http.Error(w, "No images uploaded.", http.StatusBadRequest)
		return
	}

	productReferenceIDs := []string{}

	// Ensure image directory exists
	err = os.MkdirAll(appCtx.ImageDir, 0755)
	if err != nil {
		log.Printf("Failed to create image directory: %v", err)
		http.Error(w, "Failed to create image directory.", http.StatusInternalServerError)
		return
	}

	// Save uploaded images to image directory
	for _, fileHeader := range imageFiles {
		file, err := fileHeader.Open()
		if err != nil {
			log.Printf("Error opening uploaded file %s: %v", fileHeader.Filename, err)
			continue
		}
		defer file.Close()

		// Extract product reference ID from filename (assuming filename is productRefID.jpg)
		productRefID := extractProductRefID(fileHeader.Filename)
		if productRefID == "" {
			log.Printf("Failed to extract productRefID from filename: %s", fileHeader.Filename)
			continue
		}

		// Save the file
		filename := filepath.Base(fileHeader.Filename)
		imagePath := filepath.Join(appCtx.ImageDir, filename)
		outFile, err := os.Create(imagePath)
		if err != nil {
			log.Printf("Error creating file %s: %v", imagePath, err)
			continue
		}

		_, err = io.Copy(outFile, file)
		if err != nil {
			log.Printf("Error saving file %s: %v", imagePath, err)
			outFile.Close()
			continue
		}
		outFile.Close()

		appCtx.ImageFilenameMapping[productRefID] = filename
		productReferenceIDs = append(productReferenceIDs, productRefID)
		log.Printf("Saved image for product ID %s at %s", productRefID, imagePath)
	}

	if len(productReferenceIDs) == 0 {
		log.Println("No valid images were processed.")
		http.Error(w, "No valid images were processed.", http.StatusBadRequest)
		return
	}

	// Build label set for one-hot encoding
	embeddingsAppCtx := &embeddings.AppContext{
		LabelsMapping: appCtx.LabelsMapping,
		// Initialize other necessary fields if required by embeddings.BuildLabelSet
	}
	err = embeddings.BuildLabelSet(productReferenceIDs, appCtx.RekognitionSvc, embeddingsAppCtx)
	if err != nil {
		log.Printf("Error building label set: %v", err)
		http.Error(w, "Error building label set.", http.StatusInternalServerError)
		return
	}

	// Generate embeddings for all products
	embeddingsList := [][]float32{}
	for _, productRefID := range productReferenceIDs {
		combinedEmbedding, err := embeddings.CreateProductEmbedding(
			productRefID,
			embeddingsAppCtx,
			appCtx.RekognitionSvc,
		)
		if err != nil {
			log.Printf("Error creating embedding for product %s: %v", productRefID, err)
			continue
		}
		embeddingsList = append(embeddingsList, combinedEmbedding)
	}

	if len(embeddingsList) == 0 {
		log.Println("No embeddings were computed.")
		http.Error(w, "No embeddings to process.", http.StatusBadRequest)
		return
	}

	// Update AppContext with current embeddings and product reference IDs
	appCtx.Mutex.Lock()
	appCtx.CurrentEmbeddings = embeddingsList
	appCtx.CurrentProductRefIDs = productReferenceIDs
	appCtx.Mutex.Unlock()

	// Perform clustering
	clusters, success := clustering.PerformClusteringWithConstraints(
		embeddingsList,
		productReferenceIDs,
		appCtx.MinClusterSize,
		appCtx.MaxClusterSize,
	)
	if !success {
		log.Println("Clustering failed due to constraints.")
		http.Error(w, "Clustering failed due to constraints.", http.StatusInternalServerError)
		return
	}

	log.Printf("Formed %d clusters.", len(clusters))

	// Prepare the response payload
	responsePayload := map[string]interface{}{
		"clusters": map[string]utils.ClusterDetails{},
	}

	for clusterID, products := range clusters {
		clusterIDStr := strconv.Itoa(clusterID)
		aggregatedText := aggregateFeatures(clusterIDStr, products, appCtx)

		aggregatedTextClean := utils.CleanText(aggregatedText)

		// Generate title and catchy phrase using OpenAI
		title, catchyPhrase := openai_utils.GenerateTitleAndCatchyPhrase(aggregatedTextClean, 3)
		if title == "No Title" {
			log.Println("Failed to generate title using OpenAI API")
		}
		if catchyPhrase == "No phrase available" {
			log.Println("Failed to generate catchy phrase using OpenAI API")
		}

		// Map product_reference_ids to image filenames
		images := []string{}
		for _, pid := range products {
			filename, ok := appCtx.ImageFilenameMapping[pid]
			if ok {
				images = append(images, filename)
			}
		}

		// Gather labels
		labelsSet := make(map[string]struct{})
		for _, pid := range products {
			labels, ok := appCtx.LabelsMapping[pid]
			if ok {
				for _, label := range labels {
					labelsSet[label] = struct{}{}
				}
			}
		}
		labelsList := []string{}
		for label := range labelsSet {
			labelsList = append(labelsList, label)
		}

		// Store cluster details using utils.ClusterDetails
		clusterDetail := utils.ClusterDetails{
			Title:               title,
			CatchyPhrase:        catchyPhrase,
			Labels:              strings.Join(labelsList, ", "),
			Images:              images,
			ProductReferenceIDs: products,
		}

		responsePayload["clusters"].(map[string]utils.ClusterDetails)[clusterIDStr] = clusterDetail
		appCtx.Clusters[clusterIDStr] = clusterDetail
	}

	// Prepare the final response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(responsePayload)
}

// PublishHandler handles publishing clusters to an external API.
func PublishHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("Route /publish was called")

	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// Parse JSON body
	var data map[string]interface{}
	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		log.Printf("Error parsing JSON data: %v", err)
		http.Error(w, "Invalid JSON data", http.StatusBadRequest)
		return
	}

	clusterID, ok := data["cluster_id"].(string)
	if !ok || clusterID == "" {
		log.Println("Missing 'cluster_id' in data.")
		http.Error(w, "Missing 'cluster_id' in data.", http.StatusBadRequest)
		return
	}

	// Retrieve the cluster details from app context
	cluster, ok := appCtx.Clusters[clusterID]
	if !ok {
		http.Error(w, fmt.Sprintf("Cluster ID %s not found", clusterID), http.StatusNotFound)
		return
	}

	// Construct the payload for the publish API
	apiURL := "https://qa-api-gateway.rewardstyle.com/api/pub/v1/shops/create_shop_product_collection"
	payload := map[string]interface{}{
		"add_product_reference_ids": cluster.ProductReferenceIDs,
		"subtype":                   "",
		"title":                     cluster.Title,
		"description":               cluster.CatchyPhrase,
		"attributes":                map[string]interface{}{}, // Empty JSON object
		"profile_id":                appCtx.ProfileID,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Printf("Error marshaling payload: %v", err)
		http.Error(w, "Error preparing request payload.", http.StatusInternalServerError)
		return
	}

	// Prepare the HTTP request
	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(payloadBytes))
	if err != nil {
		log.Printf("Error creating HTTP request: %v", err)
		http.Error(w, "Error creating request.", http.StatusInternalServerError)
		return
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", appCtx.AuthToken))
	req.Header.Set("Content-Type", "application/json")

	// Perform the HTTP POST request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error performing HTTP request: %v", err)
		http.Error(w, "Error performing request.", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Error reading response body: %v", err)
		http.Error(w, "Error reading response.", http.StatusInternalServerError)
		return
	}

	if resp.StatusCode >= 200 && resp.StatusCode <= 299 {
		log.Printf("Shop product collection created successfully for cluster: %s", cluster.Title)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": "Shop product collection created successfully",
		})
	} else {
		log.Printf("Failed to create shop product collection. Status code: %d", resp.StatusCode)
		log.Printf("Response: %s", string(respBody))
		http.Error(w, fmt.Sprintf("Failed to create shop product collection. Status code: %d", resp.StatusCode), resp.StatusCode)
	}
}

// GenerateHTMLOutputHandler handles the generation of HTML output for clusters.
func GenerateHTMLOutputHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("Route /generate_html_output was called")

	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// Parse JSON body
	var data map[string]interface{}
	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		log.Printf("Error parsing JSON data: %v", err)
		http.Error(w, "Invalid JSON data", http.StatusBadRequest)
		return
	}

	clustersData, ok := data["clusters"].(map[string]interface{})
	if !ok {
		http.Error(w, "Missing 'clusters' in the data", http.StatusBadRequest)
		return
	}

	imageDir, ok := data["image_dir"].(string)
	if !ok || imageDir == "" {
		http.Error(w, "Missing 'image_dir' in the data", http.StatusBadRequest)
		return
	}

	// Convert clustersData to map[string]utils.ClusterDetails
	clusterDetails := make(map[string]utils.ClusterDetails)
	for clusterID, clusterInfo := range clustersData {
		if clusterMap, ok := clusterInfo.(map[string]interface{}); ok {
			// Safely extract each field with type assertions
			title, _ := clusterMap["title"].(string)
			catchyPhrase, _ := clusterMap["catchy_phrase"].(string)
			labels, _ := clusterMap["labels"].(string)

			clusterDetail := utils.ClusterDetails{
				Title:        title,
				CatchyPhrase: catchyPhrase,
				Labels:       labels,
			}

			// Handle images
			if images, ok := clusterMap["images"].([]interface{}); ok {
				for _, img := range images {
					if imgStr, ok := img.(string); ok {
						clusterDetail.Images = append(clusterDetail.Images, imgStr)
					}
				}
			}

			// Handle product reference IDs
			if pids, ok := clusterMap["product_reference_ids"].([]interface{}); ok {
				for _, pid := range pids {
					if pidStr, ok := pid.(string); ok {
						clusterDetail.ProductReferenceIDs = append(clusterDetail.ProductReferenceIDs, pidStr)
					}
				}
			}

			clusterDetails[clusterID] = clusterDetail
		}
	}

	// Generate the HTML output using the clusters
	htmlPath, err := utils.GenerateHTMLOutput(
		clusterDetails,
		imageDir,
		"localhost",
		5003,
	)
	if err != nil {
		log.Printf("Error generating HTML output: %v", err)
		http.Error(w, "Error generating HTML output.", http.StatusInternalServerError)
		return
	}

	log.Printf("Generated HTML at path: %s", htmlPath)

	// Prepare response
	responsePayload := map[string]interface{}{
		"html_file_path": htmlPath,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(responsePayload)
}

// Helper Functions

// extractProductRefID extracts the product reference ID from the filename, assuming the file name contains the ID.
func extractProductRefID(filename string) string {
	// Assuming product reference IDs are the file names without the extension
	return strings.TrimSuffix(filename, filepath.Ext(filename))
}

// extractConfigurations extracts configurations and mappings from the JSON data.
func extractConfigurations(data map[string]interface{}, appCtx *AppContext) error {
	// Extract mappings
	appCtx.TitleMapping = parseStringMap(data["title_mapping"])
	appCtx.DescriptionMapping = parseStringMap(data["description_mapping"])
	appCtx.UpdatedAtMapping = parseStringMap(data["updated_at_mapping"])

	// Extract image directory and cache directory
	imageDir, ok := data["image_dir"].(string)
	if !ok || imageDir == "" {
		return fmt.Errorf("Invalid or missing 'image_dir' in data.")
	}
	appCtx.ImageDir = imageDir

	cacheDir, ok := data["cache_dir"].(string)
	if !ok || cacheDir == "" {
		return fmt.Errorf("Invalid or missing 'cache_dir' in data.")
	}
	appCtx.CacheDir = cacheDir

	// Set profile_id and auth_token
	profileID, ok := data["profile_id"].(string)
	if !ok || profileID == "" {
		return fmt.Errorf("Missing 'profile_id' in data.")
	}
	appCtx.ProfileID = profileID

	authToken, ok := data["auth_token"].(string)
	if !ok || authToken == "" {
		return fmt.Errorf("Missing 'auth_token' in data.")
	}
	appCtx.AuthToken = authToken

	// Set min and max cluster sizes
	if val, ok := data["min_cluster_size"].(float64); ok {
		appCtx.MinClusterSize = int(val)
	}
	if val, ok := data["max_cluster_size"].(float64); ok {
		appCtx.MaxClusterSize = int(val)
	}

	return nil
}

// parseStringMap parses a map[string]string from an interface{}.
func parseStringMap(raw interface{}) map[string]string {
	result := make(map[string]string)
	if rawMap, ok := raw.(map[string]interface{}); ok {
		for key, val := range rawMap {
			if str, ok := val.(string); ok {
				result[key] = str
			}
		}
	}
	return result
}

// aggregateFeatures aggregates features for all products in a cluster.
func aggregateFeatures(clusterID string, products []string, appCtx *AppContext) string {
	var aggregated []string

	for _, pid := range products {
		labels := appCtx.LabelsMapping[pid]
		if len(labels) > 0 {
			aggregated = append(aggregated, labels...)
		}

		title := appCtx.TitleMapping[pid]
		if title != "" {
			aggregated = append(aggregated, title)
		}

		description := appCtx.DescriptionMapping[pid]
		if description != "" {
			aggregated = append(aggregated, description)
		}
	}

	aggregatedText := strings.Join(aggregated, ", ")
	log.Printf("Aggregated text for Cluster %s: %s", clusterID, aggregatedText)
	return aggregatedText
}

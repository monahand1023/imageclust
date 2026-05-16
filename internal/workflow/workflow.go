package workflow

import (
	"context"
	"fmt"
	"imageclust/internal/clip"
	"imageclust/internal/clustering"
	"imageclust/internal/models"
	"imageclust/internal/ollama"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

var maxWorkers = runtime.NumCPU()

// ImageCluster orchestrates the end-to-end pipeline for a single request.
type ImageCluster struct {
	TempDir        string
	ClipModel      *clip.Model
	OllamaClient   *ollama.Client
	MinClusterSize int
	MaxClusterSize int
}

// NewImageCluster creates an ImageCluster. clipModel and ollamaClient are
// shared singletons loaded at startup — not created per request.
func NewImageCluster(minClusterSize, maxClusterSize int, tempDir string, clipModel *clip.Model, ollamaClient *ollama.Client) *ImageCluster {
	log.Printf("ImageCluster init: min=%d max=%d clusters, tempDir=%s", minClusterSize, maxClusterSize, tempDir)
	return &ImageCluster{
		TempDir:        tempDir,
		ClipModel:      clipModel,
		OllamaClient:   ollamaClient,
		MinClusterSize: minClusterSize,
		MaxClusterSize: maxClusterSize,
	}
}

type itemRecord struct {
	ID        string
	ImagePath string
}

// Run executes the full pipeline and returns a map of cluster key → ClusterDetails.
// ctx is propagated to Ollama calls so a cancelled request aborts in-flight LLM work.
func (ic *ImageCluster) Run(ctx context.Context, uploadedImages []models.UploadedImage) (map[string]models.ClusterDetails, error) {
	startTime := time.Now()
	log.Println("ImageCluster: starting run")

	imageDir := filepath.Join(ic.TempDir, "images")
	if err := os.MkdirAll(imageDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create image dir: %w", err)
	}

	items, err := ic.saveImages(uploadedImages, imageDir)
	if err != nil {
		return nil, err
	}

	embeddingsList, itemIDs, err := ic.generateEmbeddings(items)
	if err != nil {
		return nil, err
	}

	clusters, err := clustering.PerformClusteringWithConstraints(
		embeddingsList, itemIDs, ic.MinClusterSize, ic.MaxClusterSize,
	)
	if err != nil {
		return nil, fmt.Errorf("clustering failed: %w", err)
	}

	details, err := ic.buildClusterDetails(ctx, clusters, items, embeddingsList)
	if err != nil {
		return nil, err
	}

	log.Printf("ImageCluster: completed in %v", time.Since(startTime))
	return details, nil
}

// saveImages writes uploaded image bytes to disk and returns item records.
// Files are named img_0.ext, img_1.ext, ... to avoid collisions between
// uploads with identical or similarly-sanitized filenames.
func (ic *ImageCluster) saveImages(uploadedImages []models.UploadedImage, imageDir string) ([]itemRecord, error) {
	items := make([]itemRecord, 0, len(uploadedImages))
	for i, img := range uploadedImages {
		ext := strings.ToLower(filepath.Ext(img.Filename))
		if ext == "" {
			ext = ".jpg"
		}
		filename := fmt.Sprintf("img_%d%s", i, ext)
		imagePath := filepath.Join(imageDir, filename)
		if err := os.WriteFile(imagePath, img.Data, 0644); err != nil {
			return nil, fmt.Errorf("failed to save image %s: %w", img.Filename, err)
		}
		items = append(items, itemRecord{
			ID:        fmt.Sprintf("img_%d", i),
			ImagePath: imagePath,
		})
	}
	return items, nil
}

type embJob struct {
	index int
	item  itemRecord
}

type embResult struct {
	index     int
	itemID    string
	embedding []float32
	err       error
}

// generateEmbeddings runs CLIP inference in a bounded worker pool.
func (ic *ImageCluster) generateEmbeddings(items []itemRecord) ([][]float32, []string, error) {
	n := len(items)
	numWorkers := maxWorkers
	if n < numWorkers {
		numWorkers = n
	}

	jobs := make(chan embJob, n)
	results := make(chan embResult, n)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				emb, err := ic.ClipModel.Embed(job.item.ImagePath)
				results <- embResult{
					index:     job.index,
					itemID:    job.item.ID,
					embedding: emb,
					err:       err,
				}
			}
		}()
	}

	for i, item := range items {
		jobs <- embJob{index: i, item: item}
	}
	close(jobs)

	go func() { wg.Wait(); close(results) }()

	embeddingsList := make([][]float32, n)
	itemIDs := make([]string, n)
	var firstErr error
	for res := range results {
		if res.err != nil {
			if firstErr == nil {
				firstErr = res.err
			}
			continue
		}
		embeddingsList[res.index] = res.embedding
		itemIDs[res.index] = res.itemID
	}
	if firstErr != nil {
		return nil, nil, firstErr
	}

	log.Printf("ImageCluster: embedded %d images with %d workers", n, numWorkers)
	return embeddingsList, itemIDs, nil
}

type clusterTitleResult struct {
	key          string
	details      models.ClusterDetails
}

// buildClusterDetails generates titles via Ollama for each cluster in parallel.
// It picks up to 3 representative images (nearest to the cluster centroid).
// Ollama will queue requests it can't run concurrently; set OLLAMA_NUM_PARALLEL
// on the Ollama server to control how many inference slots it uses.
func (ic *ImageCluster) buildClusterDetails(
	ctx context.Context,
	clusters map[int][]string,
	items []itemRecord,
	embeddingsList [][]float32,
) (map[string]models.ClusterDetails, error) {
	itemByID := make(map[string]itemRecord, len(items))
	idxByID := make(map[string]int, len(items))
	for i, item := range items {
		itemByID[item.ID] = item
		idxByID[item.ID] = i
	}

	resultCh := make(chan clusterTitleResult, len(clusters))
	var wg sync.WaitGroup

	for clusterID, itemIDs := range clusters {
		wg.Add(1)
		go func(clusterID int, itemIDs []string) {
			defer wg.Done()
			key := fmt.Sprintf("Cluster-%d", clusterID)

			// Gather image paths and embeddings for this cluster.
			imagePaths := make([]string, 0, len(itemIDs))
			clusterEmbs := make([][]float32, 0, len(itemIDs))
			for _, id := range itemIDs {
				if item, ok := itemByID[id]; ok {
					imagePaths = append(imagePaths, item.ImagePath)
					if idx, ok2 := idxByID[id]; ok2 {
						clusterEmbs = append(clusterEmbs, embeddingsList[idx])
					}
				}
			}

			// Pick up to 3 images closest to the cluster centroid.
			representativeImagePaths := selectRepresentatives(imagePaths, clusterEmbs, 3)

			title, catchyPhrase, err := ic.OllamaClient.GenerateClusterTitle(ctx, representativeImagePaths, 3, 3)
			if err != nil {
				log.Printf("ImageCluster: title generation failed for %s: %v — using fallback", key, err)
				title = fmt.Sprintf("Cluster %d", clusterID)
				catchyPhrase = ""
			}

			// Store only the base filename for the API response.
			imageFilenames := make([]string, len(imagePaths))
			for i, p := range imagePaths {
				imageFilenames[i] = filepath.Base(p)
			}

			resultCh <- clusterTitleResult{
				key: key,
				details: models.ClusterDetails{
					Title:        title,
					CatchyPhrase: catchyPhrase,
					Images:       imageFilenames,
				},
			}
		}(clusterID, itemIDs)
	}

	go func() { wg.Wait(); close(resultCh) }()

	clusterDetails := make(map[string]models.ClusterDetails, len(clusters))
	for r := range resultCh {
		clusterDetails[r.key] = r.details
	}

	return clusterDetails, nil
}

// selectRepresentatives returns the paths of the k images whose embeddings are
// closest to the cluster centroid (highest cosine similarity, i.e. highest dot
// product since embeddings are already L2-normalized).
func selectRepresentatives(paths []string, embs [][]float32, k int) []string {
	if len(paths) == 0 {
		return nil
	}
	if len(paths) <= k {
		return paths
	}
	if len(embs) == 0 || len(embs[0]) == 0 {
		return paths[:k]
	}

	dim := len(embs[0])
	centroid := make([]float64, dim)
	for _, emb := range embs {
		for j, v := range emb {
			centroid[j] += float64(v)
		}
	}
	// L2-normalize the centroid.
	var norm float64
	for _, v := range centroid {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for j := range centroid {
			centroid[j] /= norm
		}
	}

	// Score each image by dot product with centroid.
	type scored struct {
		path  string
		score float64
	}
	scores := make([]scored, len(paths))
	for i, emb := range embs {
		var dot float64
		for j, v := range emb {
			dot += float64(v) * centroid[j]
		}
		scores[i] = scored{path: paths[i], score: dot}
	}

	// Partial sort: find the k highest scores.
	for i := 0; i < k && i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	result := make([]string, k)
	for i := 0; i < k; i++ {
		result[i] = scores[i].path
	}
	return result
}

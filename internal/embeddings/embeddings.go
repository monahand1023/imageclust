// Package embeddings generates image embeddings using the CLIP vision model.
package embeddings

import "imageclust/internal/clip"

// GetEmbedding returns the CLIP L2-normalized embedding for the image at imagePath.
func GetEmbedding(model *clip.Model, imagePath string) ([]float32, error) {
	return model.Embed(imagePath)
}

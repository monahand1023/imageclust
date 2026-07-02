// Package clustering/clustering.go
package clustering

import (
	"errors"
	"fmt"
	"log"
	"math"
)

// ErrDimensionMismatch is returned when vector dimensions don't match
var ErrDimensionMismatch = errors.New("vector dimensions do not match")

// Cluster represents a cluster of data points.
type Cluster struct {
	Indices  []int     // Indices of data points in the cluster
	Size     int       // Number of data points in the cluster
	Centroid []float32 // Centroid of the cluster
}

// NewCluster creates a new cluster with a single data point.
func NewCluster(index int, embedding []float32) Cluster {
	centroid := make([]float32, len(embedding))
	copy(centroid, embedding)
	return Cluster{
		Indices:  []int{index},
		Size:     1,
		Centroid: centroid,
	}
}

// MergeClusters merges two clusters into a new cluster.
func MergeClusters(a, b Cluster) Cluster {
	// New indices — copied into a fresh slice so neither input's backing
	// array is written to (append(a.Indices, ...) would alias it).
	indices := make([]int, 0, len(a.Indices)+len(b.Indices))
	indices = append(indices, a.Indices...)
	indices = append(indices, b.Indices...)

	// New size
	size := a.Size + b.Size

	// New centroid
	centroid := make([]float32, len(a.Centroid))
	for i := range centroid {
		centroid[i] = (float32(a.Size)*a.Centroid[i] + float32(b.Size)*b.Centroid[i]) / float32(size)
	}

	return Cluster{
		Indices:  indices,
		Size:     size,
		Centroid: centroid,
	}
}

// RemoveClusters removes clusters at indices i and j from the clusters slice.
// It assumes that i < j.
func RemoveClusters(clusters []Cluster, i, j int) []Cluster {
	if i > j {
		i, j = j, i
	}
	clusters = append(clusters[:j], clusters[j+1:]...)
	clusters = append(clusters[:i], clusters[i+1:]...)
	return clusters
}

// ComputeInitialDistanceMatrix computes the initial distance matrix between clusters.
// Returns an error if any distance calculation fails.
func ComputeInitialDistanceMatrix(clusters []Cluster) ([][]float32, error) {
	n := len(clusters)
	distanceMatrix := make([][]float32, n)
	for i := 0; i < n; i++ {
		distanceMatrix[i] = make([]float32, n)
		for j := 0; j < i; j++ {
			distance, err := WardDistance(clusters[i], clusters[j])
			if err != nil {
				return nil, fmt.Errorf("failed to compute distance between clusters %d and %d: %w", i, j, err)
			}
			distanceMatrix[i][j] = distance
			distanceMatrix[j][i] = distance
		}
	}
	return distanceMatrix, nil
}

// UpdateDistanceMatrix updates the distance matrix after merging clusters.
// Returns an error if any distance calculation fails.
func UpdateDistanceMatrix(distanceMatrix [][]float32, clusters []Cluster, newCluster Cluster, removedIdx1, removedIdx2 int) ([][]float32, error) {
	// Remove rows and columns corresponding to the removed clusters
	distanceMatrix = RemoveRowsAndColumns(distanceMatrix, removedIdx1, removedIdx2)

	// Add distances between new cluster and existing clusters
	n := len(clusters)
	newRow := make([]float32, n)
	for i := 0; i < n-1; i++ {
		distance, err := WardDistance(clusters[i], newCluster)
		if err != nil {
			return nil, fmt.Errorf("failed to compute distance to new cluster: %w", err)
		}
		newRow[i] = distance
	}
	newRow[n-1] = 0.0 // Distance to itself is zero

	// Append new row and column to the distance matrix
	for i := 0; i < n-1; i++ {
		distanceMatrix[i] = append(distanceMatrix[i], newRow[i])
	}
	distanceMatrix = append(distanceMatrix, newRow)

	return distanceMatrix, nil
}

// RemoveRowsAndColumns removes rows and columns at indices i and j from the distance matrix.
// It assumes that i < j.
func RemoveRowsAndColumns(matrix [][]float32, i, j int) [][]float32 {
	if i > j {
		i, j = j, i
	}

	// Remove columns
	for idx := range matrix {
		matrix[idx] = append(matrix[idx][:j], matrix[idx][j+1:]...)
		matrix[idx] = append(matrix[idx][:i], matrix[idx][i+1:]...)
	}

	// Remove rows
	matrix = append(matrix[:j], matrix[j+1:]...)
	matrix = append(matrix[:i], matrix[i+1:]...)

	return matrix
}

// FindClosestClusters finds the two clusters with the minimum distance.
func FindClosestClusters(distanceMatrix [][]float32) (int, int) {
	minDistance := float32(math.MaxFloat32)
	var idx1, idx2 = -1, -1
	n := len(distanceMatrix)
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			if distanceMatrix[i][j] < minDistance {
				minDistance = distanceMatrix[i][j]
				idx1 = i
				idx2 = j
			}
		}
	}
	return idx1, idx2
}

// WardDistance calculates the Ward's linkage distance between two clusters.
// Returns an error if centroid dimensions don't match.
func WardDistance(a, b Cluster) (float32, error) {
	diff := make([]float32, len(a.Centroid))
	for i := range diff {
		diff[i] = a.Centroid[i] - b.Centroid[i]
	}
	distanceSquared, err := DotFloat32(diff, diff)
	if err != nil {
		return 0, err
	}
	numerator := float32(a.Size * b.Size)
	denominator := float32(a.Size + b.Size)
	return (numerator / denominator) * distanceSquared, nil
}

// DotFloat32 computes the dot product of two float32 slices.
// Returns an error if slices have different lengths.
func DotFloat32(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("%w: got lengths %d and %d", ErrDimensionMismatch, len(a), len(b))
	}
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum, nil
}

// CalculateOptimalClusters calculates the optimal number of clusters based on desired cluster size constraints.
// It uses a simple heuristic to balance between minimum and maximum cluster sizes.
// Parameters:
// - totalItems: Total number of data points.
// - minSize: Minimum number of items per cluster.
// - maxSize: Maximum number of items per cluster.
// Returns:
// - Optimal number of clusters.
// - An error if constraints are impossible to satisfy.
func CalculateOptimalClusters(totalItems, minSize, maxSize int) (int, error) {
	if totalItems < minSize {
		return 0, fmt.Errorf("total items (%d) less than minimum cluster size (%d)", totalItems, minSize)
	}

	nClustersMin := int(math.Ceil(float64(totalItems) / float64(maxSize)))
	nClustersMax := int(math.Floor(float64(totalItems) / float64(minSize)))
	if nClustersMin > nClustersMax {
		return 0, fmt.Errorf("cannot satisfy cluster size constraints with total items (%d), minSize (%d), and maxSize (%d)", totalItems, minSize, maxSize)
	}

	// Heuristic: choose the number of clusters that minimizes the difference between nClustersMin and nClustersMax
	nClusters := nClustersMin
	if nClustersMin < nClustersMax {
		nClusters = (nClustersMin + nClustersMax) / 2
	}

	return nClusters, nil
}

// PerformClusteringWithConstraints performs hierarchical clustering with size constraints.
// It ensures that each cluster has between minSize and maxSize items.
// Parameters:
// - embeddings: Slice of embedding vectors.
// - productReferenceIDs: Slice of product reference IDs corresponding to embeddings.
// - minSize: Minimum number of items per cluster.
// - maxSize: Maximum number of items per cluster.
// Returns:
//   - A map where keys are cluster IDs (starting from 0) and values are slices of product reference IDs.
//   - A slice of reference IDs that could not be placed in a valid cluster
//     (leftovers smaller than minSize) — never silently dropped.
//   - An error if clustering fails.
func PerformClusteringWithConstraints(embeddings [][]float32, productReferenceIDs []string, minSize, maxSize int) (map[int][]string, []string, error) {
	totalItems := len(embeddings)
	log.Printf("Total items for clustering: %d", totalItems)

	// Calculate the optimal number of clusters
	nClusters, err := CalculateOptimalClusters(totalItems, minSize, maxSize)
	if err != nil {
		return nil, nil, fmt.Errorf("clustering constraint error: %w", err)
	}
	log.Printf("Optimal number of clusters calculated: %d", nClusters)

	// Initialize clusters: each embedding starts as its own cluster
	clusters := make([]Cluster, totalItems)
	for i := 0; i < totalItems; i++ {
		clusters[i] = NewCluster(i, embeddings[i])
	}

	// Compute initial distance matrix
	distanceMatrix, err := ComputeInitialDistanceMatrix(clusters)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compute initial distance matrix: %w", err)
	}

	// Hierarchical clustering using Ward's method with size constraints
	clusters, err = mergeUntil(clusters, distanceMatrix, nClusters, maxSize)
	if err != nil {
		return nil, nil, err
	}

	// Convert clusters to map with product reference IDs. mergeUntil never
	// merges past maxSize, so no cluster can exceed it here. Clusters below
	// minSize are reported as unclustered rather than silently dropped.
	clusterMap := make(map[int][]string)
	var unclustered []string
	clusterID := 0
	for _, cluster := range clusters {
		refs := make([]string, len(cluster.Indices))
		for i, idx := range cluster.Indices {
			refs[i] = productReferenceIDs[idx]
		}
		if cluster.Size < minSize {
			unclustered = append(unclustered, refs...)
			continue
		}
		clusterMap[clusterID] = refs
		clusterID++
	}

	log.Printf("Clustering successful. Formed %d valid clusters, %d unclustered items.", len(clusterMap), len(unclustered))
	return clusterMap, unclustered, nil
}

// mergeUntil repeatedly merges the two closest clusters until targetN clusters
// remain, never letting a merge exceed maxSize. Pairs whose merge would exceed
// maxSize are marked non-mergeable; the loop stops early when nothing mergeable
// is left. It mutates and returns both the clusters slice and distance matrix.
func mergeUntil(clusters []Cluster, distanceMatrix [][]float32, targetN, maxSize int) ([]Cluster, error) {
	for len(clusters) > targetN {
		i, j := FindClosestClusters(distanceMatrix)
		if i == -1 || j == -1 {
			break
		}

		if clusters[i].Size+clusters[j].Size > maxSize {
			// Mark this pair as non-mergeable by setting their distance to infinity
			distanceMatrix[i][j] = math.MaxFloat32
			distanceMatrix[j][i] = math.MaxFloat32
			continue
		}

		newCluster := MergeClusters(clusters[i], clusters[j])
		clusters = RemoveClusters(clusters, i, j)
		clusters = append(clusters, newCluster)

		var err error
		distanceMatrix, err = UpdateDistanceMatrix(distanceMatrix, clusters, newCluster, i, j)
		if err != nil {
			return nil, fmt.Errorf("failed to update distance matrix: %w", err)
		}
	}
	return clusters, nil
}

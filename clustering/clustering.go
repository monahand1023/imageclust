// clustering/clustering.go
package clustering

import (
	"fmt"
	"log"
	"math"
)

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
	// New indices
	indices := append(a.Indices, b.Indices...)

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
func ComputeInitialDistanceMatrix(clusters []Cluster) [][]float32 {
	n := len(clusters)
	distanceMatrix := make([][]float32, n)
	for i := 0; i < n; i++ {
		distanceMatrix[i] = make([]float32, n)
		for j := 0; j < i; j++ {
			distance := WardDistance(clusters[i], clusters[j])
			distanceMatrix[i][j] = distance
			distanceMatrix[j][i] = distance
		}
	}
	return distanceMatrix
}

// UpdateDistanceMatrix updates the distance matrix after merging clusters.
func UpdateDistanceMatrix(distanceMatrix [][]float32, clusters []Cluster, newCluster Cluster, removedIdx1, removedIdx2 int) [][]float32 {
	// Remove rows and columns corresponding to the removed clusters
	distanceMatrix = RemoveRowsAndColumns(distanceMatrix, removedIdx1, removedIdx2)

	// Add distances between new cluster and existing clusters
	n := len(clusters)
	newRow := make([]float32, n)
	for i := 0; i < n-1; i++ {
		distance := WardDistance(clusters[i], newCluster)
		newRow[i] = distance
	}
	newRow[n-1] = 0.0 // Distance to itself is zero

	// Append new row and column to the distance matrix
	for i := 0; i < n-1; i++ {
		distanceMatrix[i] = append(distanceMatrix[i], newRow[i])
	}
	distanceMatrix = append(distanceMatrix, newRow)

	return distanceMatrix
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
	var idx1, idx2 int = -1, -1
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
func WardDistance(a, b Cluster) float32 {
	diff := make([]float32, len(a.Centroid))
	for i := range diff {
		diff[i] = a.Centroid[i] - b.Centroid[i]
	}
	distanceSquared := DotFloat32(diff, diff)
	numerator := float32(a.Size * b.Size)
	denominator := float32(a.Size + b.Size)
	return (numerator / denominator) * distanceSquared
}

// DotFloat32 computes the dot product of two float32 slices
func DotFloat32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("DotFloat32: slices have different lengths")
	}
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
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
// - A map where keys are cluster IDs (starting from 0) and values are slices of product reference IDs.
// - A boolean indicating whether clustering was successful.
func PerformClusteringWithConstraints(embeddings [][]float32, productReferenceIDs []string, minSize, maxSize int) (map[int][]string, bool) {
	totalItems := len(embeddings)
	log.Printf("Total items for clustering: %d", totalItems)

	// Calculate the optimal number of clusters
	nClusters, err := CalculateOptimalClusters(totalItems, minSize, maxSize)
	if err != nil {
		log.Printf("Clustering constraint error: %v", err)
		return nil, false
	}
	log.Printf("Optimal number of clusters calculated: %d", nClusters)

	// Initialize clusters: each embedding starts as its own cluster
	clusters := make([]Cluster, totalItems)
	for i := 0; i < totalItems; i++ {
		clusters[i] = NewCluster(i, embeddings[i])
	}

	// Compute initial distance matrix
	distanceMatrix := ComputeInitialDistanceMatrix(clusters)

	// Hierarchical clustering using Ward's method
	for len(clusters) > nClusters {
		i, j := FindClosestClusters(distanceMatrix)
		if i == -1 || j == -1 {
			log.Println("No more clusters to merge.")
			break
		}

		// Merge clusters[i] and clusters[j]
		newCluster := MergeClusters(clusters[i], clusters[j])

		// Remove old clusters and update the clusters slice
		clusters = RemoveClusters(clusters, i, j)
		clusters = append(clusters, newCluster)

		// Recompute distances
		distanceMatrix = UpdateDistanceMatrix(distanceMatrix, clusters, newCluster, i, j)
		log.Printf("Merged clusters %d and %d into new cluster with size %d", i, j, newCluster.Size)
	}

	// After clustering, enforce size constraints
	// This step ensures that clusters adhere to minSize and maxSize constraints
	clusterMap := make(map[int][]string)
	clusterID := 0
	for _, cluster := range clusters {
		if cluster.Size < minSize || cluster.Size > maxSize {
			log.Printf("Cluster %d size %d violates constraints (min: %d, max: %d)", clusterID, cluster.Size, minSize, maxSize)
			return nil, false
		}

		refs := make([]string, len(cluster.Indices))
		for i, idx := range cluster.Indices {
			refs[i] = productReferenceIDs[idx]
		}
		clusterMap[clusterID] = refs
		clusterID++
	}

	log.Printf("Clustering successful. Formed %d clusters.", len(clusterMap))
	return clusterMap, true
}

// ComputeDistanceMatrix computes the distance matrix for the current set of clusters.
// Deprecated: Use ComputeInitialDistanceMatrix instead.
func ComputeDistanceMatrix(clusters []Cluster) [][]float32 {
	return ComputeInitialDistanceMatrix(clusters)
}

// AddCluster adds a new cluster to the distance matrix.
// Deprecated: Use UpdateDistanceMatrix instead.
func AddCluster(distanceMatrix [][]float32, clusters []Cluster, newCluster Cluster) [][]float32 {
	// Not implemented as UpdateDistanceMatrix handles this.
	return distanceMatrix
}

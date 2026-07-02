package clustering

import (
	"errors"
	"testing"
)

func TestDotFloat32_Success(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{4.0, 5.0, 6.0}

	result, err := DotFloat32(a, b)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	expected := float32(32.0) // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

func TestDotFloat32_DimensionMismatch(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{4.0, 5.0}

	_, err := DotFloat32(a, b)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}

	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestDotFloat32_EmptySlices(t *testing.T) {
	a := []float32{}
	b := []float32{}

	result, err := DotFloat32(a, b)
	if err != nil {
		t.Fatalf("expected no error for empty slices, got: %v", err)
	}

	if result != 0 {
		t.Errorf("expected 0 for empty slices, got %f", result)
	}
}

func TestWardDistance_Success(t *testing.T) {
	a := Cluster{
		Indices:  []int{0},
		Size:     1,
		Centroid: []float32{1.0, 2.0, 3.0},
	}
	b := Cluster{
		Indices:  []int{1},
		Size:     1,
		Centroid: []float32{4.0, 5.0, 6.0},
	}

	distance, err := WardDistance(a, b)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	// diff = [-3, -3, -3], dot = 27, ward = (1*1)/(1+1) * 27 = 0.5 * 27 = 13.5
	expected := float32(13.5)
	if distance != expected {
		t.Errorf("expected %f, got %f", expected, distance)
	}
}

func TestWardDistance_DifferentSizes(t *testing.T) {
	a := Cluster{
		Indices:  []int{0, 1},
		Size:     2,
		Centroid: []float32{1.0, 0.0},
	}
	b := Cluster{
		Indices:  []int{2, 3, 4},
		Size:     3,
		Centroid: []float32{0.0, 1.0},
	}

	distance, err := WardDistance(a, b)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	// diff = [1, -1], dot = 2, ward = (2*3)/(2+3) * 2 = 6/5 * 2 = 2.4
	expected := float32(2.4)
	if distance != expected {
		t.Errorf("expected %f, got %f", expected, distance)
	}
}

func TestComputeInitialDistanceMatrix_Success(t *testing.T) {
	clusters := []Cluster{
		{Indices: []int{0}, Size: 1, Centroid: []float32{0.0, 0.0}},
		{Indices: []int{1}, Size: 1, Centroid: []float32{1.0, 0.0}},
		{Indices: []int{2}, Size: 1, Centroid: []float32{0.0, 1.0}},
	}

	matrix, err := ComputeInitialDistanceMatrix(clusters)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	if len(matrix) != 3 {
		t.Errorf("expected matrix size 3, got %d", len(matrix))
	}

	// Check symmetry
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix); j++ {
			if matrix[i][j] != matrix[j][i] {
				t.Errorf("matrix not symmetric at [%d][%d]: %f != %f", i, j, matrix[i][j], matrix[j][i])
			}
		}
	}

	// Diagonal should be 0
	for i := 0; i < len(matrix); i++ {
		if matrix[i][i] != 0 {
			t.Errorf("expected diagonal to be 0, got %f at [%d][%d]", matrix[i][i], i, i)
		}
	}
}

func TestNewCluster(t *testing.T) {
	embedding := []float32{1.0, 2.0, 3.0}
	cluster := NewCluster(5, embedding)

	if len(cluster.Indices) != 1 || cluster.Indices[0] != 5 {
		t.Errorf("expected indices [5], got %v", cluster.Indices)
	}

	if cluster.Size != 1 {
		t.Errorf("expected size 1, got %d", cluster.Size)
	}

	// Verify centroid is a copy, not same slice
	embedding[0] = 99.0
	if cluster.Centroid[0] == 99.0 {
		t.Error("centroid should be a copy, not the original slice")
	}
}

func TestMergeClusters(t *testing.T) {
	a := Cluster{
		Indices:  []int{0, 1},
		Size:     2,
		Centroid: []float32{2.0, 4.0},
	}
	b := Cluster{
		Indices:  []int{2},
		Size:     1,
		Centroid: []float32{5.0, 7.0},
	}

	merged := MergeClusters(a, b)

	if merged.Size != 3 {
		t.Errorf("expected size 3, got %d", merged.Size)
	}

	if len(merged.Indices) != 3 {
		t.Errorf("expected 3 indices, got %d", len(merged.Indices))
	}

	// Centroid should be weighted average: (2*[2,4] + 1*[5,7]) / 3 = [9/3, 15/3] = [3, 5]
	expectedCentroid := []float32{3.0, 5.0}
	for i, v := range merged.Centroid {
		if v != expectedCentroid[i] {
			t.Errorf("expected centroid[%d] = %f, got %f", i, expectedCentroid[i], v)
		}
	}
}

func TestMergeClusters_DoesNotAliasInputIndices(t *testing.T) {
	// a.Indices has spare capacity; a naive append(a.Indices, b.Indices...)
	// would write into that backing array, corrupting earlier merge results.
	shared := make([]int, 2, 4)
	shared[0], shared[1] = 0, 1
	a := Cluster{Indices: shared, Size: 2, Centroid: []float32{0, 0}}
	b := Cluster{Indices: []int{2}, Size: 1, Centroid: []float32{1, 1}}
	c := Cluster{Indices: []int{3}, Size: 1, Centroid: []float32{2, 2}}

	ab := MergeClusters(a, b)
	_ = MergeClusters(a, c)

	want := []int{0, 1, 2}
	for i, v := range want {
		if ab.Indices[i] != v {
			t.Fatalf("ab.Indices = %v, want %v — merge aliased the input's backing array", ab.Indices, want)
		}
	}
}

func TestCalculateOptimalClusters(t *testing.T) {
	tests := []struct {
		name        string
		totalItems  int
		minSize     int
		maxSize     int
		expectError bool
	}{
		{"valid constraints", 10, 2, 5, false},
		{"exact fit", 6, 2, 3, false},
		{"too few items", 2, 3, 5, true},
		{"impossible constraints", 10, 5, 4, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := CalculateOptimalClusters(tt.totalItems, tt.minSize, tt.maxSize)
			if tt.expectError && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("expected no error, got: %v", err)
			}
		})
	}
}

func TestPerformClusteringWithConstraints(t *testing.T) {
	// Create simple 2D embeddings
	embeddings := [][]float32{
		{0.0, 0.0},
		{0.1, 0.1},
		{0.2, 0.2},
		{5.0, 5.0},
		{5.1, 5.1},
		{5.2, 5.2},
	}
	ids := []string{"a", "b", "c", "d", "e", "f"}

	result, unclustered, err := PerformClusteringWithConstraints(embeddings, ids, 2, 4)

	if err != nil {
		t.Fatalf("clustering should have succeeded: %v", err)
	}

	if len(result) == 0 {
		t.Error("expected at least one cluster")
	}

	// Every input ID must appear exactly once, either clustered or unclustered.
	seen := make(map[string]int)
	for _, items := range result {
		if len(items) < 2 || len(items) > 4 {
			t.Errorf("cluster size %d outside constraints [2, 4]", len(items))
		}
		for _, id := range items {
			seen[id]++
		}
	}
	for _, id := range unclustered {
		seen[id]++
	}
	for _, id := range ids {
		if seen[id] != 1 {
			t.Errorf("id %q appeared %d times across clusters+unclustered, want exactly 1", id, seen[id])
		}
	}
}

func TestPerformClusteringWithConstraints_OutlierReturnedAsUnclustered(t *testing.T) {
	// Three tight points plus one distant outlier. With minSize=2 the outlier
	// can't form a valid cluster — it must be reported, not silently dropped.
	embeddings := [][]float32{
		{0.0, 0.0},
		{0.1, 0.1},
		{0.2, 0.2},
		{50.0, 50.0},
	}
	ids := []string{"a", "b", "c", "outlier"}

	result, unclustered, err := PerformClusteringWithConstraints(embeddings, ids, 2, 3)
	if err != nil {
		t.Fatalf("clustering should have succeeded: %v", err)
	}

	if len(unclustered) != 1 || unclustered[0] != "outlier" {
		t.Errorf("unclustered = %v, want [outlier]", unclustered)
	}
	total := 0
	for _, items := range result {
		total += len(items)
	}
	if total != 3 {
		t.Errorf("clustered items = %d, want 3", total)
	}
}

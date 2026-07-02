package clustering

import (
	"math/rand"
	"testing"
)

func randEmbeddings(n, dim int) [][]float32 {
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = make([]float32, dim)
		var norm float32
		for j := range embs[i] {
			v := float32(rand.NormFloat64())
			embs[i][j] = v
			norm += v * v
		}
		if norm > 0 {
			norm = float32(1.0 / float64(norm))
			for j := range embs[i] {
				embs[i][j] *= norm
			}
		}
	}
	return embs
}

func randIDs(n int) []string {
	ids := make([]string, n)
	for i := range ids {
		ids[i] = "img_" + string(rune('0'+i%10))
	}
	return ids
}

func benchmarkClustering(b *testing.B, n int) {
	embs := randEmbeddings(n, 768)
	ids := randIDs(n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := PerformClusteringWithConstraints(embs, ids, 3, 8)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkClustering10(b *testing.B)  { benchmarkClustering(b, 10) }
func BenchmarkClustering20(b *testing.B)  { benchmarkClustering(b, 20) }
func BenchmarkClustering50(b *testing.B)  { benchmarkClustering(b, 50) }
func BenchmarkClustering100(b *testing.B) { benchmarkClustering(b, 100) }
func BenchmarkClustering200(b *testing.B) { benchmarkClustering(b, 200) }

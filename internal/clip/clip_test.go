package clip

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"testing"
)

// makePNG creates a solid-colour PNG in a temp dir and returns its path.
func makePNG(t *testing.T, w, h int, c color.NRGBA) string {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode: %v", err)
	}
	f, err := os.CreateTemp(t.TempDir(), "*.png")
	if err != nil {
		t.Fatalf("CreateTemp: %v", err)
	}
	defer f.Close()
	f.Write(buf.Bytes())
	return f.Name()
}

// --- l2Normalize -----------------------------------------------------------

func TestL2Normalize(t *testing.T) {
	const eps = 1e-6
	tests := []struct {
		name  string
		input []float32
		want  []float32
	}{
		{
			name:  "3-4-5 triple",
			input: []float32{3, 4},
			want:  []float32{0.6, 0.8},
		},
		{
			name:  "zero vector unchanged",
			input: []float32{0, 0, 0},
			want:  []float32{0, 0, 0},
		},
		{
			name:  "already unit vector",
			input: []float32{1, 0, 0},
			want:  []float32{1, 0, 0},
		},
		{
			name:  "negative components",
			input: []float32{-3, 4},
			want:  []float32{-0.6, 0.8},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := make([]float32, len(tc.input))
			copy(v, tc.input)
			l2Normalize(v)
			for i, want := range tc.want {
				if math.Abs(float64(v[i])-float64(want)) > eps {
					t.Errorf("[%d] got %f, want %f", i, v[i], want)
				}
			}
		})
	}
}

func TestL2Normalize_ResultIsUnit(t *testing.T) {
	v := []float32{1, 2, 3, 4}
	l2Normalize(v)
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("norm² = %f, want 1.0", sum)
	}
}

// --- preprocessImage -------------------------------------------------------

func TestPreprocessImage_OutputShape(t *testing.T) {
	path := makePNG(t, 64, 48, color.NRGBA{R: 128, G: 64, B: 32, A: 255})
	data, err := preprocessImage(path)
	if err != nil {
		t.Fatalf("preprocessImage: %v", err)
	}
	want := 3 * inputSize * inputSize
	if len(data) != want {
		t.Errorf("len(data) = %d, want %d", len(data), want)
	}
}

func TestPreprocessImage_WhitePixelNormalization(t *testing.T) {
	// Solid white image: all pixels (255,255,255).
	// CLIP normalization: (1.0 - mean) / std per channel.
	path := makePNG(t, 32, 32, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	data, err := preprocessImage(path)
	if err != nil {
		t.Fatalf("preprocessImage: %v", err)
	}

	wantR := (1.0 - float64(meanR)) / float64(stdR)
	wantG := (1.0 - float64(meanG)) / float64(stdG)
	wantB := (1.0 - float64(meanB)) / float64(stdB)

	stride := inputSize * inputSize
	center := (inputSize/2)*inputSize + inputSize/2

	const tol = 1e-4
	if d := math.Abs(float64(data[center]) - wantR); d > tol {
		t.Errorf("R: got %f, want %f (diff %f)", data[center], wantR, d)
	}
	if d := math.Abs(float64(data[stride+center]) - wantG); d > tol {
		t.Errorf("G: got %f, want %f (diff %f)", data[stride+center], wantG, d)
	}
	if d := math.Abs(float64(data[2*stride+center]) - wantB); d > tol {
		t.Errorf("B: got %f, want %f (diff %f)", data[2*stride+center], wantB, d)
	}
}

func TestPreprocessImage_BlackPixelNormalization(t *testing.T) {
	// Solid black image: all pixels (0,0,0).
	// CLIP normalization: (0.0 - mean) / std per channel → negative values.
	path := makePNG(t, 32, 32, color.NRGBA{R: 0, G: 0, B: 0, A: 255})
	data, err := preprocessImage(path)
	if err != nil {
		t.Fatalf("preprocessImage: %v", err)
	}

	wantR := (0.0 - float64(meanR)) / float64(stdR)
	wantG := (0.0 - float64(meanG)) / float64(stdG)
	wantB := (0.0 - float64(meanB)) / float64(stdB)

	stride := inputSize * inputSize
	center := (inputSize/2)*inputSize + inputSize/2

	const tol = 1e-4
	if d := math.Abs(float64(data[center]) - wantR); d > tol {
		t.Errorf("R: got %f, want %f", data[center], wantR)
	}
	if d := math.Abs(float64(data[stride+center]) - wantG); d > tol {
		t.Errorf("G: got %f, want %f", data[stride+center], wantG)
	}
	if d := math.Abs(float64(data[2*stride+center]) - wantB); d > tol {
		t.Errorf("B: got %f, want %f", data[2*stride+center], wantB)
	}
}

func TestPreprocessImage_NonexistentFile(t *testing.T) {
	_, err := preprocessImage("/nonexistent/path/image.jpg")
	if err == nil {
		t.Error("expected error for nonexistent file, got nil")
	}
}

func TestPreprocessImage_InvalidData(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "*.jpg")
	if err != nil {
		t.Fatal(err)
	}
	f.Write([]byte("this is not image data"))
	f.Close()
	_, err = preprocessImage(f.Name())
	if err == nil {
		t.Error("expected error for invalid image data, got nil")
	}
}

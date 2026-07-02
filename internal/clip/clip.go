// Package clip provides CLIP ViT-L/14 image embeddings via ONNX Runtime.
package clip

import (
	"fmt"
	"image"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
	xdraw "golang.org/x/image/draw"
)

const (
	inputSize = 224

	// Xenova/clip-vit-large-patch14 ONNX model I/O names and output shape.
	// GetInputOutputInfo fails for large (>1GB) models so these are hardcoded.
	outputTensorName = "image_embeds"
	embeddingDim     = 768

	// CLIP ViT preprocessing normalization (openai/clip)
	meanR = float32(0.48145466)
	meanG = float32(0.4578275)
	meanB = float32(0.40821073)
	stdR  = float32(0.26862954)
	stdG  = float32(0.26130258)
	stdB  = float32(0.27577711)
)

// Embedder abstracts CLIP embedding to allow testing without ONNX Runtime.
type Embedder interface {
	Embed(imagePath string) ([]float32, error)
	Close()
}

var initOnce sync.Once

// InitONNXRuntime sets the shared library path and initializes the ONNX Runtime
// environment. libPath may be empty to use default discovery.
func InitONNXRuntime(libPath string) error {
	var initErr error
	initOnce.Do(func() {
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

// Model wraps an ONNX Runtime AdvancedSession for CLIP vision inference.
// Pre-allocated input/output tensors are reused across calls; concurrent
// calls are serialized internally via a mutex.
type Model struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputData    []float32 // backing slice shared with inputTensor
	outputData   []float32 // backing slice shared with outputTensor
	mu           sync.Mutex
	EmbeddingDim int
}

// LoadModel opens the ONNX model at modelPath and creates the inference session.
// InitONNXRuntime must be called before this.
func LoadModel(modelPath string) (*Model, error) {
	// Pre-allocate backing slices; ort.NewTensor shares this memory with C.
	inputData := make([]float32, 3*inputSize*inputSize)
	outputData := make([]float32, embeddingDim)

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, inputSize, inputSize), inputData)
	if err != nil {
		return nil, fmt.Errorf("clip: failed to create input tensor: %w", err)
	}

	outputTensor, err := ort.NewTensor(ort.NewShape(1, embeddingDim), outputData)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("clip: failed to create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"pixel_values"},
		[]string{outputTensorName},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("clip: failed to create ONNX session: %w", err)
	}

	return &Model{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		inputData:    inputData,
		outputData:   outputData,
		EmbeddingDim: embeddingDim,
	}, nil
}

// Embed returns the L2-normalized CLIP embedding for the image at imagePath.
func (m *Model) Embed(imagePath string) ([]float32, error) {
	preprocessed, err := preprocessImage(imagePath)
	if err != nil {
		return nil, err
	}

	m.mu.Lock()
	copy(m.inputData, preprocessed) // update input tensor backing slice in-place
	err = m.session.Run()
	embedding := make([]float32, len(m.outputData))
	copy(embedding, m.outputData) // copy before releasing lock
	m.mu.Unlock()

	if err != nil {
		return nil, fmt.Errorf("clip: inference failed for %s: %w", imagePath, err)
	}

	l2Normalize(embedding)
	return embedding, nil
}

// Close releases the ONNX session and associated tensor resources.
func (m *Model) Close() {
	m.session.Destroy()
	m.inputTensor.Destroy()
	m.outputTensor.Destroy()
}

// preprocessImage decodes, resizes, and converts an image to the NCHW float32
// layout expected by CLIP ViT-L/14 (3 × 224 × 224), with CLIP normalization.
func preprocessImage(imagePath string) ([]float32, error) {
	f, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("clip: open %s: %w", imagePath, err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("clip: decode %s: %w", imagePath, err)
	}

	// Standard CLIP preprocessing resizes the shortest side to 224 and
	// center-crops. Cropping the central square first and scaling it to
	// 224×224 is equivalent (up to resampling) and avoids aspect distortion.
	b := img.Bounds()
	side := b.Dx()
	if b.Dy() < side {
		side = b.Dy()
	}
	srcRect := image.Rect(
		b.Min.X+(b.Dx()-side)/2,
		b.Min.Y+(b.Dy()-side)/2,
		b.Min.X+(b.Dx()-side)/2+side,
		b.Min.Y+(b.Dy()-side)/2+side,
	)

	// Scale using CatmullRom (Lanczos-quality bicubic); result is NRGBA.
	resized := image.NewNRGBA(image.Rect(0, 0, inputSize, inputSize))
	xdraw.CatmullRom.Scale(resized, resized.Bounds(), img, srcRect, draw.Src, nil)

	stride := inputSize * inputSize
	data := make([]float32, 3*stride)

	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			base := resized.PixOffset(x, y)
			rf := float32(resized.Pix[base]) / 255.0
			gf := float32(resized.Pix[base+1]) / 255.0
			bf := float32(resized.Pix[base+2]) / 255.0

			pos := y*inputSize + x
			data[pos] = (rf - meanR) / stdR
			data[stride+pos] = (gf - meanG) / stdG
			data[2*stride+pos] = (bf - meanB) / stdB
		}
	}

	return data, nil
}

func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if norm := float32(math.Sqrt(sum)); norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

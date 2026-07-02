// Package ollama calls the Ollama REST API for vision-capable cluster title generation.
package ollama

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	defaultModel     = "llama3.2-vision:11b"
	defaultHost      = "http://localhost:11434"
	defaultMaxImages = 3
	defaultRetries   = 3
	initialBackoff   = 2 * time.Second
	maxBackoff       = 30 * time.Second
	backoffFactor    = 2.0
	jitterFactor     = 0.3
)

// TitleGenerator abstracts Ollama title generation to allow testing.
type TitleGenerator interface {
	GenerateClusterTitle(ctx context.Context, imagePaths []string) (string, string, error)
}

// Client calls the Ollama HTTP API.
type Client struct {
	baseURL    string
	model      string
	httpClient *http.Client
	maxImages  int // images sent per title request
	retries    int // attempts before giving up
}

// NewClient creates a Client. host defaults to OLLAMA_HOST env or http://localhost:11434.
// model defaults to OLLAMA_MODEL env or llama3.2-vision:11b.
func NewClient(host, model string) *Client {
	if host == "" {
		host = os.Getenv("OLLAMA_HOST")
		if host == "" {
			host = defaultHost
		}
	}
	if model == "" {
		model = os.Getenv("OLLAMA_MODEL")
		if model == "" {
			model = defaultModel
		}
	}
	return &Client{
		baseURL:    strings.TrimRight(host, "/"),
		model:      model,
		httpClient: &http.Client{Timeout: 5 * time.Minute},
		maxImages:  defaultMaxImages,
		retries:    defaultRetries,
	}
}

// generateRequest mirrors the Ollama REST API generate body.
type generateRequest struct {
	Model  string   `json:"model"`
	Prompt string   `json:"prompt"`
	Images []string `json:"images,omitempty"` // base64-encoded image bytes
	Stream bool     `json:"stream"`
}

// generateResponse mirrors the non-streaming Ollama generate response.
type generateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// titleResponse is the JSON we ask the vision model to return.
type titleResponse struct {
	Title        string `json:"title"`
	CatchyPhrase string `json:"catchy_phrase"`
}

// Ping checks whether the Ollama server is reachable by calling /api/tags with
// a short timeout. It returns nil if the server responds with HTTP 200.
func (c *Client) Ping(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("ping: build request: %w", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("ping: %w", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ping: unexpected status %d", resp.StatusCode)
	}
	return nil
}

// GenerateClusterTitle sends up to Client.maxImages representative images to
// the vision model and returns a title (≤25 chars) and catchy phrase (≤100
// chars). On failure it returns zero values and an error — the caller owns
// any fallback naming.
func (c *Client) GenerateClusterTitle(ctx context.Context, imagePaths []string) (string, string, error) {
	if len(imagePaths) > c.maxImages {
		imagePaths = imagePaths[:c.maxImages]
	}

	images := make([]string, 0, len(imagePaths))
	for _, p := range imagePaths {
		b, err := os.ReadFile(p)
		if err != nil {
			log.Printf("ollama: skipping unreadable image %s: %v", p, err)
			continue
		}
		images = append(images, base64.StdEncoding.EncodeToString(b))
	}

	if len(images) == 0 {
		return "", "", fmt.Errorf("no readable images among %d paths", len(imagePaths))
	}

	prompt := `Analyze this cluster of related images. Return ONLY a JSON object — no markdown, no explanation:
{"title": "short title here", "catchy_phrase": "catchy phrase here"}
Rules: title max 25 characters, catchy_phrase max 100 characters.`

	for attempt := 0; attempt < c.retries; attempt++ {
		if attempt > 0 {
			backoff := calculateBackoff(attempt - 1)
			log.Printf("ollama: attempt %d/%d, waiting %v", attempt+1, c.retries, backoff)
			select {
			case <-ctx.Done():
				return "", "", ctx.Err()
			case <-time.After(backoff):
			}
		}

		responseText, err := c.generate(ctx, prompt, images)
		if err != nil {
			log.Printf("ollama: generate error (attempt %d): %v", attempt+1, err)
			continue
		}

		jsonStr := extractJSON(responseText)
		var result titleResponse
		if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
			log.Printf("ollama: JSON parse error (attempt %d): %v — raw: %.200s", attempt+1, err, responseText)
			continue
		}

		result.Title = truncate(strings.TrimSpace(result.Title), 25)
		result.CatchyPhrase = truncate(strings.TrimSpace(result.CatchyPhrase), 100)

		if result.Title == "" {
			log.Printf("ollama: empty title in response (attempt %d)", attempt+1)
			continue
		}

		return result.Title, result.CatchyPhrase, nil
	}

	return "", "", fmt.Errorf("exhausted %d retries", c.retries)
}

func (c *Client) generate(ctx context.Context, prompt string, images []string) (string, error) {
	body, err := json.Marshal(generateRequest{
		Model:  c.model,
		Prompt: prompt,
		Images: images,
		Stream: false,
	})
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTP POST: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama returned HTTP %d", resp.StatusCode)
	}

	var genResp generateResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	return genResp.Response, nil
}

func extractJSON(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start == -1 || end == -1 || end < start {
		return s
	}
	return s[start : end+1]
}

func truncate(s string, max int) string {
	runes := []rune(s)
	if len(runes) > max {
		return string(runes[:max])
	}
	return s
}

func calculateBackoff(attempt int) time.Duration {
	backoff := float64(initialBackoff) * math.Pow(backoffFactor, float64(attempt))
	if backoff > float64(maxBackoff) {
		backoff = float64(maxBackoff)
	}
	return time.Duration(backoff + backoff*jitterFactor*rand.Float64())
}

package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// --- extractJSON -----------------------------------------------------------

func TestExtractJSON_PlainJSON(t *testing.T) {
	input := `{"title": "Beach Day", "catchy_phrase": "Sun, sand and fun"}`
	got := extractJSON(input)
	if got != input {
		t.Errorf("extractJSON plain JSON: got %q, want %q", got, input)
	}
}

func TestExtractJSON_MarkdownWrapped(t *testing.T) {
	input := "```json\n{\"title\": \"Beach Day\", \"catchy_phrase\": \"Sun and sand\"}\n```"
	got := extractJSON(input)
	var result titleResponse
	if err := json.Unmarshal([]byte(got), &result); err != nil {
		t.Errorf("extractJSON markdown-wrapped: could not parse result %q: %v", got, err)
	}
	if result.Title != "Beach Day" {
		t.Errorf("extractJSON markdown-wrapped: title = %q, want %q", result.Title, "Beach Day")
	}
}

func TestExtractJSON_LeadingTrailingText(t *testing.T) {
	input := `Here is your JSON: {"title": "Pets", "catchy_phrase": "Furry friends"} Hope that helps!`
	got := extractJSON(input)
	var result titleResponse
	if err := json.Unmarshal([]byte(got), &result); err != nil {
		t.Errorf("extractJSON with leading/trailing text: could not parse %q: %v", got, err)
	}
	if result.Title != "Pets" {
		t.Errorf("extractJSON: title = %q, want %q", result.Title, "Pets")
	}
}

func TestExtractJSON_NoJSON(t *testing.T) {
	input := "no json here at all"
	got := extractJSON(input)
	// When there are no braces, extractJSON returns the original string unchanged.
	// Attempting to unmarshal should fail, confirming no valid JSON was found.
	var result titleResponse
	if err := json.Unmarshal([]byte(got), &result); err == nil {
		t.Error("expected unmarshal to fail for input with no JSON, but it succeeded")
	}
}

func TestExtractJSON_Empty(t *testing.T) {
	got := extractJSON("")
	if got != "" {
		t.Errorf("extractJSON empty string: got %q, want %q", got, "")
	}
}

// --- truncate --------------------------------------------------------------

func TestTruncate_ShorterThanLimit(t *testing.T) {
	got := truncate("hello", 10)
	if got != "hello" {
		t.Errorf("truncate short: got %q, want %q", got, "hello")
	}
}

func TestTruncate_ExactLimit(t *testing.T) {
	got := truncate("hello", 5)
	if got != "hello" {
		t.Errorf("truncate exact: got %q, want %q", got, "hello")
	}
}

func TestTruncate_LongerThanLimit(t *testing.T) {
	got := truncate("hello world", 5)
	if got != "hello" {
		t.Errorf("truncate long: got %q, want %q", got, "hello")
	}
}

func TestTruncate_MultibyteSafe(t *testing.T) {
	// "日本語" is 3 runes; truncate to 2 should give "日本"
	got := truncate("日本語", 2)
	if got != "日本" {
		t.Errorf("truncate multibyte: got %q, want %q", got, "日本")
	}
}

func TestTruncate_Empty(t *testing.T) {
	got := truncate("", 5)
	if got != "" {
		t.Errorf("truncate empty: got %q, want empty string", got)
	}
}

// --- calculateBackoff ------------------------------------------------------

func TestCalculateBackoff_FirstAttempt(t *testing.T) {
	b := calculateBackoff(0)
	// attempt=0 → base = initialBackoff * backoffFactor^0 = initialBackoff
	// jitter adds up to jitterFactor * base, so result is in [initialBackoff, initialBackoff*(1+jitterFactor)]
	minExpected := initialBackoff
	maxExpected := time.Duration(float64(initialBackoff) * (1 + jitterFactor))
	if b < minExpected || b > maxExpected {
		t.Errorf("calculateBackoff(0) = %v, want [%v, %v]", b, minExpected, maxExpected)
	}
}

func TestCalculateBackoff_GrowsExponentially(t *testing.T) {
	b0 := calculateBackoff(0)
	b1 := calculateBackoff(1)
	// With backoffFactor=2, attempt 1 should be roughly double attempt 0.
	// Account for jitter: at minimum b1 >= 2*initialBackoff (no jitter bonus on b0, full on b1 excluded).
	// A looser check: b1 should be strictly greater than b0.
	if b1 <= b0 {
		t.Errorf("calculateBackoff should grow: backoff(1)=%v <= backoff(0)=%v", b1, b0)
	}
}

func TestCalculateBackoff_CapsAtMaxBackoff(t *testing.T) {
	// A very large attempt number should be capped.
	b := calculateBackoff(100)
	maxAllowed := time.Duration(float64(maxBackoff) * (1 + jitterFactor))
	if b > maxAllowed {
		t.Errorf("calculateBackoff(100) = %v, want <= %v (maxBackoff + jitter)", b, maxAllowed)
	}
}

// --- GenerateClusterTitle with mock server ---------------------------------

// writeTempImage writes arbitrary bytes to a temp file; GenerateClusterTitle
// only base64-encodes the bytes, so any content works.
func writeTempImage(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "*.jpg")
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(content)
	f.Close()
	return f.Name()
}

func mockGenerateServer(t *testing.T, response string, callCount *int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			http.NotFound(w, r)
			return
		}
		if callCount != nil {
			*callCount++
		}
		body, _ := json.Marshal(struct {
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}{Response: response, Done: true})
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	}))
}

func TestNewClient_TrimsTrailingSlashAndSetsDefaults(t *testing.T) {
	c := NewClient("http://example.com:1234/", "some-model")
	if c.baseURL != "http://example.com:1234" {
		t.Errorf("baseURL = %q, want trailing slash trimmed", c.baseURL)
	}
	if c.model != "some-model" {
		t.Errorf("model = %q", c.model)
	}
	if c.maxImages < 1 || c.retries < 1 {
		t.Errorf("defaults not set: maxImages=%d retries=%d", c.maxImages, c.retries)
	}
}

func TestGenerateClusterTitle_ReturnsTitleAndPhrase(t *testing.T) {
	srv := mockGenerateServer(t, `{"title": "Beach Day", "catchy_phrase": "Sun and sand"}`, nil)
	defer srv.Close()

	c := NewClient(srv.URL, "test-model")
	img := writeTempImage(t, "fake image bytes")

	title, phrase, err := c.GenerateClusterTitle(context.Background(), []string{img})
	if err != nil {
		t.Fatalf("GenerateClusterTitle: %v", err)
	}
	if title != "Beach Day" {
		t.Errorf("title = %q, want %q", title, "Beach Day")
	}
	if phrase != "Sun and sand" {
		t.Errorf("phrase = %q, want %q", phrase, "Sun and sand")
	}
}

func TestGenerateClusterTitle_ErrorAfterExhaustedRetries(t *testing.T) {
	callCount := 0
	srv := mockGenerateServer(t, "not json at all", &callCount)
	defer srv.Close()

	c := NewClient(srv.URL, "test-model")
	c.retries = 1 // avoid backoff sleeps in tests
	img := writeTempImage(t, "fake image bytes")

	title, phrase, err := c.GenerateClusterTitle(context.Background(), []string{img})
	if err == nil {
		t.Fatal("expected error after exhausting retries with invalid JSON, got nil")
	}
	// Failures must not fabricate placeholder values — the caller owns fallbacks.
	if title != "" || phrase != "" {
		t.Errorf("got title=%q phrase=%q, want empty strings on error", title, phrase)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call to mock server, got %d", callCount)
	}
}

func TestGenerateClusterTitle_SendsAtMostMaxImages(t *testing.T) {
	var gotImages int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req generateRequest
		json.NewDecoder(r.Body).Decode(&req)
		gotImages = len(req.Images)
		body, _ := json.Marshal(struct {
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}{Response: `{"title": "T", "catchy_phrase": "P"}`, Done: true})
		w.Write(body)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "test-model")
	c.maxImages = 2

	paths := []string{
		writeTempImage(t, "a"),
		writeTempImage(t, "b"),
		writeTempImage(t, "c"),
		writeTempImage(t, "d"),
	}
	if _, _, err := c.GenerateClusterTitle(context.Background(), paths); err != nil {
		t.Fatalf("GenerateClusterTitle: %v", err)
	}
	if gotImages != 2 {
		t.Errorf("server received %d images, want 2 (maxImages)", gotImages)
	}
}

func TestGenerateClusterTitle_NoReadableImagesIsError(t *testing.T) {
	c := NewClient("http://localhost:0", "test-model")
	_, _, err := c.GenerateClusterTitle(context.Background(), []string{"/nonexistent/img.jpg"})
	if err == nil {
		t.Fatal("expected error when no image can be read, got nil")
	}
}

func TestGenerateClusterTitle_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "service unavailable", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	c := &Client{
		baseURL:    srv.URL,
		model:      "test-model",
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}

	ctx := context.Background()
	_, err := c.generate(ctx, "prompt", []string{})
	if err == nil {
		t.Error("expected error for HTTP 503, got nil")
	}
	if !strings.Contains(err.Error(), "503") {
		t.Errorf("expected error to mention HTTP status 503, got: %v", err)
	}
}

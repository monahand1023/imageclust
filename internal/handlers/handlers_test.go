package handlers

import (
	"encoding/json"
	"imageclust/internal/models"
	"imageclust/internal/ollama"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/mux"
)

// --- HealthHandler tests ------------------------------------------------

func newHealthHandlers(t *testing.T, ollamaURL, modelPath string) *Handlers {
	t.Helper()
	client, err := ollama.NewClient(ollamaURL, "")
	if err != nil {
		t.Fatalf("ollama.NewClient: %v", err)
	}
	return &Handlers{
		ollamaClient: client,
		modelPath:    modelPath,
		store:        newTestStore(),
	}
}

func TestHealthHandler_ReturnsOK(t *testing.T) {
	// Start a mock Ollama server that responds 200 to GET /api/tags.
	mockOllama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" && r.Method == http.MethodGet {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer mockOllama.Close()

	// Create a temporary model file so the path-exists check passes.
	modelFile := filepath.Join(t.TempDir(), "vision_model.onnx")
	if err := os.WriteFile(modelFile, []byte("dummy"), 0644); err != nil {
		t.Fatal(err)
	}

	h := newHealthHandlers(t, mockOllama.URL, modelFile)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.HealthHandler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status code = %d, want 200", w.Code)
	}

	var resp healthResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Status != "ok" {
		t.Errorf("status = %q, want ok", resp.Status)
	}
	if resp.Version != version {
		t.Errorf("version = %q, want %q", resp.Version, version)
	}
	if resp.Checks.Ollama != "ok" {
		t.Errorf("checks.ollama = %q, want ok", resp.Checks.Ollama)
	}
	if !resp.Checks.ModelLoaded {
		t.Error("checks.model_loaded = false, want true")
	}
}

func TestHealthHandler_DegradedWhenOllamaDown(t *testing.T) {
	// Start a server and immediately close it so the port is unreachable.
	deadServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadServer.Close()

	// Model file exists so that check passes independently.
	modelFile := filepath.Join(t.TempDir(), "vision_model.onnx")
	if err := os.WriteFile(modelFile, []byte("dummy"), 0644); err != nil {
		t.Fatal(err)
	}

	h := newHealthHandlers(t, deadServer.URL, modelFile)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.HealthHandler(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status code = %d, want 503", w.Code)
	}

	var resp healthResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Status != "degraded" {
		t.Errorf("status = %q, want degraded", resp.Status)
	}
	if resp.Checks.Ollama != "unavailable" {
		t.Errorf("checks.ollama = %q, want unavailable", resp.Checks.Ollama)
	}
}

// --- sessionStore tests -------------------------------------------------

func newTestStore() *sessionStore {
	return &sessionStore{sessions: make(map[string]sessionEntry)}
}

func TestSessionStore_SetAndGet(t *testing.T) {
	s := newTestStore()
	s.set("s1", "/tmp/dir1")
	s.set("s2", "/tmp/dir2")

	dir, ok := s.get("s1")
	if !ok || dir != "/tmp/dir1" {
		t.Errorf("get(s1) = %q, %v; want /tmp/dir1, true", dir, ok)
	}

	_, ok = s.get("nonexistent")
	if ok {
		t.Error("expected nonexistent key to be missing")
	}
}

func TestSessionStore_Overwrite(t *testing.T) {
	s := newTestStore()
	s.set("s1", "/tmp/dir1")
	s.set("s1", "/tmp/dir2")

	dir, ok := s.get("s1")
	if !ok || dir != "/tmp/dir2" {
		t.Errorf("after overwrite get(s1) = %q; want /tmp/dir2", dir)
	}
}

func TestSessionStore_Concurrent(t *testing.T) {
	s := newTestStore()
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(2)
		id := string(rune('a' + i%26))
		go func(id string) { defer wg.Done(); s.set(id, "/tmp/"+id) }(id)
		go func(id string) { defer wg.Done(); s.get(id) }(id)
	}
	wg.Wait()
}

func TestSessionStore_CreatedAt(t *testing.T) {
	s := newTestStore()
	before := time.Now()
	s.set("t", "/tmp/t")
	after := time.Now()

	s.mu.RLock()
	entry := s.sessions["t"]
	s.mu.RUnlock()

	if entry.createdAt.Before(before) || entry.createdAt.After(after) {
		t.Errorf("createdAt %v outside [%v, %v]", entry.createdAt, before, after)
	}
}

// --- generateSessionID tests -------------------------------------------

func TestGenerateSessionID_Format(t *testing.T) {
	id, err := generateSessionID()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(id) != 32 {
		t.Errorf("len(id) = %d, want 32", len(id))
	}
	for _, c := range id {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			t.Errorf("non-hex character %q in session ID", c)
		}
	}
}

func TestGenerateSessionID_Uniqueness(t *testing.T) {
	seen := make(map[string]bool, 100)
	for i := 0; i < 100; i++ {
		id, err := generateSessionID()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if seen[id] {
			t.Fatalf("duplicate session ID %q", id)
		}
		seen[id] = true
	}
}

// --- CORS middleware tests ---------------------------------------------

func TestEnableCORS_Headers(t *testing.T) {
	h := EnableCORS(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Allow-Origin = %q, want *", got)
	}
	if got := w.Header().Get("Access-Control-Allow-Methods"); got != "POST, GET, OPTIONS" {
		t.Errorf("Allow-Methods = %q, want POST, GET, OPTIONS", got)
	}
}

func TestEnableCORS_Options(t *testing.T) {
	h := EnableCORS(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot) // should not be reached
	}))

	req := httptest.NewRequest(http.MethodOptions, "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("OPTIONS code = %d, want 200", w.Code)
	}
}

// --- respondWith* helpers ----------------------------------------------

func TestRespondWithJSON(t *testing.T) {
	w := httptest.NewRecorder()
	respondWithJSON(w, http.StatusOK, map[string]string{"k": "v"})
	if w.Code != http.StatusOK {
		t.Errorf("code = %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q", ct)
	}
}

func TestRespondWithError(t *testing.T) {
	w := httptest.NewRecorder()
	respondWithError(w, http.StatusBadRequest, "bad input")
	if w.Code != http.StatusBadRequest {
		t.Errorf("code = %d", w.Code)
	}
}

// --- parseClusterSizes -------------------------------------------------

func TestParseClusterSizes_Defaults(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/", nil)
	min, max := parseClusterSizes(r, 3, 6)
	if min != 3 || max != 6 {
		t.Errorf("got %d,%d; want 3,6", min, max)
	}
}

func TestParseClusterSizes_ValidCustom(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/?minClusterSize=4&maxClusterSize=10", nil)
	min, max := parseClusterSizes(r, 3, 6)
	if min != 4 || max != 10 {
		t.Errorf("got %d,%d; want 4,10", min, max)
	}
}

func TestParseClusterSizes_MinBelowFloor(t *testing.T) {
	// minClusterSize=1 is below the floor of 2 → falls back to default 3.
	r := httptest.NewRequest(http.MethodPost, "/?minClusterSize=1&maxClusterSize=8", nil)
	min, _ := parseClusterSizes(r, 3, 6)
	if min != 3 {
		t.Errorf("min below 2 should use default 3, got %d", min)
	}
}

func TestParseClusterSizes_MaxBelowMin(t *testing.T) {
	// maxClusterSize=3 is less than minClusterSize=5 → falls back to default 6.
	r := httptest.NewRequest(http.MethodPost, "/?minClusterSize=5&maxClusterSize=3", nil)
	_, max := parseClusterSizes(r, 3, 6)
	if max != 6 {
		t.Errorf("max below min should use default 6, got %d", max)
	}
}

// --- buildClusterResponse ----------------------------------------------

func TestBuildClusterResponse(t *testing.T) {
	details := map[string]models.ClusterDetails{
		"Cluster-0": {Title: "Animals", CatchyPhrase: "Furry friends", Images: []string{"a.jpg", "b.jpg"}},
		"Cluster-1": {Title: "Cities", CatchyPhrase: "Urban life", Images: []string{"c.jpg"}},
	}
	resp := buildClusterResponse("sess123", details)

	if resp.Status != "success" {
		t.Errorf("Status = %q, want success", resp.Status)
	}
	if resp.SessionID != "sess123" {
		t.Errorf("SessionID = %q, want sess123", resp.SessionID)
	}
	if len(resp.Clusters) != 2 {
		t.Fatalf("len(Clusters) = %d, want 2", len(resp.Clusters))
	}
	// Verify all cluster IDs and image lists are present.
	byID := make(map[string]clusterPayload)
	for _, c := range resp.Clusters {
		byID[c.ID] = c
	}
	if c, ok := byID["Cluster-0"]; !ok || c.Title != "Animals" || len(c.Images) != 2 {
		t.Errorf("Cluster-0 unexpected: %+v", byID["Cluster-0"])
	}
	if c, ok := byID["Cluster-1"]; !ok || c.Title != "Cities" || len(c.Images) != 1 {
		t.Errorf("Cluster-1 unexpected: %+v", c)
	}
}

// --- ServeImage / ClusterAndGenerate (handler method) ---------------

func TestServeImage_MissingSession(t *testing.T) {
	h := &Handlers{store: newTestStore()}
	req := httptest.NewRequest(http.MethodGet, "/api/image/test.jpg", nil)
	w := httptest.NewRecorder()
	h.ServeImage(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("code = %d, want 400", w.Code)
	}
}

func TestServeImage_InvalidSession(t *testing.T) {
	h := &Handlers{store: newTestStore()}
	req := httptest.NewRequest(http.MethodGet, "/api/image/test.jpg?session=bad", nil)
	w := httptest.NewRecorder()
	h.ServeImage(w, req)
	if w.Code != http.StatusNotFound {
		t.Errorf("code = %d, want 404", w.Code)
	}
}

func TestServeImage_PathTraversal(t *testing.T) {
	// The path traversal guard (filepath.Rel check) must reject ".." names even
	// though SanitizeFilename allows dots. These are the names mux.Vars would
	// supply for traversal attempts that survive URL routing.
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "images"), 0755); err != nil {
		t.Fatal(err)
	}
	s := newTestStore()
	s.set("sess", dir)
	h := &Handlers{store: s}

	for _, name := range []string{"..", "../sibling", "../../etc/passwd"} {
		req := httptest.NewRequest(http.MethodGet, "/api/image/"+name+"?session=sess", nil)
		req = mux.SetURLVars(req, map[string]string{"imageName": name})
		w := httptest.NewRecorder()
		h.ServeImage(w, req)
		if w.Code == http.StatusOK {
			t.Errorf("traversal %q: got 200, want non-200", name)
		}
	}
}

func TestClusterAndGenerate_WrongMethod(t *testing.T) {
	h := &Handlers{store: newTestStore()}
	for _, method := range []string{http.MethodGet, http.MethodPut, http.MethodDelete} {
		req := httptest.NewRequest(method, "/api/cluster", nil)
		w := httptest.NewRecorder()
		h.ClusterAndGenerate(w, req)
		if w.Code != http.StatusMethodNotAllowed {
			t.Errorf("%s: code = %d, want 405", method, w.Code)
		}
	}
}

// --- SpaHandler -------------------------------------------------------

func TestSpaHandler_Fields(t *testing.T) {
	h := SpaHandler{StaticPath: "/var/www", IndexPath: "index.html"}
	if h.StaticPath != "/var/www" || h.IndexPath != "index.html" {
		t.Error("SpaHandler fields not set correctly")
	}
}

func TestSpaHandler_FallsBackToIndex(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<html>app</html>"), 0644); err != nil {
		t.Fatal(err)
	}
	h := SpaHandler{StaticPath: dir, IndexPath: "index.html"}

	req := httptest.NewRequest(http.MethodGet, "/some/unknown/route", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("unknown route: code = %d, want 200", w.Code)
	}
	if body := w.Body.String(); body != "<html>app</html>" {
		t.Errorf("body = %q, want index.html content", body)
	}
}

func TestSpaHandler_ServesExistingFile(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "index.html"), []byte("<html>app</html>"), 0644)
	os.WriteFile(filepath.Join(dir, "app.js"), []byte("var x=1;"), 0644)

	h := SpaHandler{StaticPath: dir, IndexPath: "index.html"}

	req := httptest.NewRequest(http.MethodGet, "/app.js", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("existing file: code = %d, want 200", w.Code)
	}
	if body := w.Body.String(); body != "var x=1;" {
		t.Errorf("body = %q, want app.js content", body)
	}
}

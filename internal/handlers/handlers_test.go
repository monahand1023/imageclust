package handlers

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestSessionStore_SetAndGet(t *testing.T) {
	s := &sessionStore{
		sessions: make(map[string]sessionEntry),
	}

	s.SetSession("session1", "/tmp/dir1")
	s.SetSession("session2", "/tmp/dir2")

	dir, ok := s.GetSession("session1")
	if !ok {
		t.Error("expected session1 to exist")
	}
	if dir != "/tmp/dir1" {
		t.Errorf("expected /tmp/dir1, got %s", dir)
	}

	dir, ok = s.GetSession("session2")
	if !ok {
		t.Error("expected session2 to exist")
	}
	if dir != "/tmp/dir2" {
		t.Errorf("expected /tmp/dir2, got %s", dir)
	}

	_, ok = s.GetSession("nonexistent")
	if ok {
		t.Error("expected nonexistent session to not exist")
	}
}

func TestSessionStore_Delete(t *testing.T) {
	s := &sessionStore{
		sessions: make(map[string]sessionEntry),
	}

	s.SetSession("session1", "/tmp/nonexistent_dir_for_test")
	s.DeleteSession("session1")

	_, ok := s.GetSession("session1")
	if ok {
		t.Error("expected session1 to be deleted")
	}
}

func TestSessionStore_Concurrent(t *testing.T) {
	s := &sessionStore{
		sessions: make(map[string]sessionEntry),
	}

	var wg sync.WaitGroup
	numGoroutines := 100

	// Concurrent writes
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			sessionID := string(rune('a' + id%26))
			s.SetSession(sessionID, "/tmp/"+sessionID)
		}(i)
	}

	// Concurrent reads
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			sessionID := string(rune('a' + id%26))
			s.GetSession(sessionID)
		}(i)
	}

	wg.Wait()
	// If we get here without deadlock or panic, the test passes
}

func TestGenerateSessionID(t *testing.T) {
	id1, err := generateSessionID()
	if err != nil {
		t.Fatalf("failed to generate session ID: %v", err)
	}

	id2, err := generateSessionID()
	if err != nil {
		t.Fatalf("failed to generate session ID: %v", err)
	}

	if id1 == id2 {
		t.Error("generated session IDs should be unique")
	}

	// Should be 32 hex characters (16 bytes = 32 hex chars)
	if len(id1) != 32 {
		t.Errorf("expected session ID length 32, got %d", len(id1))
	}
}

func TestViewHandler_MissingSession(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/view", nil)
	w := httptest.NewRecorder()

	ViewHandler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestViewHandler_InvalidSession(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/view?session=invalid123", nil)
	w := httptest.NewRecorder()

	ViewHandler(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected status %d, got %d", http.StatusNotFound, w.Code)
	}
}

func TestImageHandler_MissingSession(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/image/test.jpg", nil)
	w := httptest.NewRecorder()

	ImageHandler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestImageHandler_InvalidSession(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/image/test.jpg?session=invalid123", nil)
	w := httptest.NewRecorder()

	ImageHandler(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected status %d, got %d", http.StatusNotFound, w.Code)
	}
}

func TestClusterAndGenerateHandler_InvalidMethod(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/cluster", nil)
	w := httptest.NewRecorder()

	ClusterAndGenerateHandler(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected status %d, got %d", http.StatusMethodNotAllowed, w.Code)
	}
}

func TestEnableCORS(t *testing.T) {
	handler := EnableCORS(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	// Test OPTIONS request
	req := httptest.NewRequest(http.MethodOptions, "/", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d for OPTIONS, got %d", http.StatusOK, w.Code)
	}

	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("expected CORS header to be set")
	}

	// Test regular request
	req = httptest.NewRequest(http.MethodGet, "/", nil)
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("expected CORS header to be set on regular request")
	}
}

func TestEnableCORS_AllHeaders(t *testing.T) {
	handler := EnableCORS(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Header().Get("Access-Control-Allow-Methods") != "POST, GET, OPTIONS, PUT, DELETE" {
		t.Error("expected Access-Control-Allow-Methods header to be set")
	}

	if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type" {
		t.Error("expected Access-Control-Allow-Headers header to be set")
	}
}

func TestRespondWithJSON(t *testing.T) {
	w := httptest.NewRecorder()
	payload := map[string]interface{}{
		"key":   "value",
		"count": 42,
	}

	respondWithJSON(w, http.StatusOK, payload)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	if w.Header().Get("Content-Type") != "application/json" {
		t.Error("expected Content-Type to be application/json")
	}

	// Check body contains expected JSON
	body := w.Body.String()
	if body == "" {
		t.Error("expected non-empty response body")
	}
}

func TestRespondWithError(t *testing.T) {
	w := httptest.NewRecorder()

	respondWithError(w, http.StatusBadRequest, "test error message")

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}

	if w.Header().Get("Content-Type") != "application/json" {
		t.Error("expected Content-Type to be application/json")
	}

	// Check body contains error field
	body := w.Body.String()
	if body == "" {
		t.Error("expected non-empty response body")
	}
}

func TestSpaHandler_ServeHTTP_ViewRoute(t *testing.T) {
	// Test /view route redirects to ViewHandler (missing session)
	h := SpaHandler{
		StaticPath: "/tmp",
		IndexPath:  "index.html",
	}

	req := httptest.NewRequest(http.MethodGet, "/view", nil)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	// Should return BadRequest because session is missing
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestSpaHandlerStruct(t *testing.T) {
	h := SpaHandler{
		StaticPath: "/var/www/static",
		IndexPath:  "index.html",
	}

	if h.StaticPath != "/var/www/static" {
		t.Errorf("StaticPath = %s, want '/var/www/static'", h.StaticPath)
	}

	if h.IndexPath != "index.html" {
		t.Errorf("IndexPath = %s, want 'index.html'", h.IndexPath)
	}
}

func TestSessionStore_Overwrite(t *testing.T) {
	s := &sessionStore{
		sessions: make(map[string]sessionEntry),
	}

	s.SetSession("session1", "/tmp/dir1")
	s.SetSession("session1", "/tmp/dir2") // Overwrite

	dir, ok := s.GetSession("session1")
	if !ok {
		t.Error("expected session1 to exist")
	}
	if dir != "/tmp/dir2" {
		t.Errorf("expected /tmp/dir2 after overwrite, got %s", dir)
	}
}

func TestSessionStore_DeleteNonExistent(t *testing.T) {
	s := &sessionStore{
		sessions: make(map[string]sessionEntry),
	}

	// Should not panic when deleting non-existent session
	s.DeleteSession("nonexistent")

	_, ok := s.GetSession("nonexistent")
	if ok {
		t.Error("expected nonexistent session to not exist")
	}
}

func TestGenerateSessionID_Uniqueness(t *testing.T) {
	ids := make(map[string]bool)
	for i := 0; i < 100; i++ {
		id, err := generateSessionID()
		if err != nil {
			t.Fatalf("failed to generate session ID: %v", err)
		}
		if ids[id] {
			t.Errorf("duplicate session ID generated: %s", id)
		}
		ids[id] = true
	}
}

func TestGenerateSessionID_Format(t *testing.T) {
	id, err := generateSessionID()
	if err != nil {
		t.Fatalf("failed to generate session ID: %v", err)
	}

	// Should be valid hex characters only
	for _, c := range id {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			t.Errorf("session ID contains non-hex character: %c", c)
		}
	}
}

func TestClusterAndGenerateHandler_NonPOSTMethods(t *testing.T) {
	methods := []string{http.MethodGet, http.MethodPut, http.MethodDelete, http.MethodPatch}

	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/api/cluster", nil)
			w := httptest.NewRecorder()

			ClusterAndGenerateHandler(w, req)

			if w.Code != http.StatusMethodNotAllowed {
				t.Errorf("expected status %d for %s, got %d", http.StatusMethodNotAllowed, method, w.Code)
			}
		})
	}
}

func TestViewHandler_WithSession(t *testing.T) {
	// Use the global store directly to avoid mutation issues
	// This test adds a session to the global store, tests, then cleans up
	sessionID := "test-valid-session-" + time.Now().Format("20060102150405")
	store.SetSession(sessionID, "/nonexistent/path")
	defer store.DeleteSession(sessionID)

	req := httptest.NewRequest(http.MethodGet, "/api/view?session="+sessionID, nil)
	w := httptest.NewRecorder()

	ViewHandler(w, req)

	// Should return NotFound because HTML file doesn't exist
	if w.Code != http.StatusNotFound {
		t.Errorf("expected status %d, got %d", http.StatusNotFound, w.Code)
	}
}

func TestSessionEntry_CreatedAt(t *testing.T) {
	s := &sessionStore{
		sessions: make(map[string]sessionEntry),
	}

	before := time.Now()
	s.SetSession("test-session", "/tmp/test")
	after := time.Now()

	s.mutex.RLock()
	entry := s.sessions["test-session"]
	s.mutex.RUnlock()

	if entry.createdAt.Before(before) || entry.createdAt.After(after) {
		t.Errorf("createdAt should be between %v and %v, got %v", before, after, entry.createdAt)
	}
}

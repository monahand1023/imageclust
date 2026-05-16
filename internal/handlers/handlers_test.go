package handlers

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

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

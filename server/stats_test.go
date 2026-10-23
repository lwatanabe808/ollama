package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
)

// newTestStats returns stats saved under a temp dir, with a clock the test controls.
func newTestStats(t *testing.T, now *time.Time) *usageStats {
	t.Helper()
	u := newUsageStats(filepath.Join(t.TempDir(), "stats.json"))
	u.now = func() time.Time { return *now }
	u.data.Since = *now
	return u
}

func findStats(t *testing.T, models []api.ModelStats, name string) api.ModelStats {
	t.Helper()
	for _, m := range models {
		if m.Model == name {
			return m
		}
	}
	t.Fatalf("no stats for model %q in %+v", name, models)
	return api.ModelStats{}
}

func TestUsageStatsRecord(t *testing.T) {
	now := time.Date(2026, 10, 22, 9, 0, 0, 0, time.Local)
	u := newTestStats(t, &now)

	u.record("llama3.2:latest", api.Metrics{
		PromptEvalCount:    10,
		PromptEvalDuration: 100 * time.Millisecond,
		EvalCount:          50,
		EvalDuration:       time.Second,
		TotalDuration:      2 * time.Second,
		LoadDuration:       500 * time.Millisecond,
	}, statsRequestInfo{}, false)
	u.record("llama3.2:latest", api.Metrics{
		PromptEvalCount: 20,
		EvalCount:       150,
		EvalDuration:    time.Second,
		TotalDuration:   4 * time.Second,
	}, statsRequestInfo{}, false)
	u.record("llama3.2:latest", api.Metrics{}, statsRequestInfo{}, true)
	u.record("phi3:latest", api.Metrics{EvalCount: 1, EvalDuration: time.Second, TotalDuration: time.Second}, statsRequestInfo{}, false)

	snap := u.snapshot()
	if snap.Today != "2026-10-22" {
		t.Errorf("today = %q, want 2026-10-22", snap.Today)
	}
	if len(snap.AllTimeModels) != 2 || snap.AllTimeModels[0].Model != "llama3.2:latest" {
		t.Fatalf("models should be sorted by most requests, got %+v", snap.AllTimeModels)
	}

	got := findStats(t, snap.TodayModels, "llama3.2:latest")
	if got.Requests != 3 || got.Errors != 1 {
		t.Errorf("requests, errors = %d, %d, want 3, 1", got.Requests, got.Errors)
	}
	if got.PromptTokens != 30 || got.OutputTokens != 200 {
		t.Errorf("prompt, output tokens = %d, %d, want 30, 200", got.PromptTokens, got.OutputTokens)
	}
	if tps := got.TokensPerSecond(); tps != 100 {
		t.Errorf("tokens per second = %v, want 100", tps)
	}
	// Errors don't count toward the average time.
	if avg := got.AvgDuration(); avg != 3*time.Second {
		t.Errorf("avg duration = %v, want 3s", avg)
	}
	if !got.LastUsed.Equal(now) {
		t.Errorf("last used = %v, want %v", got.LastUsed, now)
	}
}

func TestUsageStatsTodayAndAllTime(t *testing.T) {
	now := time.Date(2026, 10, 21, 23, 59, 0, 0, time.Local)
	u := newTestStats(t, &now)

	u.record("m", api.Metrics{EvalCount: 5}, statsRequestInfo{}, false)
	now = now.Add(2 * time.Minute) // past midnight
	u.record("m", api.Metrics{EvalCount: 7}, statsRequestInfo{}, false)

	snap := u.snapshot()
	if got := findStats(t, snap.TodayModels, "m"); got.Requests != 1 || got.OutputTokens != 7 {
		t.Errorf("today = %+v, want 1 request and 7 tokens", got)
	}
	if got := findStats(t, snap.AllTimeModels, "m"); got.Requests != 2 || got.OutputTokens != 12 {
		t.Errorf("all time = %+v, want 2 requests and 12 tokens", got)
	}

	// A new day with no requests yet shows an empty today list.
	now = now.Add(24 * time.Hour)
	if snap := u.snapshot(); len(snap.TodayModels) != 0 {
		t.Errorf("today should be empty on a new day, got %+v", snap.TodayModels)
	}
}

func TestUsageStatsPrunesOldDays(t *testing.T) {
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.Local)
	u := newTestStats(t, &now)
	u.record("m", api.Metrics{}, statsRequestInfo{}, false)

	now = now.AddDate(0, 0, statsKeepDays+5)
	u.record("m", api.Metrics{}, statsRequestInfo{}, false)

	if _, ok := u.data.Days["2026-01-01"]; ok {
		t.Error("days older than the keep window should be removed")
	}
	if got := findStats(t, u.snapshot().AllTimeModels, "m"); got.Requests != 2 {
		t.Errorf("all-time totals must keep old days, got %d requests", got.Requests)
	}
}

func TestUsageStatsSurviveRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "stats.json")

	first := newUsageStats(path)
	first.record("m", api.Metrics{EvalCount: 10, EvalDuration: time.Second, TotalDuration: time.Second}, statsRequestInfo{}, false)
	if err := first.close(); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stats file was not written: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("file permissions = %o, want 600", perm)
	}

	// A new server picks up where the old one stopped.
	second := newUsageStats(path)
	second.record("m", api.Metrics{EvalCount: 5, EvalDuration: time.Second, TotalDuration: time.Second}, statsRequestInfo{}, false)

	got := findStats(t, second.snapshot().AllTimeModels, "m")
	if got.Requests != 2 || got.OutputTokens != 15 {
		t.Errorf("after restart = %+v, want 2 requests and 15 tokens", got)
	}
	if !second.snapshot().Since.Equal(first.data.Since) {
		t.Error("tracking start date should carry over across restarts")
	}
}

func TestUsageStatsSaveOnlyWhenChanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stats.json")
	u := newUsageStats(path)

	if err := u.save(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Error("nothing recorded yet, so no file should be written")
	}

	u.record("m", api.Metrics{}, statsRequestInfo{}, false)
	if err := u.save(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("file should exist after a request: %v", err)
	}
}

func TestUsageStatsCorruptFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stats.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}

	u := newUsageStats(path)
	if n := len(u.snapshot().AllTimeModels); n != 0 {
		t.Errorf("should start fresh, got %d models", n)
	}
	if _, err := os.Stat(path + ".corrupt"); err != nil {
		t.Errorf("bad file should be kept as stats.json.corrupt: %v", err)
	}
}

func TestUsageStatsReset(t *testing.T) {
	now := time.Date(2026, 10, 22, 9, 0, 0, 0, time.Local)
	u := newTestStats(t, &now)
	u.record("m", api.Metrics{}, statsRequestInfo{}, false)

	now = now.Add(time.Hour)
	if err := u.reset(); err != nil {
		t.Fatal(err)
	}

	snap := u.snapshot()
	if len(snap.AllTimeModels) != 0 || len(snap.TodayModels) != 0 {
		t.Errorf("reset should clear everything, got %+v", snap)
	}
	if !snap.Since.Equal(now) {
		t.Errorf("since = %v, want reset time %v", snap.Since, now)
	}

	// The reset is saved right away, so a restart doesn't bring old stats back.
	if n := len(newUsageStats(u.path).snapshot().AllTimeModels); n != 0 {
		t.Errorf("reset was not saved, reloaded %d models", n)
	}
}

func TestUsageStatsObserve(t *testing.T) {
	now := time.Now()
	u := newTestStats(t, &now)

	t.Run("passes items through and records once", func(t *testing.T) {
		in := make(chan any)
		out := u.observe(context.Background(), "m", statsRequestInfo{}, in)

		go func() {
			in <- api.ChatResponse{Message: api.Message{Content: "hi"}}
			in <- api.ChatResponse{Done: true, Metrics: api.Metrics{EvalCount: 3}}
			in <- gin.H{"error": "late error"} // already recorded, should not count
			close(in)
		}()

		var items []any
		for v := range out {
			items = append(items, v)
		}
		if len(items) != 3 {
			t.Fatalf("got %d items, want all 3 passed through", len(items))
		}

		got := findStats(t, u.snapshot().AllTimeModels, "m")
		if got.Requests != 1 || got.Errors != 0 || got.OutputTokens != 3 {
			t.Errorf("stats = %+v, want 1 request, 0 errors, 3 tokens", got)
		}
		if a := u.active.Load(); a != 0 {
			t.Errorf("active = %d, want 0 after the request ends", a)
		}
	})

	t.Run("counts errors", func(t *testing.T) {
		in := make(chan any, 1)
		in <- gin.H{"error": "boom"}
		close(in)
		for range u.observe(context.Background(), "broken", statsRequestInfo{}, in) {
		}

		if got := findStats(t, u.snapshot().AllTimeModels, "broken"); got.Errors != 1 {
			t.Errorf("errors = %d, want 1", got.Errors)
		}
	})

	t.Run("does not block when the client goes away", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		in := make(chan any)
		_ = u.observe(ctx, "m", statsRequestInfo{}, in) // nobody reads the output
		cancel()

		sent := make(chan struct{})
		go func() {
			defer close(sent)
			in <- api.GenerateResponse{Response: "a"}
			in <- api.GenerateResponse{Done: true}
			close(in)
		}()

		select {
		case <-sent:
		case <-time.After(5 * time.Second):
			t.Fatal("sender blocked after the client went away")
		}
	})

	t.Run("nil stats is a no-op", func(t *testing.T) {
		var none *usageStats
		in := make(chan any)
		if out := none.observe(context.Background(), "m", statsRequestInfo{}, in); out != in {
			t.Error("nil stats should return the channel unchanged")
		}
	})
}

func TestStatsHandlers(t *testing.T) {
	gin.SetMode(gin.TestMode)
	now := time.Date(2026, 10, 22, 9, 0, 0, 0, time.Local)
	s := &Server{stats: newTestStats(t, &now)}
	s.stats.record("m", api.Metrics{EvalCount: 4}, statsRequestInfo{}, false)

	router := gin.New()
	router.GET("/api/stats", s.StatsHandler)
	router.DELETE("/api/stats", s.ResetStatsHandler)

	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/stats", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("GET status = %d", w.Code)
	}
	var resp api.StatsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if got := findStats(t, resp.AllTimeModels, "m"); got.OutputTokens != 4 {
		t.Errorf("output tokens = %d, want 4", got.OutputTokens)
	}

	w = httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest(http.MethodDelete, "/api/stats", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("DELETE status = %d", w.Code)
	}
	if n := len(s.stats.snapshot().AllTimeModels); n != 0 {
		t.Errorf("reset left %d models", n)
	}
}

// TestStatsFromRealRequests sends chat and generate requests through the real
// handlers with a fake model, and checks they show up in the stats.
func TestStatsFromRealRequests(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	gin.SetMode(gin.TestMode)

	failNext := false
	mock := mockRunner{
		CompletionFn: func(_ context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			if failNext {
				return errors.New("model crashed")
			}
			fn(llm.CompletionResponse{Content: "hello "})
			fn(llm.CompletionResponse{
				Content:            "there",
				Done:               true,
				DoneReason:         llm.DoneReasonStop,
				PromptEvalCount:    4,
				PromptEvalDuration: 10 * time.Millisecond,
				EvalCount:          20,
				EvalDuration:       200 * time.Millisecond,
			})
			return nil
		},
	}
	mock.contextLength = 4096
	mock.memorySize = 3 << 30
	s := newServerWithMockRunner(t, &mock)
	now := time.Now()
	s.stats = newTestStats(t, &now)
	createMinimalGGUFModel(t, s, "stats-test", gguftest.KV{}, "{{ .Prompt }}", nil)

	// Streaming generate request.
	w := createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "stats-test", Prompt: "hi"})
	if w.Code != http.StatusOK {
		t.Fatalf("generate status = %d: %s", w.Code, w.Body.String())
	}

	// Non-streaming chat request, using the short name without ":latest".
	noStream := false
	w = createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "stats-test",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Stream:   &noStream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("chat status = %d: %s", w.Code, w.Body.String())
	}

	// A request that fails while the model is answering.
	failNext = true
	w = createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "stats-test",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Stream:   &noStream,
	})
	if w.Code == http.StatusOK {
		t.Fatalf("expected the failing request to return an error")
	}

	got := findStats(t, s.stats.snapshot().AllTimeModels, "stats-test:latest")
	if got.Requests != 3 || got.Errors != 1 {
		t.Errorf("requests, errors = %d, %d, want 3, 1", got.Requests, got.Errors)
	}
	if got.PromptTokens != 8 || got.OutputTokens != 40 {
		t.Errorf("prompt, output tokens = %d, %d, want 8, 40", got.PromptTokens, got.OutputTokens)
	}
	if tps := got.TokensPerSecond(); tps != 100 {
		t.Errorf("speed = %v tokens/s, want 100", tps)
	}
	if got.TotalDuration <= 0 {
		t.Error("total duration should be recorded")
	}
	// Each request used 4 prompt tokens and 20 answer tokens.
	if got.MaxContextUsed != 24 || got.ContextLength != 4096 {
		t.Errorf("context = %d of %d, want 24 of 4096", got.MaxContextUsed, got.ContextLength)
	}
	if got.MemorySize != 3<<30 {
		t.Errorf("memory size = %d, want 3 GiB", got.MemorySize)
	}
	if a := s.stats.active.Load(); a != 0 {
		t.Errorf("active = %d, want 0 when no requests are running", a)
	}
}

// TestStatsFromNativeChat covers models that use their own built-in chat
// template, which goes through a different code path than Go templates.
func TestStatsFromNativeChat(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("OLLAMA_GO_TEMPLATE", "")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		ChatFn: func(_ context.Context, _ llm.ChatRequest, fn func(llm.ChatResponse)) error {
			fn(llm.ChatResponse{
				Message:      api.Message{Role: "assistant", Content: "hi"},
				Done:         true,
				DoneReason:   llm.DoneReasonStop,
				EvalCount:    6,
				EvalDuration: 3 * time.Second,
			})
			return nil
		},
	}
	s := newServerWithMockRunner(t, &mock)
	now := time.Now()
	s.stats = newTestStats(t, &now)
	createMinimalGGUFModel(t, s, "native-chat", gguftest.KV{
		"tokenizer.chat_template": "{{ messages[0]['content'] }}",
	}, "", nil)

	w := createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "native-chat",
		Messages: []api.Message{{Role: "user", Content: "hello"}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("chat status = %d: %s", w.Code, w.Body.String())
	}

	got := findStats(t, s.stats.snapshot().AllTimeModels, "native-chat:latest")
	if got.Requests != 1 || got.OutputTokens != 6 || got.TokensPerSecond() != 2 {
		t.Errorf("stats = %+v, want 1 request, 6 tokens, 2 tokens/s", got)
	}
}

func TestUsageStatsWaitAndContext(t *testing.T) {
	now := time.Now()
	u := newTestStats(t, &now)

	// Request 1: 1.5s total, 1s of it writing, so 0.5s before the first word.
	u.record("m", api.Metrics{
		PromptEvalCount: 1000,
		EvalCount:       200,
		EvalDuration:    time.Second,
		TotalDuration:   1500 * time.Millisecond,
	}, statsRequestInfo{ContextLength: 8192}, false)
	// Request 2: a bigger chat. 2.5s total, 1s writing, so 1.5s wait.
	u.record("m", api.Metrics{
		PromptEvalCount: 6000,
		EvalCount:       500,
		EvalDuration:    time.Second,
		TotalDuration:   2500 * time.Millisecond,
	}, statsRequestInfo{ContextLength: 8192}, false)
	// A failed request doesn't change the wait or context numbers.
	u.record("m", api.Metrics{}, statsRequestInfo{ContextLength: 8192}, true)

	got := findStats(t, u.snapshot().AllTimeModels, "m")
	if wait := got.AvgWait(); wait != time.Second {
		t.Errorf("avg wait = %v, want 1s", wait)
	}
	if got.MaxContextUsed != 6500 {
		t.Errorf("max context used = %d, want 6500 (largest prompt + answer)", got.MaxContextUsed)
	}
	if got.ContextLength != 8192 {
		t.Errorf("context length = %d, want 8192", got.ContextLength)
	}

	// A later, smaller request doesn't lower the max.
	u.record("m", api.Metrics{PromptEvalCount: 10, EvalCount: 10}, statsRequestInfo{ContextLength: 8192}, false)
	if got := findStats(t, u.snapshot().AllTimeModels, "m"); got.MaxContextUsed != 6500 {
		t.Errorf("max context used dropped to %d", got.MaxContextUsed)
	}
}

func TestUsageStatsLastRequest(t *testing.T) {
	now := time.Date(2026, 10, 22, 9, 0, 0, 0, time.Local)
	u := newTestStats(t, &now)

	u.record("m", api.Metrics{
		PromptEvalCount: 10,
		EvalCount:       100,
		EvalDuration:    time.Second,
		TotalDuration:   3 * time.Second,
	}, statsRequestInfo{}, false)

	now = now.Add(time.Minute)
	u.record("m", api.Metrics{
		PromptEvalCount: 20,
		EvalCount:       50,
		EvalDuration:    time.Second,
		TotalDuration:   1500 * time.Millisecond,
	}, statsRequestInfo{}, false)

	// A failed request after it doesn't replace the last good one.
	now = now.Add(time.Minute)
	u.record("m", api.Metrics{}, statsRequestInfo{}, true)

	got := findStats(t, u.snapshot().AllTimeModels, "m")
	if got.Last == nil {
		t.Fatal("last request should be saved")
	}
	last := *got.Last
	if !last.At.Equal(now.Add(-time.Minute)) {
		t.Errorf("last.At = %v, want the second request's time", last.At)
	}
	if last.PromptTokens != 20 || last.OutputTokens != 50 || last.ContextUsed() != 70 {
		t.Errorf("last tokens = %d in, %d out, want 20 in, 50 out", last.PromptTokens, last.OutputTokens)
	}
	if last.TokensPerSecond() != 50 {
		t.Errorf("last speed = %v, want 50", last.TokensPerSecond())
	}
	if last.Wait() != 500*time.Millisecond {
		t.Errorf("last wait = %v, want 500ms", last.Wait())
	}

	// Changing the returned copy must not change the saved stats.
	got.Last.OutputTokens = 999
	if again := findStats(t, u.snapshot().AllTimeModels, "m"); again.Last.OutputTokens != 50 {
		t.Error("snapshot shares memory with the saved stats")
	}

	// The last request survives a restart.
	if err := u.close(); err != nil {
		t.Fatal(err)
	}
	reloaded := findStats(t, newUsageStats(u.path).snapshot().AllTimeModels, "m")
	if reloaded.Last == nil || reloaded.Last.OutputTokens != 50 {
		t.Errorf("last request not saved across restart: %+v", reloaded.Last)
	}
}

func TestUsageStatsDays(t *testing.T) {
	now := time.Date(2026, 10, 20, 9, 0, 0, 0, time.Local)
	u := newTestStats(t, &now)

	u.record("a", api.Metrics{EvalCount: 10}, statsRequestInfo{}, false)
	u.record("b", api.Metrics{EvalCount: 5}, statsRequestInfo{}, false)
	now = now.AddDate(0, 0, 2)
	u.record("a", api.Metrics{EvalCount: 1}, statsRequestInfo{}, false)
	u.record("a", api.Metrics{}, statsRequestInfo{}, true)

	days := u.snapshot().Days
	want := []api.DayStats{
		{Date: "2026-10-20", Requests: 2, OutputTokens: 15},
		{Date: "2026-10-22", Requests: 2, Errors: 1, OutputTokens: 1},
	}
	if len(days) != len(want) {
		t.Fatalf("days = %+v, want %+v", days, want)
	}
	for i := range want {
		if days[i] != want[i] {
			t.Errorf("day %d = %+v, want %+v", i, days[i], want[i])
		}
	}
}

func TestStatsDashboard(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	router := gin.New()
	router.GET("/stats", s.StatsDashboardHandler)
	router.GET("/stats/grading.js", s.StatsGradingScriptHandler)

	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/stats", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/html; charset=utf-8" {
		t.Errorf("content type = %q", ct)
	}
	csp := w.Header().Get("Content-Security-Policy")
	if !strings.Contains(csp, "connect-src 'self'") || !strings.Contains(csp, "default-src 'none'") || !strings.Contains(csp, "script-src 'self'") {
		t.Errorf("dashboard should only talk to and load from this server, got CSP %q", csp)
	}
	body := w.Body.String()
	for _, want := range []string{"<title>Ollama Usage</title>", `getJSON("/api/stats"`, `"/api/stats/log?`, `id="tags-card"`, `id="tag-search"`, `id="chart-time"`, `id="cost-card"`, `<script src="/stats/grading.js">`} {
		if !strings.Contains(body, want) {
			t.Errorf("page missing %q", want)
		}
	}
	// The page must not load anything from outside, so it works offline.
	// Its only script file comes from this server.
	if n := strings.Count(body, "<script src"); n != 1 {
		t.Errorf("page loads %d script files, want only grading.js", n)
	}
	for _, bad := range []string{"<link ", "https://"} {
		if strings.Contains(body, bad) {
			t.Errorf("page should not load outside files, found %q", bad)
		}
	}

	// The grading script it loads is served from the same server.
	w = httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/stats/grading.js", nil))
	if w.Code != http.StatusOK || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/javascript") {
		t.Fatalf("grading.js: status %d, type %q", w.Code, w.Header().Get("Content-Type"))
	}
	if !strings.Contains(w.Body.String(), "OllamaGrading") {
		t.Error("grading.js doesn't define OllamaGrading")
	}
}

func TestUsageStatsTimeSplitReuseAndGPU(t *testing.T) {
	now := time.Now()
	u := newTestStats(t, &now)

	cached := 30
	u.record("m", api.Metrics{
		LoadDuration:          2 * time.Second,
		PromptEvalCount:       40,
		PromptEvalCachedCount: &cached,
		PromptEvalDuration:    500 * time.Millisecond,
		EvalCount:             10,
		EvalDuration:          time.Second,
		TotalDuration:         3600 * time.Millisecond,
	}, statsRequestInfo{MemorySize: 1000, MemoryVRAM: 600}, false)
	u.record("m", api.Metrics{
		PromptEvalCount:    60,
		PromptEvalDuration: 1500 * time.Millisecond,
		EvalCount:          10,
		EvalDuration:       time.Second,
		TotalDuration:      2600 * time.Millisecond,
	}, statsRequestInfo{MemorySize: 1000, MemoryVRAM: 600}, false)

	got := findStats(t, u.snapshot().AllTimeModels, "m")
	if got.AvgLoad() != time.Second || got.AvgPromptRead() != time.Second || got.AvgWriting() != time.Second {
		t.Errorf("avg load/read/write = %v/%v/%v, want 1s each", got.AvgLoad(), got.AvgPromptRead(), got.AvgWriting())
	}
	// 30 of 100 prompt tokens were reused.
	if share, ok := got.PromptReused(); !ok || share != 0.3 {
		t.Errorf("prompt reused = %v (%v), want 0.3", share, ok)
	}
	if share, ok := got.GPUShare(); !ok || share != 0.6 {
		t.Errorf("GPU share = %v (%v), want 0.6", share, ok)
	}

	// The last request keeps its own split, and so does the log.
	if got.Last.LoadDuration != 0 || got.Last.PromptDuration != 1500*time.Millisecond {
		t.Errorf("last request split = %+v", got.Last)
	}
	first := u.logQuery("", 10).Requests[1]
	if first.LoadDuration != 2*time.Second || first.PromptCachedTokens != 30 {
		t.Errorf("logged request = %+v, want 2s load and 30 reused tokens", first)
	}
	if first.PromptTokens != 40 || first.PromptNewTokens() != 10 {
		t.Errorf("logged prompt = %d tokens, %d new; want 40 with 10 new", first.PromptTokens, first.PromptNewTokens())
	}

	// Totals rebuilt from the log match the saved totals.
	fromLog := findStats(t, u.logQuery("", 10).Models, "m")
	if fromLog.AvgLoad() != time.Second || fromLog.PromptCachedTokens != 30 {
		t.Errorf("totals from log = %+v", fromLog)
	}
}

func TestModelStatsGPUShareUnknown(t *testing.T) {
	// Stats saved before the GPU split was tracked have a size but no split.
	m := api.ModelStats{MemorySize: 1000}
	if _, ok := m.GPUShare(); ok {
		t.Error("GPU share should be unknown when the split was never saved, not 100% CPU")
	}
	zero := int64(0)
	m.MemoryVRAM = &zero
	if share, ok := m.GPUShare(); !ok || share != 0 {
		t.Errorf("all on CPU should be share 0, got %v %v", share, ok)
	}
}

func TestUsageStatsStoppedRequestIsNotAnError(t *testing.T) {
	now := time.Now()
	u := newTestStats(t, &now)
	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan any, 2)
	out := u.observe(ctx, "m", statsRequestInfo{}, in)

	cancel() // the caller stopped the answer
	in <- gin.H{"error": "context canceled"}
	close(in)
	for range out {
	}

	if n := len(u.snapshot().AllTimeModels); n != 0 {
		t.Errorf("a stopped request was counted (%d models in stats)", n)
	}
}

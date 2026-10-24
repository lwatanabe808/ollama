package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

// fakeCloud stands in for Ollama's cloud. reply picks the status and body
// for each request path.
func fakeCloud(t *testing.T, reply func(path string) (int, string)) {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		status, body := reply(r.URL.Path)
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(status)
		// Send the body in small pieces, so lines arrive split across writes.
		for i := 0; i < len(body); i += 7 {
			_, _ = w.Write([]byte(body[i:min(i+7, len(body))]))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	t.Cleanup(upstream.Close)

	originalURL, originalSign := cloudProxyBaseURL, cloudProxySignRequest
	cloudProxyBaseURL = upstream.URL
	cloudProxySignRequest = func(context.Context, *http.Request) error { return nil }
	t.Cleanup(func() { cloudProxyBaseURL, cloudProxySignRequest = originalURL, originalSign })
}

const (
	cloudChatStream = `{"model":"m","message":{"role":"assistant","content":"Hel"},"done":false}
{"model":"m","message":{"role":"assistant","content":"lo"},"done":false}
{"model":"m","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":900000000,"load_duration":500000000,"prompt_eval_count":12,"prompt_eval_duration":20000000,"eval_count":40,"eval_duration":400000000}
`
	cloudGenerateOnce = `{"model":"m","response":"Hi","done":true,"prompt_eval_count":5,"eval_count":8,"eval_duration":100000000}`
)

func cloudRouter(t *testing.T, s *Server) http.Handler {
	t.Helper()
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	return router
}

func sendCloud(t *testing.T, router http.Handler, path, body string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	return w
}

func TestStatsCloudRequests(t *testing.T) {
	fakeCloud(t, func(path string) (int, string) {
		if path == "/api/generate" {
			return http.StatusOK, cloudGenerateOnce
		}
		return http.StatusOK, cloudChatStream
	})
	now := time.Now()
	s := &Server{stats: newTestStats(t, &now)}
	router := cloudRouter(t, s)

	chat := `{"model":"test-model:cloud","messages":[{"role":"user","content":"hi"}]}`
	w := sendCloud(t, router, "/api/chat", chat, map[string]string{statsTagHeader: "compare"})
	if w.Code != http.StatusOK || w.Body.String() != cloudChatStream {
		t.Fatalf("the answer should pass through unchanged: status %d, body %q", w.Code, w.Body.String())
	}
	w = sendCloud(t, router, "/api/generate", `{"model":"test-model:cloud","prompt":"hi","stream":false}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("generate status %d", w.Code)
	}

	got := findStats(t, s.stats.snapshot().AllTimeModels, "test-model:cloud")
	if !got.Cloud || got.Requests != 2 || got.Errors != 0 {
		t.Fatalf("cloud stats = %+v, want 2 cloud requests", got)
	}
	if got.OutputTokens != 48 || got.PromptTokens != 17 || got.OutputDuration != 500*time.Millisecond {
		t.Errorf("tokens and writing time should come from the cloud's numbers: %+v", got)
	}
	if got.LoadDuration != 0 || got.MemorySize != 0 {
		t.Errorf("loading and memory don't apply to cloud: load %v, memory %d", got.LoadDuration, got.MemorySize)
	}
	// Total time is measured here, not taken from the cloud's 900ms.
	if got.Last == nil || got.Last.TotalDuration <= 0 || got.Last.TotalDuration > 5*time.Second {
		t.Errorf("total time should be measured on this computer, got %+v", got.Last)
	}

	tagged := s.stats.logQuery("compare", 10)
	if tagged.Total != 1 || !tagged.Requests[0].Cloud || tagged.Requests[0].Model != "test-model:cloud" {
		t.Errorf("tagged cloud request = %+v", tagged)
	}
	if st := s.stats.snapshot(); st.Active != 0 {
		t.Errorf("%d requests still marked running", st.Active)
	}
}

func TestStatsCloudOutcomes(t *testing.T) {
	chat := `{"model":"test-model:cloud","messages":[{"role":"user","content":"hi"}]}`
	for _, tt := range []struct {
		name      string
		status    int
		body      string
		headers   map[string]string
		wantLog   int64
		wantUsage int64
		wantError int64
	}{
		{name: "not signed in isn't counted", status: http.StatusUnauthorized, body: `{"error":"unauthorized"}`},
		{name: "unknown model isn't counted", status: http.StatusNotFound, body: `{"error":"model not found"}`},
		{name: "cloud server error is an error", status: http.StatusBadGateway, body: `{"error":"upstream down"}`, wantLog: 1, wantUsage: 1, wantError: 1},
		{name: "error in the answer is an error", status: http.StatusOK, body: `{"message":{"content":"Hi"},"done":false}` + "\n" + `{"error":"model crashed"}` + "\n", wantLog: 1, wantUsage: 1, wantError: 1},
		{name: "answer that never finishes is an error", status: http.StatusOK, body: `{"message":{"content":"Hi"},"done":false}` + "\n", wantLog: 1, wantUsage: 1, wantError: 1},
		{name: "test requests are logged, not counted", status: http.StatusOK, body: cloudChatStream, headers: map[string]string{statsKindHeader: "test"}, wantLog: 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fakeCloud(t, func(string) (int, string) { return tt.status, tt.body })
			now := time.Now()
			s := &Server{stats: newTestStats(t, &now)}
			sendCloud(t, cloudRouter(t, s), "/api/chat", chat, tt.headers)

			if got := int64(s.stats.logQuery("", 10).Total); got != tt.wantLog {
				t.Errorf("logged %d requests, want %d", got, tt.wantLog)
			}
			var usage, errs int64
			for _, m := range s.stats.snapshot().AllTimeModels {
				usage += m.Requests
				errs += m.Errors
			}
			if usage != tt.wantUsage || errs != tt.wantError {
				t.Errorf("usage %d requests, %d errors; want %d and %d", usage, errs, tt.wantUsage, tt.wantError)
			}
		})
	}
}

func TestStatsCloudWhenOff(t *testing.T) {
	fakeCloud(t, func(string) (int, string) { return http.StatusOK, cloudChatStream })
	s := &Server{}
	w := sendCloud(t, cloudRouter(t, s), "/api/chat", `{"model":"test-model:cloud","messages":[]}`, nil)
	if w.Code != http.StatusOK || w.Body.String() != cloudChatStream {
		t.Errorf("with stats off, cloud chats still work: status %d", w.Code)
	}
}

func TestCloudMeterSkipsLongLines(t *testing.T) {
	now := time.Now()
	m := &cloudMeter{u: newTestStats(t, &now), model: "m:cloud", info: statsRequestInfo{Cloud: true}}
	m.begin()
	m.scan([]byte(`{"message":{"content":"` + strings.Repeat("x", statsCloudMaxLine) + `"},"done":false}` + "\n"))
	m.scan([]byte(`{"done":true,"eval_count":3,"eval_duration":1000}`))

	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/api/chat", nil)
	m.finish(c, http.StatusOK)

	if got := findStats(t, m.u.snapshot().AllTimeModels, "m:cloud"); got.Requests != 1 || got.OutputTokens != 3 {
		t.Errorf("after a long line, the final line should still be read: %+v", got)
	}
}

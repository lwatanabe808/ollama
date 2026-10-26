package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

// skipUnlessPermissionsWork skips tests that rely on a file being unreadable,
// which doesn't hold on Windows or when running as root.
func skipUnlessPermissionsWork(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" || os.Getuid() == 0 {
		t.Skip("needs file permissions that the test user can't override")
	}
}

// TestStatsUnreadableFilesAreNotOverwritten checks that when the saved stats
// exist but can't be read, Ollama keeps counting in memory and never
// replaces the files with fresh, nearly empty ones.
func TestStatsUnreadableFilesAreNotOverwritten(t *testing.T) {
	skipUnlessPermissionsWork(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "stats.json")
	logPath := statsLogPath(path)
	totals := []byte(`{"version":1,"all_time":{"m":{"model":"m","requests":500}}}`)
	logLine := []byte(`{"at":"2026-10-20T10:00:00Z","model":"m","tag":"old"}` + "\n")
	for p, b := range map[string][]byte{path: totals, logPath: logLine} {
		if err := os.WriteFile(p, b, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(p, 0o000); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chmod(p, 0o600) })
	}

	u := newUsageStats(path)
	u.record("m", api.Metrics{EvalCount: 3}, statsRequestInfo{Tag: "new"}, false)
	if err := u.save(); err != nil {
		t.Fatalf("save: %v", err)
	}
	if got := findStats(t, u.snapshot().AllTimeModels, "m"); got.Requests != 1 {
		t.Errorf("stats should still count in memory, got %d requests", got.Requests)
	}

	for p, want := range map[string][]byte{path: totals, logPath: logLine} {
		if err := os.Chmod(p, 0o600); err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != string(want) {
			t.Errorf("%s was overwritten: %q", filepath.Base(p), got)
		}
	}
}

// TestStatsSuiteRunsUnreadableNotOverwritten checks that saving a test run
// never replaces saved runs that couldn't be read.
func TestStatsSuiteRunsUnreadableNotOverwritten(t *testing.T) {
	skipUnlessPermissionsWork(t)
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	path, err := statsSuiteRunsPath()
	if err != nil {
		t.Fatal(err)
	}
	old := []byte(`{"id":"run-old","suite":"basics","models":["m"],"sections":["fact_checks"],"results":[]}` + "\n")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, old, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(path, 0o600) })

	now := time.Now()
	s := &Server{stats: newTestStats(t, &now)}
	router := gin.New()
	router.GET("/api/stats/suite-runs", s.StatsSuiteRunsHandler)
	router.POST("/api/stats/suite-runs", s.SaveStatsSuiteRunHandler)

	req := httptest.NewRequest(http.MethodPost, "/api/stats/suite-runs", strings.NewReader(
		`{"id":"run-new","suite":"basics","tag":"basics:run-new","models":["m"],"sections":["fact_checks"],"results":[{"pass":true}]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != http.StatusInternalServerError {
		t.Errorf("save status = %d, want 500 when saved runs can't be read", w.Code)
	}
	w = httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/stats/suite-runs", nil))
	if w.Code != http.StatusInternalServerError {
		t.Errorf("list status = %d, want 500 when saved runs can't be read", w.Code)
	}

	if err := os.Chmod(path, 0o600); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != string(old) {
		t.Errorf("saved runs were overwritten: %q", got)
	}
}

// TestCloudMeterCountsFinishedAnswerAfterHangUp checks that an answer that
// finished is counted even if the caller hung up right after reading it.
func TestCloudMeterCountsFinishedAnswerAfterHangUp(t *testing.T) {
	gin.SetMode(gin.TestMode)
	now := time.Now()
	m := &cloudMeter{u: newTestStats(t, &now), model: "m:cloud", info: statsRequestInfo{Cloud: true}}
	m.begin()
	m.scan([]byte(`{"done":true,"total_duration":5000000,"eval_count":7}` + "\n"))

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/api/chat", nil).WithContext(ctx)
	m.finish(c, http.StatusOK)

	if got := findStats(t, m.u.snapshot().AllTimeModels, "m:cloud"); got.Requests != 1 || got.Errors != 0 || got.OutputTokens != 7 {
		t.Errorf("a finished answer should count after the caller hangs up: %+v", got)
	}
	if st := m.u.snapshot(); st.Active != 0 {
		t.Errorf("%d requests still marked running", st.Active)
	}
}

// TestStatsCloudUnreachableIsAnError checks that a cloud request that can't
// reach Ollama's servers at all is counted as an error.
func TestStatsCloudUnreachableIsAnError(t *testing.T) {
	gone := httptest.NewServer(http.NotFoundHandler())
	gone.Close() // nothing listens at this address now

	originalURL, originalSign := cloudProxyBaseURL, cloudProxySignRequest
	cloudProxyBaseURL = gone.URL
	cloudProxySignRequest = func(context.Context, *http.Request) error { return nil }
	t.Cleanup(func() { cloudProxyBaseURL, cloudProxySignRequest = originalURL, originalSign })

	now := time.Now()
	s := &Server{stats: newTestStats(t, &now)}
	w := sendCloud(t, cloudRouter(t, s), "/api/chat", `{"model":"test-model:cloud","messages":[{"role":"user","content":"hi"}]}`, nil)
	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", w.Code)
	}
	if got := findStats(t, s.stats.snapshot().AllTimeModels, "test-model:cloud"); got.Requests != 1 || got.Errors != 1 {
		t.Errorf("an unreachable cloud should count as one error: %+v", got)
	}
}

// TestStatsDashboardNeverInsertsHTML checks the dashboard only ever sets
// text, never HTML, so nothing in the stats can inject markup or scripts.
func TestStatsDashboardNeverInsertsHTML(t *testing.T) {
	unsafe := regexp.MustCompile(`innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function\(|\son[a-z]+=`)
	files := map[string][]byte{"stats_dashboard.html": statsDashboardHTML}
	for _, f := range statsDashboardFiles {
		bts, err := statsDashboardFS.ReadFile(f.path)
		if err != nil {
			t.Fatal(err)
		}
		files[f.path] = bts
	}
	for name, bts := range files {
		for i, line := range strings.Split(string(bts), "\n") {
			if m := unsafe.FindString(line); m != "" {
				t.Errorf("%s:%d uses %q: set text with textContent instead", name, i+1, strings.TrimSpace(m))
			}
		}
	}
}

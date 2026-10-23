package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
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

func countLines(t *testing.T, path string) int {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	n := 0
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if strings.TrimSpace(sc.Text()) != "" {
			n++
		}
	}
	return n
}

func TestStatsTag(t *testing.T) {
	gin.SetMode(gin.TestMode)
	newCtx := func(header string) (*gin.Context, *httptest.ResponseRecorder) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/chat", nil)
		if header != "" {
			c.Request.Header.Set(statsTagHeader, header)
		}
		return c, w
	}
	on := &Server{stats: &usageStats{}}

	for _, tt := range []struct {
		name, body, header, want string
		wantErr                  bool
	}{
		{name: "no tag"},
		{name: "body", body: "cloud-test", want: "cloud-test"},
		{name: "header", header: "from header", want: "from header"},
		{name: "body wins over header", body: "body", header: "header", want: "body"},
		{name: "trims spaces", body: "  run 1  ", want: "run 1"},
		{name: "allowed punctuation", body: "team/eval:v2.1_a@b", want: "team/eval:v2.1_a@b"},
		{name: "letters in other languages", body: "テスト", want: "テスト"},
		{name: "longest allowed", body: strings.Repeat("a", api.TagMaxLength), want: strings.Repeat("a", api.TagMaxLength)},
		{name: "too long", body: strings.Repeat("a", api.TagMaxLength+1), wantErr: true},
		{name: "html is rejected", body: "<b>x</b>", wantErr: true},
		{name: "newline is rejected", body: "a\nb", wantErr: true},
		// A bad header is ignored instead of failing the chat.
		{name: "bad header is ignored", header: `a"b`},
		{name: "bad header with a body tag", body: "ok", header: `a"b`, want: "ok"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, w := newCtx(tt.header)
			body := tt.body
			ok := on.takeStatsTag(c, &body)
			if body != "" {
				t.Errorf("body tag left on the request: %q", body)
			}
			if tt.wantErr {
				if ok || w.Code != http.StatusBadRequest {
					t.Fatalf("ok = %v, status = %d; want a 400", ok, w.Code)
				}
				return
			}
			if !ok {
				t.Fatalf("unexpected error: %s", w.Body.String())
			}
			if got := c.GetString(statsTagKey); got != tt.want {
				t.Errorf("tag = %q, want %q", got, tt.want)
			}
		})
	}

	t.Run("stats off ignores tags", func(t *testing.T) {
		c, _ := newCtx("header")
		body := "<bad>"
		if !(&Server{}).takeStatsTag(c, &body) {
			t.Fatal("a bad tag failed the request with stats off")
		}
		if body != "" || c.GetString(statsTagKey) != "" {
			t.Errorf("tag kept with stats off: body %q, context %q", body, c.GetString(statsTagKey))
		}
	})
}

func TestStatsLogQuery(t *testing.T) {
	now := time.Date(2026, 10, 22, 9, 0, 0, 0, time.Local)
	u := newTestStats(t, &now)

	u.record("gemma", api.Metrics{PromptEvalCount: 10, EvalCount: 100, EvalDuration: time.Second, TotalDuration: 2 * time.Second}, statsRequestInfo{Tag: "cloud-test", ContextLength: 4096, MemorySize: 1 << 30}, false)
	now = now.Add(time.Minute)
	u.record("llama", api.Metrics{PromptEvalCount: 20, EvalCount: 50, EvalDuration: time.Second, TotalDuration: 3 * time.Second}, statsRequestInfo{Tag: "cloud-test"}, false)
	now = now.Add(time.Minute)
	u.record("llama", api.Metrics{EvalCount: 7}, statsRequestInfo{}, false) // untagged
	now = now.Add(time.Minute)
	u.record("llama", api.Metrics{}, statsRequestInfo{Tag: "cloud-test"}, true)
	now = now.Add(time.Minute)
	u.record("gemma", api.Metrics{EvalCount: 1}, statsRequestInfo{Tag: "other"}, false)

	got := u.logQuery("cloud-test", 2)
	if got.Total != 3 {
		t.Errorf("total = %d, want 3 tagged requests", got.Total)
	}
	if len(got.Requests) != 2 {
		t.Fatalf("returned %d requests, want the limit of 2", len(got.Requests))
	}
	if !got.Requests[0].Error || got.Requests[1].Model != "llama" {
		t.Errorf("requests should be newest first, got %+v", got.Requests)
	}
	for _, e := range got.Requests {
		if e.Tag != "cloud-test" {
			t.Errorf("got a request tagged %q", e.Tag)
		}
	}

	// Totals cover every match, not just the ones returned.
	llama := findStats(t, got.Models, "llama")
	if llama.Requests != 2 || llama.Errors != 1 || llama.OutputTokens != 50 {
		t.Errorf("llama totals = %+v, want 2 requests, 1 error, 50 tokens", llama)
	}
	gemma := findStats(t, got.Models, "gemma")
	if gemma.TokensPerSecond() != 100 || gemma.AvgWait() != time.Second {
		t.Errorf("gemma speed %v, wait %v, want 100 tok/s and 1s", gemma.TokensPerSecond(), gemma.AvgWait())
	}
	if gemma.ContextLength != 4096 || gemma.MemorySize != 1<<30 {
		t.Errorf("gemma model details = context %d, memory %d; want them filled in from the totals", gemma.ContextLength, gemma.MemorySize)
	}
	if gemma.Last == nil || gemma.Last.OutputTokens != 100 {
		t.Errorf("gemma last request = %+v", gemma.Last)
	}

	// No tag means every request.
	if all := u.logQuery("", 100); all.Total != 5 {
		t.Errorf("all requests = %d, want 5", all.Total)
	}

	// The tag list is newest first, with counts.
	tags := u.snapshot().Tags
	if len(tags) != 2 || tags[0].Tag != "other" || tags[1].Tag != "cloud-test" {
		t.Fatalf("tags = %+v, want other then cloud-test", tags)
	}
	if tags[1].Requests != 3 || tags[1].Errors != 1 {
		t.Errorf("cloud-test summary = %+v, want 3 requests, 1 error", tags[1])
	}
}

func TestStatsLogSurvivesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stats.json")
	logPath := statsLogPath(path)
	if filepath.Base(logPath) != "stats-log.jsonl" {
		t.Fatalf("log path = %q, want stats-log.jsonl next to stats.json", logPath)
	}

	first := newUsageStats(path)
	first.record("m", api.Metrics{EvalCount: 3}, statsRequestInfo{Tag: "a"}, false)
	first.record("m", api.Metrics{EvalCount: 4}, statsRequestInfo{Tag: "b"}, false)
	if err := first.close(); err != nil {
		t.Fatal(err)
	}
	if n := countLines(t, logPath); n != 2 {
		t.Fatalf("log has %d lines, want 2", n)
	}
	info, err := os.Stat(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("log permissions = %o, want 600", perm)
	}

	// A new server reads the log and appends to it.
	second := newUsageStats(path)
	if got := second.logQuery("a", 10); got.Total != 1 || got.Requests[0].OutputTokens != 3 {
		t.Errorf("after restart, tag a = %+v", got)
	}
	second.record("m", api.Metrics{EvalCount: 5}, statsRequestInfo{Tag: "a"}, false)
	if err := second.save(); err != nil {
		t.Fatal(err)
	}
	if n := countLines(t, logPath); n != 3 {
		t.Errorf("log has %d lines after appending, want 3", n)
	}

	// Nothing new means nothing written.
	before, _ := os.ReadFile(logPath)
	if err := second.save(); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(logPath)
	if !bytes.Equal(before, after) {
		t.Error("saving with no new requests should not change the log")
	}
}

func TestStatsLogBadLines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stats.json")
	logPath := statsLogPath(path)
	good, _ := json.Marshal(api.RequestLogEntry{Model: "m", Tag: "a", RequestStats: api.RequestStats{At: time.Now()}})
	content := string(good) + "\n{not json\n\n" + string(good) + "\n"
	if err := os.WriteFile(logPath, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	u := newUsageStats(path)
	if got := u.logQuery("a", 10); got.Total != 2 {
		t.Errorf("read %d good lines, want 2 (bad line skipped)", got.Total)
	}
	// The next save cleans the bad line out of the file.
	if err := u.save(); err != nil {
		t.Fatal(err)
	}
	if n := countLines(t, logPath); n != 2 {
		t.Errorf("log has %d lines after cleanup, want 2", n)
	}
}

func TestStatsLogTrim(t *testing.T) {
	now := time.Date(2026, 10, 22, 9, 0, 0, 0, time.Local)
	u := newTestStats(t, &now)
	u.record("m", api.Metrics{}, statsRequestInfo{Tag: "old"}, false)

	now = now.AddDate(0, 0, statsLogKeepDays+1)
	u.record("m", api.Metrics{}, statsRequestInfo{Tag: "new"}, false)
	if got := u.logQuery("old", 10); got.Total != 0 {
		t.Errorf("requests older than %d days should be dropped", statsLogKeepDays)
	}

	// The log never holds more than the cap.
	l := statsLog{}
	for i := range statsLogMaxEntries + 10 {
		l.add(api.RequestLogEntry{Model: "m", RequestStats: api.RequestStats{At: now.Add(time.Duration(i) * time.Millisecond)}}, now.Add(time.Hour))
	}
	if len(l.entries) != statsLogMaxEntries {
		t.Errorf("log holds %d entries, want the cap of %d", len(l.entries), statsLogMaxEntries)
	}
	if !l.entries[0].At.Equal(now.Add(10 * time.Millisecond)) {
		t.Error("the oldest entries should be the ones dropped")
	}
}

func TestStatsLogReset(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stats.json")
	u := newUsageStats(path)
	u.record("m", api.Metrics{}, statsRequestInfo{Tag: "a"}, false)
	if err := u.save(); err != nil {
		t.Fatal(err)
	}
	if err := u.reset(); err != nil {
		t.Fatal(err)
	}
	if n := countLines(t, statsLogPath(path)); n != 0 {
		t.Errorf("log has %d lines after reset, want 0", n)
	}
	if tags := u.snapshot().Tags; len(tags) != 0 {
		t.Errorf("tags after reset = %+v", tags)
	}
}

func TestStatsLogHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	now := time.Now()
	s := &Server{stats: newTestStats(t, &now)}
	for range 3 {
		s.stats.record("m", api.Metrics{EvalCount: 1}, statsRequestInfo{Tag: "run 1"}, false)
	}
	router := gin.New()
	router.GET("/api/stats/log", s.StatsLogHandler)

	get := func(url string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, url, nil))
		return w
	}

	w := get("/api/stats/log?tag=run+1&limit=2")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	var resp api.StatsLogResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Tag != "run 1" || resp.Total != 3 || len(resp.Requests) != 2 {
		t.Errorf("response = tag %q, total %d, %d requests; want run 1, 3, 2", resp.Tag, resp.Total, len(resp.Requests))
	}

	for _, bad := range []string{"?limit=0", "?limit=abc", "?tag=%3Cscript%3E"} {
		if w := get("/api/stats/log" + bad); w.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400", bad, w.Code)
		}
	}
}

// TestStatsTagsFromRealRequests tags requests through the real handlers,
// both with the "tag" field and the X-Ollama-Tag header.
func TestStatsTagsFromRealRequests(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		CompletionFn: func(_ context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			fn(llm.CompletionResponse{Content: "hi", Done: true, DoneReason: llm.DoneReasonStop, EvalCount: 2, EvalDuration: time.Second})
			return nil
		},
	}
	s := newServerWithMockRunner(t, &mock)
	now := time.Now()
	s.stats = newTestStats(t, &now)
	createMinimalGGUFModel(t, s, "tag-test", gguftest.KV{}, "{{ .Prompt }}", nil)

	noStream := false
	w := createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "tag-test", Prompt: "hi", Stream: &noStream, Tag: "from-body"})
	if w.Code != http.StatusOK {
		t.Fatalf("generate status = %d: %s", w.Code, w.Body.String())
	}

	// Send a chat with the header, the way an OpenAI-style app would.
	body, _ := json.Marshal(api.ChatRequest{Model: "tag-test", Messages: []api.Message{{Role: "user", Content: "hi"}}, Stream: &noStream})
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/chat", io.NopCloser(bytes.NewReader(body)))
	c.Request.Header.Set(statsTagHeader, "from-header")
	s.ChatHandler(c)
	if rec.Code != http.StatusOK {
		t.Fatalf("chat status = %d: %s", rec.Code, rec.Body.String())
	}

	for _, tag := range []string{"from-body", "from-header"} {
		got := s.stats.logQuery(tag, 10)
		if got.Total != 1 || got.Requests[0].Model != "tag-test:latest" || got.Requests[0].OutputTokens != 2 {
			t.Errorf("tag %q: %+v", tag, got)
		}
	}

	// A bad tag is rejected before the model runs.
	before := s.stats.logQuery("", 100).Total
	w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "tag-test", Prompt: "hi", Stream: &noStream, Tag: "<bad>"})
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "tag") {
		t.Errorf("bad tag: status %d, body %s", w.Code, w.Body.String())
	}
	if after := s.stats.logQuery("", 100).Total; after != before {
		t.Error("a rejected request should not be logged")
	}
}

func TestStatsTagsListCapped(t *testing.T) {
	now := time.Now()
	u := newTestStats(t, &now)
	for i := range statsTagsListed + 5 {
		now = now.Add(time.Second)
		u.record("m", api.Metrics{}, statsRequestInfo{Tag: fmt.Sprintf("tag-%d", i)}, false)
	}
	st := u.snapshot()
	if len(st.Tags) != statsTagsListed || st.TagsTotal != statsTagsListed+5 {
		t.Errorf("listed %d of %d tags, want %d of %d", len(st.Tags), st.TagsTotal, statsTagsListed, statsTagsListed+5)
	}
	if st.Tags[0].Tag != fmt.Sprintf("tag-%d", statsTagsListed+4) {
		t.Errorf("first tag = %q, want the most recent", st.Tags[0].Tag)
	}
	// A tag past the cap can still be looked up by name.
	if got := u.logQuery("tag-0", 10); got.Total != 1 {
		t.Errorf("oldest tag lookup = %d requests, want 1", got.Total)
	}
}

// TestStatsContextFillUsesEachRequestsWindow covers a context test run in a
// small window, followed by normal use in the usual window.
func TestStatsContextFillUsesEachRequestsWindow(t *testing.T) {
	now := time.Now()
	u := newTestStats(t, &now)

	// A full request in a 2K test window.
	u.record("m", api.Metrics{PromptEvalCount: 2000, EvalCount: 20}, statsRequestInfo{ContextLength: 2048, MemorySize: 2 << 30, Tag: "ctx"}, false)
	// Then normal use in a 32K window: more tokens, but a much smaller share.
	now = now.Add(time.Minute)
	u.record("m", api.Metrics{PromptEvalCount: 3000, EvalCount: 100}, statsRequestInfo{ContextLength: 32768, MemorySize: 6 << 30}, false)

	all := findStats(t, u.snapshot().AllTimeModels, "m")
	if used, window := all.MaxContextFill(); used != 2020 || window != 2048 {
		t.Errorf("all time fullest = %d of %d, want 2020 of 2048 (the 2K request was fuller)", used, window)
	}
	if all.ContextLength != 32768 || all.MemorySize != 6<<30 {
		t.Errorf("current window and memory = %d, %d; want the latest 32K setup", all.ContextLength, all.MemorySize)
	}

	// The tag view uses the window and memory its own requests ran with.
	tagged := findStats(t, u.logQuery("ctx", 10).Models, "m")
	if used, window := tagged.MaxContextFill(); used != 2020 || window != 2048 {
		t.Errorf("tag fullest = %d of %d, want 2020 of 2048", used, window)
	}
	if tagged.ContextLength != 2048 || tagged.MemorySize != 2<<30 {
		t.Errorf("tag window and memory = %d, %d; want the 2K test setup", tagged.ContextLength, tagged.MemorySize)
	}
}

// TestStatsTestRequestsAreNotUsage covers requests marked as tests: they're
// logged and can be looked up by tag, but don't count as usage.
func TestStatsTestRequestsAreNotUsage(t *testing.T) {
	now := time.Now()
	u := newTestStats(t, &now)

	u.record("m", api.Metrics{EvalCount: 10}, statsRequestInfo{Tag: "real"}, false)
	u.record("m", api.Metrics{EvalCount: 99}, statsRequestInfo{Tag: "basics:run-1", Test: true}, false)
	u.record("m", api.Metrics{EvalCount: 99}, statsRequestInfo{Test: true}, false) // untagged helper

	snap := u.snapshot()
	if got := findStats(t, snap.AllTimeModels, "m"); got.Requests != 1 || got.OutputTokens != 10 {
		t.Errorf("usage = %d requests, %d tokens; test requests should be left out", got.Requests, got.OutputTokens)
	}
	if len(snap.Days) != 1 || snap.Days[0].Requests != 1 {
		t.Errorf("daily totals = %+v, should only count the real request", snap.Days)
	}
	if len(snap.Tags) != 1 || snap.Tags[0].Tag != "real" {
		t.Errorf("tags = %+v, test runs shouldn't fill the tag list", snap.Tags)
	}

	// The test run can still be looked up by its tag.
	run := u.logQuery("basics:run-1", 10)
	if run.Total != 1 || !run.Requests[0].Test || run.Requests[0].OutputTokens != 99 {
		t.Errorf("test run lookup = %+v", run)
	}
}

func TestStatsIsTest(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for header, want := range map[string]bool{"test": true, " TEST ": true, "": false, "usage": false} {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/api/chat", nil)
		if header != "" {
			c.Request.Header.Set(statsKindHeader, header)
		}
		if got := statsIsTest(c); got != want {
			t.Errorf("header %q: statsIsTest = %v, want %v", header, got, want)
		}
	}
}

func TestStatsOffSwitch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	s := &Server{} // what Serve builds when OLLAMA_NOSTATS is set
	router := gin.New()
	router.GET("/api/stats", s.StatsHandler)
	router.POST("/api/stats/suite-runs", s.SaveStatsSuiteRunHandler)

	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/stats", nil))
	var st api.StatsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &st); err != nil {
		t.Fatal(err)
	}
	if !st.Disabled {
		t.Error("stats response should say stats are turned off")
	}

	req := httptest.NewRequest(http.MethodPost, "/api/stats/suite-runs", strings.NewReader(`{"id":"r","suite":"basics","models":["m"],"results":[]}`))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != http.StatusConflict {
		t.Errorf("saving a test run with stats off: status = %d, want 409", w.Code)
	}
}

// recountTags builds the tag list the slow way, from every entry.
func recountTags(l *statsLog) map[string]api.TagStats {
	out := map[string]api.TagStats{}
	for _, e := range l.entries {
		if e.Tag == "" || e.Test {
			continue
		}
		t := out[e.Tag]
		t.Tag = e.Tag
		t.Requests++
		if e.Error {
			t.Errors++
		}
		if e.At.After(t.LastUsed) {
			t.LastUsed = e.At
		}
		out[e.Tag] = t
	}
	return out
}

func checkTagsMatch(t *testing.T, step string, l *statsLog) {
	t.Helper()
	want := recountTags(l)
	got := l.tags()
	if len(got) != len(want) {
		t.Fatalf("%s: %d tags listed, recount has %d", step, len(got), len(want))
	}
	for _, g := range got {
		w := want[g.Tag]
		if g.Requests != w.Requests || g.Errors != w.Errors || !g.LastUsed.Equal(w.LastUsed) {
			t.Fatalf("%s: tag %q = %+v, recount = %+v", step, g.Tag, g, w)
		}
	}
}

// TestStatsTagListStaysInSync checks the tag list kept up to date as entries
// are added and dropped always matches a full recount.
func TestStatsTagListStaysInSync(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stats.json")
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	l := statsLog{path: statsLogPath(path)}
	add := func(tag string, failed, test bool) {
		now = now.Add(time.Hour)
		l.add(api.RequestLogEntry{Model: "m", Tag: tag, Error: failed, Test: test, RequestStats: api.RequestStats{At: now}}, now)
	}

	for i := range 200 {
		add(fmt.Sprintf("tag-%d", i%7), i%5 == 0, i%11 == 0)
		add("", false, false)
	}
	checkTagsMatch(t, "after adding", &l)

	// Jump ahead so the oldest entries age out.
	now = now.AddDate(0, 0, statsLogKeepDays-10)
	add("late", false, false)
	checkTagsMatch(t, "after old entries aged out", &l)

	// Save, reload, and compare again.
	u := &usageStats{log: l, now: func() time.Time { return now }}
	if err := u.saveLog(); err != nil {
		t.Fatal(err)
	}
	reloaded := loadStatsLog(statsLogPath(path), now)
	checkTagsMatch(t, "after reload", &reloaded)

	l.clear()
	checkTagsMatch(t, "after reset", &l)
	add("fresh", false, false)
	checkTagsMatch(t, "after reset and a new request", &l)
}

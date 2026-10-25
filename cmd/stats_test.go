package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/types/model"
)

func TestRenderStats(t *testing.T) {
	llama := api.ModelStats{
		Model:              "llama3.2:latest",
		Requests:           12,
		Errors:             1,
		PromptTokens:       1500,
		OutputTokens:       4800,
		PromptCachedTokens: 1000,
		OutputDuration:     100 * time.Second,
		TotalDuration:      122 * time.Second,
		LastUsed:           time.Now().Add(-3 * time.Minute),
		MaxContextUsed:     30000,
		ContextLength:      32768,
		MemorySize:         2_800_000_000,
		Last: &api.RequestStats{
			At:             time.Now().Add(-3 * time.Minute),
			PromptTokens:   200,
			OutputTokens:   600,
			OutputDuration: 10 * time.Second,
			TotalDuration:  10500 * time.Millisecond,
		},
	}
	phi := api.ModelStats{Model: "phi3:latest", Requests: 2, OutputTokens: 10, OutputDuration: time.Second, TotalDuration: 2 * time.Second}

	st := &api.StatsResponse{
		Since:         time.Date(2026, 10, 3, 9, 0, 0, 0, time.Local),
		Today:         "2026-10-22",
		TodayModels:   []api.ModelStats{llama},
		AllTimeModels: []api.ModelStats{llama, phi},
		Active:        1,
	}
	var buf bytes.Buffer
	renderStats(&buf, st)
	out := buf.String()

	for _, want := range []string{
		"tracking since Oct 3, 2026",
		"1 request running now",
		"TODAY",
		"ALL TIME",
		"MODEL", "AVG FIRST WORD", "WRITING SPEED", "TIME TO FINISH", "TOTAL TOKENS IN", "NEWLY READ", "REUSED", "TOTAL TOKENS OUT", "MAX CONTEXT", "MEMORY NEEDED", "LAST USED",
		"llama3.2:latest",
		"48.0 tok/s",      // 4800 tokens / 100s
		"11.1s",           // 122s over 11 successful requests
		"2.0s",            // (122s total - 100s writing) / 11 requests
		"30K / 32K (92%)", // max context used of the window
		"Note: llama3.2:latest used 30K of its 32K context window.",
		"LAST REQUEST (compared to all-time average)",
		"0.5s (avg 2.0s)",           // last wait vs average wait
		"60.0 tok/s (avg 48.0)",     // last speed vs average speed
		"10.5s (avg 11.1s)",         // last total time vs average
		"200 / 600 (avg 136 / 436)", // last tokens vs average per request
		"800 (max 30K)",             // last context used vs the max
		"2.8 GB",                    // saved memory size, shown even when not loaded
		"3 minutes ago",
		"Total: 14 requests, 1 errors, 5K tokens written.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n%s", want, out)
		}
	}

	// 1,500 in = 500 newly read + 1,000 reused, then 4,800 out.
	if !regexp.MustCompile(`llama3\.2:latest .*2K\s+500\s+1K\s+5K`).MatchString(out) {
		t.Errorf("tokens should split into total, newly read, reused, out:\n%s", out)
	}

	// phi3 has no saved memory size, so it shows as "-".
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "phi3:latest") && !strings.Contains(line, "    -    ") {
			t.Errorf("unloaded model should show '-' for memory: %q", line)
		}
	}
}

func TestRenderStatsEmpty(t *testing.T) {
	var buf bytes.Buffer
	renderStats(&buf, &api.StatsResponse{})
	out := buf.String()

	if !strings.Contains(out, "No requests yet today.") {
		t.Errorf("missing empty today message:\n%s", out)
	}
	if !strings.Contains(out, "Chat with a model and its stats will show up here.") {
		t.Errorf("missing empty all-time message:\n%s", out)
	}
}

func TestStatsDuration(t *testing.T) {
	for d, want := range map[time.Duration]string{
		0: "-", 40 * time.Millisecond: "40ms",

		400 * time.Millisecond:   "0.4s",
		12500 * time.Millisecond: "12.5s",
		75 * time.Second:         "1m 15s",
	} {
		if got := statsDuration(d); got != want {
			t.Errorf("statsDuration(%v) = %q, want %q", d, got, want)
		}
	}
}

func TestStatsContext(t *testing.T) {
	for _, tt := range []struct {
		m    api.ModelStats
		want string
	}{
		{api.ModelStats{}, "-"},
		{api.ModelStats{MaxContextUsed: 500}, "500"},
		{api.ModelStats{MaxContextUsed: 2048, ContextLength: 4096}, "2K / 4K (50%)"},
		{api.ModelStats{MaxContextUsed: 1200, ContextLength: 131072}, "1K / 128K (1%)"},
	} {
		if got := statsContext(tt.m); got != tt.want {
			t.Errorf("statsContext(%+v) = %q, want %q", tt.m, got, tt.want)
		}
	}
}

func TestRenderStatsNoContextWarning(t *testing.T) {
	m := api.ModelStats{Model: "small", Requests: 1, MaxContextUsed: 100, ContextLength: 4096}
	var buf bytes.Buffer
	renderStats(&buf, &api.StatsResponse{AllTimeModels: []api.ModelStats{m}})
	if strings.Contains(buf.String(), "Note:") {
		t.Errorf("no warning expected for a mostly empty context window:\n%s", buf.String())
	}
}

func TestRenderStatsNoLastRequest(t *testing.T) {
	m := api.ModelStats{Model: "old", Requests: 1}
	var buf bytes.Buffer
	renderStats(&buf, &api.StatsResponse{AllTimeModels: []api.ModelStats{m}})
	if strings.Contains(buf.String(), "LAST REQUEST") {
		t.Errorf("no LAST REQUEST section expected without saved data:\n%s", buf.String())
	}
}

// runStatsCommand runs `ollama stats` with the given flags against a fake server.
func runStatsCommand(t *testing.T, handler http.HandlerFunc, args ...string) error {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)

	cmd := NewCLI()
	statsCmd, _, err := cmd.Find([]string{"stats"})
	if err != nil {
		t.Fatal(err)
	}
	statsCmd.PreRunE = nil
	cmd.SetArgs(append([]string{"stats"}, args...))

	done := make(chan error, 1)
	go func() { done <- cmd.Execute() }()
	select {
	case err := <-done:
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("ollama stats did not return; it should print once and exit")
		return nil
	}
}

func TestStatsCommandPrintsOnce(t *testing.T) {
	calls := 0
	err := runStatsCommand(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/stats" {
			http.NotFound(w, r)
			return
		}
		calls++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(api.StatsResponse{Today: "2026-10-22"})
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Errorf("asked the server %d times, want once", calls)
	}
}

func TestStatsCommandOldServer(t *testing.T) {
	err := runStatsCommand(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"404 page not found"}`, http.StatusNotFound)
	})
	if err == nil || !strings.Contains(err.Error(), "does not support usage stats") {
		t.Errorf("err = %v, want a clear message about an old server", err)
	}
}

func TestStatsCommandTag(t *testing.T) {
	var gotQuery string
	err := runStatsCommand(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/stats/log" {
			http.NotFound(w, r)
			return
		}
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(api.StatsLogResponse{Tag: "cloud test"})
	}, "--tag", "cloud test", "--limit", "5")
	if err != nil {
		t.Fatal(err)
	}
	if gotQuery != "limit=5&tag=cloud+test" {
		t.Errorf("query = %q, want limit=5&tag=cloud+test", gotQuery)
	}
}

func TestRenderTagLog(t *testing.T) {
	at := time.Date(2026, 10, 22, 15, 4, 5, 0, time.Local)
	log := &api.StatsLogResponse{
		Tag:   "cloud-test",
		Total: 3,
		Models: []api.ModelStats{
			{Model: "gemma3:270m", Requests: 3, Errors: 1, OutputTokens: 200, OutputDuration: time.Second, TotalDuration: 3 * time.Second},
		},
		Requests: []api.RequestLogEntry{
			{Model: "gemma3:270m", Tag: "cloud-test", Error: true, RequestStats: api.RequestStats{At: at}},
			{Model: "gemma3:270m", Tag: "cloud-test", RequestStats: api.RequestStats{
				At: at.Add(-time.Minute), PromptTokens: 18, PromptCachedTokens: 10, OutputTokens: 100,
				OutputDuration: 500 * time.Millisecond, TotalDuration: 1500 * time.Millisecond,
			}},
		},
	}

	var buf bytes.Buffer
	renderTagLog(&buf, log)
	out := buf.String()
	for _, want := range []string{
		`Requests tagged "cloud-test": 3`,
		"BY MODEL",
		"REQUESTS (newest first)",
		"Oct 22 15:04:05",
		"error",
		"200.0 tok/s", // 100 tokens in 0.5s
		"1.0s",        // 1.5s total - 0.5s writing
		"TOTAL IN", "NEWLY READ", "REUSED", "OUT",
		"Showing the newest 2 of 3. Use --limit to see more.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n%s", want, out)
		}
	}
}

func TestRenderTagLogTokenSplit(t *testing.T) {
	log := &api.StatsLogResponse{Tag: "x", Total: 1, Requests: []api.RequestLogEntry{{
		Model: "m", RequestStats: api.RequestStats{PromptTokens: 18, PromptCachedTokens: 10, OutputTokens: 100},
	}}}
	var buf bytes.Buffer
	renderTagLog(&buf, log)
	// 18 in = 8 newly read + 10 reused, then 100 out.
	if !regexp.MustCompile(`\s18\s+8\s+10\s+100\s`).MatchString(buf.String()) {
		t.Errorf("request row should show 18 in, 8 newly read, 10 reused, 100 out:\n%s", buf.String())
	}
}

func TestRenderTagLogEmpty(t *testing.T) {
	var buf bytes.Buffer
	renderTagLog(&buf, &api.StatsLogResponse{Tag: "my run"})
	if !strings.Contains(buf.String(), `--tag "my run"`) {
		t.Errorf("empty result should show how to tag, with quotes for spaces:\n%s", buf.String())
	}
}

func TestRenderStatsTags(t *testing.T) {
	st := &api.StatsResponse{
		AllTimeModels: []api.ModelStats{{Model: "m", Requests: 4}},
		Tags:          []api.TagStats{{Tag: "cloud-test", Requests: 4, LastUsed: time.Now().Add(-2 * time.Minute)}},
	}
	var buf bytes.Buffer
	renderStats(&buf, st)
	for _, want := range []string{"TAGS", "cloud-test", "2 minutes ago", "ollama stats --tag NAME"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("output missing %q\n%s", want, buf.String())
		}
	}
}

func TestRunTagFlag(t *testing.T) {
	runCmd, _, err := NewCLI().Find([]string{"run"})
	if err != nil {
		t.Fatal(err)
	}
	if runCmd.Flags().Lookup("tag") == nil {
		t.Fatal("ollama run should have a --tag option")
	}
	if got := (runOptions{Tag: "a"}).Copy().Tag; got != "a" {
		t.Errorf("Copy dropped the tag, got %q", got)
	}
}

func TestStatsProcessor(t *testing.T) {
	v := func(n int64) *int64 { return &n }
	for _, tt := range []struct {
		m    api.ModelStats
		want string
	}{
		{api.ModelStats{}, "-"},
		{api.ModelStats{MemorySize: 1000}, "-"},
		{api.ModelStats{MemorySize: 1000, MemoryVRAM: v(1000)}, "100% GPU"},
		{api.ModelStats{MemorySize: 1000, MemoryVRAM: v(0)}, "100% CPU"},
		{api.ModelStats{MemorySize: 1000, MemoryVRAM: v(520)}, "48%/52% CPU/GPU"},
		{api.ModelStats{Cloud: true}, "Cloud"},
	} {
		if got := statsProcessor(tt.m); got != tt.want {
			t.Errorf("statsProcessor(%+v) = %q, want %q", tt.m, got, tt.want)
		}
	}
}

func TestRenderTimeBreakdown(t *testing.T) {
	vram := int64(500)
	models := []api.ModelStats{{
		Model: "big", Requests: 2,
		LoadDuration: 12 * time.Second, PromptDuration: 400 * time.Millisecond, OutputDuration: 2 * time.Second,
		PromptTokens: 100, PromptCachedTokens: 80,
		MemorySize: 1000, MemoryVRAM: &vram,
	}}
	var buf bytes.Buffer
	renderTimeBreakdown(&buf, models)
	out := buf.String()
	for _, want := range []string{
		"WHERE THE TIME GOES (average per request)",
		"LOADING MODEL", "READING PROMPT", "WRITING ANSWER", "PROMPT REUSED", "PROCESSOR",
		"6.0s", "0.2s", "1.0s", "80%", "50%/50% CPU/GPU",
		"Note: big is running 50% on the CPU",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n%s", want, out)
		}
	}
}

func TestRenderTagsShowsTenAndCount(t *testing.T) {
	var tags []api.TagStats
	for i := range 15 {
		tags = append(tags, api.TagStats{Tag: fmt.Sprintf("tag-%02d", i), Requests: 1})
	}
	var buf bytes.Buffer
	renderTags(&buf, tags, 40)
	out := buf.String()
	if !strings.Contains(out, "tag-09") || strings.Contains(out, "tag-10") {
		t.Errorf("should list exactly the first 10 tags:\n%s", out)
	}
	if !strings.Contains(out, "...and 30 more.") {
		t.Errorf("should say how many more tags exist:\n%s", out)
	}
}

func TestRenderTagLogContextWarning(t *testing.T) {
	log := &api.StatsLogResponse{Tag: "ctx", Total: 1, Models: []api.ModelStats{
		{Model: "m", Requests: 1, MaxContextUsed: 2020, MaxContextWindow: 2048, ContextLength: 32768},
	}}
	var buf bytes.Buffer
	renderTagLog(&buf, log)
	out := buf.String()
	if !strings.Contains(out, "2K / 2K (99%)") {
		t.Errorf("max context should use the window the request ran in:\n%s", out)
	}
	if !strings.Contains(out, "Note: m used 2K of its 2K context window") {
		t.Errorf("tag view should warn about a nearly full window:\n%s", out)
	}
}

func TestStatsCommandWhenOff(t *testing.T) {
	err := runStatsCommand(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(api.StatsResponse{Disabled: true})
	})
	if err != nil {
		t.Fatalf("stats off should print a note, not fail: %v", err)
	}
}

// statsServer is a fake Ollama server that answers /api/stats with st and
// records the requests it gets.
func statsServer(st api.StatsResponse, got *[]string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		*got = append(*got, r.Method+" "+r.URL.Path)
		if r.URL.Path != "/api/stats" {
			http.NotFound(w, r)
			return
		}
		if r.Method == http.MethodDelete && st.Disabled {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"error":"usage stats are turned off on this server (OLLAMA_NOSTATS)"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(st)
	}
}

func TestStatsWebOpensDashboard(t *testing.T) {
	var opened string
	openBrowser = func(url string) { opened = url }
	t.Cleanup(func() { openBrowser = launch.OpenBrowser })

	var got []string
	if err := runStatsCommand(t, statsServer(api.StatsResponse{}, &got), "--web"); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(opened, "/stats") || !strings.HasPrefix(opened, "http://127.0.0.1:") {
		t.Errorf("opened %q, want this server's /stats page", opened)
	}
	// It checks the server has stats before opening the page.
	if len(got) != 1 || got[0] != "GET /api/stats" {
		t.Errorf("requests = %v, want one stats check", got)
	}
}

func TestStatsWebWhenOff(t *testing.T) {
	opened := false
	openBrowser = func(string) { opened = true }
	t.Cleanup(func() { openBrowser = launch.OpenBrowser })

	var got []string
	if err := runStatsCommand(t, statsServer(api.StatsResponse{Disabled: true}, &got), "--web"); err != nil {
		t.Fatal(err)
	}
	if opened {
		t.Error("opened the dashboard even though stats are off")
	}
}

func TestDashboardURL(t *testing.T) {
	for host, want := range map[string]string{
		"http://127.0.0.1:11434":  "http://127.0.0.1:11434/stats",
		"http://0.0.0.0:11434":    "http://127.0.0.1:11434/stats",
		"http://[::]:8080":        "http://127.0.0.1:8080/stats",
		"https://ollama.lan:443":  "https://ollama.lan:443/stats",
		"http://example.com/base": "http://example.com/base/stats",
	} {
		u, err := url.Parse(host)
		if err != nil {
			t.Fatal(err)
		}
		if got := dashboardURL(u); got != want {
			t.Errorf("dashboardURL(%s) = %s, want %s", host, got, want)
		}
	}
}

func TestStatsReset(t *testing.T) {
	var got []string
	if err := runStatsCommand(t, statsServer(api.StatsResponse{}, &got), "--reset"); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != "DELETE /api/stats" {
		t.Errorf("requests = %v, want one DELETE /api/stats", got)
	}
}

func TestStatsResetWhenOff(t *testing.T) {
	var got []string
	err := runStatsCommand(t, statsServer(api.StatsResponse{Disabled: true}, &got), "--reset")
	if err == nil || !strings.Contains(err.Error(), "turned off") {
		t.Errorf("err = %v, want a note that stats are off", err)
	}
}

func TestStatsFlagErrors(t *testing.T) {
	for _, tt := range []struct {
		args []string
		want string
	}{
		{[]string{"--reset", "--web"}, "none of the others"},
		{[]string{"--web", "--tag", "x"}, "none of the others"},
		{[]string{"--limit", "5"}, "only works with --tag"},
		{[]string{"--tag", "  "}, "needs a tag name"},
		{[]string{"--tag", "<b>"}, "isn't allowed"},
		{[]string{"--tag", "x", "--limit", "0"}, "at least 1"},
	} {
		t.Run(strings.Join(tt.args, " "), func(t *testing.T) {
			var got []string
			err := runStatsCommand(t, statsServer(api.StatsResponse{}, &got), tt.args...)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want it to mention %q", err, tt.want)
			}
			if len(got) != 0 {
				t.Errorf("bad flags still reached the server: %v", got)
			}
		})
	}
}

func TestStatsTagWhenOff(t *testing.T) {
	err := runStatsCommand(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(api.StatsLogResponse{Disabled: true})
	}, "--tag", "x")
	if err != nil {
		t.Fatalf("stats off should print a note, not fail: %v", err)
	}
}

// TestRunSendsTag checks that 'ollama run --tag' puts the tag on the request.
func TestRunSendsTag(t *testing.T) {
	reqs := make(chan api.GenerateRequest, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilityCompletion}, ModelInfo: map[string]any{}})
		case "/api/generate":
			var req api.GenerateRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			reqs <- req
			_ = json.NewEncoder(w).Encode(api.GenerateResponse{Model: "m", Done: true})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)

	cmd := NewCLI()
	runCmd, _, err := cmd.Find([]string{"run"})
	if err != nil {
		t.Fatal(err)
	}
	runCmd.PreRunE = nil
	cmd.SetArgs([]string{"run", "m", "hello", "--tag", " cloud test "})
	captureStdout(t, func() { err = cmd.Execute() })
	if err != nil {
		t.Fatal(err)
	}
	select {
	case req := <-reqs:
		if req.Tag != "cloud test" {
			t.Errorf("tag = %q, want %q", req.Tag, "cloud test")
		}
	default:
		t.Fatal("no generate request was sent")
	}
}

func TestRunRejectsBadTag(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; http.NotFound(w, r) }))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)

	cmd := NewCLI()
	runCmd, _, _ := cmd.Find([]string{"run"})
	runCmd.PreRunE = nil
	cmd.SetArgs([]string{"run", "m", "hello", "--tag", "<b>"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--tag") {
		t.Errorf("err = %v, want a --tag error", err)
	}
	if calls != 0 {
		t.Errorf("a bad tag still reached the server %d times", calls)
	}
}

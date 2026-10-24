package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
)

// TestStatsSaveLoop checks the background save loop writes new stats on its
// own, and that close stops it, saves, and is safe to call twice.
func TestStatsSaveLoop(t *testing.T) {
	now := time.Now()
	u := newTestStats(t, &now)
	done := make(chan struct{})
	go func() { u.runEvery(10 * time.Millisecond); close(done) }()

	u.record("m", api.Metrics{EvalCount: 5}, statsRequestInfo{Tag: "loop"}, false)
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(u.path); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the save loop never wrote the stats file")
		}
		time.Sleep(10 * time.Millisecond)
	}

	u.record("m", api.Metrics{EvalCount: 7}, statsRequestInfo{}, false)
	if err := u.close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("close didn't stop the save loop")
	}
	if err := u.close(); err != nil {
		t.Fatalf("second close: %v", err)
	}

	again := newUsageStats(u.path)
	if got := findStats(t, again.snapshot().AllTimeModels, "m"); got.Requests != 2 || got.OutputTokens != 12 {
		t.Errorf("after close, saved %d requests and %d tokens; want 2 and 12", got.Requests, got.OutputTokens)
	}
	if got := again.logQuery("", 10).Total; got != 2 {
		t.Errorf("after close, the log has %d requests, want 2", got)
	}
}

// TestStatsSaveFailureRetries makes the folder unwritable, then writable
// again, and checks nothing recorded in between is lost.
func TestStatsSaveFailureRetries(t *testing.T) {
	if runtime.GOOS == "windows" || os.Getuid() == 0 {
		t.Skip("needs a folder the test can make read-only")
	}
	dir := t.TempDir()
	now := time.Now()
	u := newUsageStats(filepath.Join(dir, "stats.json"))
	u.now = func() time.Time { return now }

	u.record("m", api.Metrics{EvalCount: 1}, statsRequestInfo{Tag: "a"}, false)
	if err := u.save(); err != nil {
		t.Fatal(err)
	}

	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0o700) })
	u.record("m", api.Metrics{EvalCount: 2}, statsRequestInfo{Tag: "b"}, false)
	if err := u.save(); err == nil {
		t.Fatal("saving to a read-only folder should fail")
	}
	// A reset while the disk is failing forces a full rewrite later.
	u.record("m", api.Metrics{EvalCount: 3}, statsRequestInfo{Tag: "c"}, false)
	if err := u.save(); err == nil {
		t.Fatal("saving to a read-only folder should fail again")
	}

	if err := os.Chmod(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := u.save(); err != nil {
		t.Fatalf("save after the folder is writable: %v", err)
	}

	again := newUsageStats(u.path)
	if got := findStats(t, again.snapshot().AllTimeModels, "m"); got.Requests != 3 || got.OutputTokens != 6 {
		t.Errorf("saved %d requests and %d tokens, want 3 and 6", got.Requests, got.OutputTokens)
	}
	for _, tag := range []string{"a", "b", "c"} {
		if got := again.logQuery(tag, 10).Total; got != 1 {
			t.Errorf("tag %q: %d logged requests after the retry, want 1", tag, got)
		}
	}
}

// TestStatsLogLongLineSkipped checks that one damaged, very long line in the
// log is skipped and the lines after it still load.
func TestStatsLogLongLineSkipped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stats-log.jsonl")
	now := time.Now()
	line := func(tag string) string {
		b, _ := json.Marshal(api.RequestLogEntry{Model: "m", Tag: tag, RequestStats: api.RequestStats{At: now}})
		return string(b) + "\n"
	}
	data := line("before") + strings.Repeat("x", statsLogMaxLine+10) + "\n" + line("after")
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}

	l := loadStatsLog(path, now)
	if len(l.entries) != 2 || l.entries[0].Tag != "before" || l.entries[1].Tag != "after" {
		t.Errorf("loaded %+v, want the lines before and after the long one", l.entries)
	}
	if !l.rewrite {
		t.Error("the damaged line should be dropped on the next save")
	}
}

// TestStatsHeadersOnCompatEndpoints sends requests the way OpenAI and
// Anthropic style apps do, through the real router, with the tag and test
// headers.
func TestStatsHeadersOnCompatEndpoints(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		CompletionFn: func(_ context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			fn(llm.CompletionResponse{Content: "hi", Done: true, DoneReason: llm.DoneReasonStop, EvalCount: 3, EvalDuration: time.Second})
			return nil
		},
	}
	s := newServerWithMockRunner(t, &mock)
	now := time.Now()
	s.stats = newTestStats(t, &now)
	createMinimalGGUFModel(t, s, "compat", gguftest.KV{}, "{{ .Prompt }}", nil)
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}

	send := func(path, body string, headers map[string]string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("%s: status %d: %s", path, w.Code, w.Body.String())
		}
	}
	openai := `{"model":"compat","messages":[{"role":"user","content":"hi"}]}`
	anthropic := `{"model":"compat","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`

	send("/v1/chat/completions", openai, map[string]string{statsTagHeader: "openai-app"})
	send("/v1/messages", anthropic, map[string]string{statsTagHeader: "anthropic-app"})
	send("/v1/chat/completions", openai, map[string]string{statsTagHeader: "eval", statsKindHeader: "test"})
	send("/v1/messages", anthropic, map[string]string{statsKindHeader: "TEST"})

	for _, tag := range []string{"openai-app", "anthropic-app", "eval"} {
		if got := s.stats.logQuery(tag, 10); got.Total != 1 || got.Requests[0].OutputTokens != 3 {
			t.Errorf("tag %q: logged %+v, want one request", tag, got)
		}
	}
	if got := s.stats.logQuery("eval", 10).Requests[0]; !got.Test {
		t.Error("the test header didn't mark the request as a test")
	}
	if got := findStats(t, s.stats.snapshot().AllTimeModels, "compat:latest"); got.Requests != 2 {
		t.Errorf("usage counts %d requests, want 2 (the test requests are left out)", got.Requests)
	}
}

// TestStatsStreamCanceled stops a streaming request partway through, the
// way Ctrl+C does. It shouldn't be counted, as a request or as an error.
func TestStatsStreamCanceled(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	gin.SetMode(gin.TestMode)

	started := make(chan struct{})
	mock := mockRunner{
		CompletionFn: func(ctx context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			fn(llm.CompletionResponse{Content: "one "})
			close(started)
			<-ctx.Done()
			return ctx.Err()
		},
	}
	s := newServerWithMockRunner(t, &mock)
	now := time.Now()
	s.stats = newTestStats(t, &now)
	createMinimalGGUFModel(t, s, "cancel", gguftest.KV{}, "{{ .Prompt }}", nil)

	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(router)
	t.Cleanup(srv.Close)

	// Start a streaming request, read the first chunk, then hang up.
	ctx, cancel := context.WithCancel(t.Context())
	body, _ := json.Marshal(api.GenerateRequest{Model: "cancel", Prompt: "count", Tag: "stopped"})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/api/generate", bytes.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if _, err := bufio.NewReader(resp.Body).ReadBytes('\n'); err != nil {
		t.Fatalf("reading the first chunk: %v", err)
	}
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("the model never started writing")
	}
	cancel()

	// Wait for the server to notice and finish the request.
	deadline := time.Now().Add(5 * time.Second)
	for s.stats.snapshot().Active != 0 {
		if time.Now().After(deadline) {
			t.Fatal("the canceled request never finished")
		}
		time.Sleep(10 * time.Millisecond)
	}

	if got := s.stats.logQuery("", 10); got.Total != 0 {
		t.Errorf("a canceled request was logged: %+v", got.Requests)
	}
	if st := s.stats.snapshot(); len(st.AllTimeModels) != 0 {
		t.Errorf("a canceled request was counted: %+v", st.AllTimeModels)
	}
}

// TestStatsSuiteRunsRejectsBadBodies checks the size limit and broken JSON.
func TestStatsSuiteRunsRejectsBadBodies(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	now := time.Now()
	s := &Server{stats: newTestStats(t, &now)}
	router := gin.New()
	router.POST("/api/stats/suite-runs", s.SaveStatsSuiteRunHandler)

	post := func(body string) int {
		req := httptest.NewRequest(http.MethodPost, "/api/stats/suite-runs", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		return w.Code
	}

	big := `{"id":"run-big","suite":"basics","tag":"basics:run-big","models":["m"],"sections":["performance"],"results":[{"note":"` +
		strings.Repeat("x", statsSuiteRunMaxSize) + `"}]}`
	if code := post(big); code != http.StatusRequestEntityTooLarge {
		t.Errorf("oversized run: status %d, want 413", code)
	}
	for name, body := range map[string]string{
		"not json":     `{"id":`,
		"wrong shape":  `["a run"]`,
		"empty body":   ``,
		"unknown kind": `{"id":"run-1","suite":"basics","tag":"basics:run-1","models":["m"],"sections":["speed"],"results":[{}]}`,
	} {
		if code := post(body); code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400", name, code)
		}
	}

	// With stats off, runs aren't saved.
	off := gin.New()
	off.POST("/api/stats/suite-runs", (&Server{}).SaveStatsSuiteRunHandler)
	req := httptest.NewRequest(http.MethodPost, "/api/stats/suite-runs", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	off.ServeHTTP(w, req)
	if w.Code != http.StatusConflict {
		t.Errorf("stats off: status %d, want 409", w.Code)
	}
}

// TestStatsGradingScript runs the dashboard's grading code with Node, so the
// rules that decide Pass or Fail are tested. It's skipped if Node isn't
// installed.
func TestStatsGradingScript(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node isn't installed")
	}
	script, err := filepath.Abs("stats_grading.js")
	if err != nil {
		t.Fatal(err)
	}
	test := filepath.Join(t.TempDir(), "grading_test.js")
	if err := os.WriteFile(test, []byte(gradingCases), 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(node, test, script).CombinedOutput()
	if err != nil {
		t.Fatalf("grading checks failed:\n%s", out)
	}
}

// gradingCases is a small Node script: each case calls the grading code and
// compares the result. It prints every failure and exits non-zero if any.
const gradingCases = `
const G = require(process.argv[2]);
const fails = [];
function eq(name, got, want) {
  if (JSON.stringify(got) !== JSON.stringify(want)) fails.push(name + ": got " + JSON.stringify(got) + ", want " + JSON.stringify(want));
}
const reply = (answer) => JSON.stringify({ reasoning: "r", answer: answer });

// Numbers
const num = { type: "number", value: 42 };
eq("number exact", G.gradeAnswer(num, 42), true);
eq("number as text", G.gradeAnswer(num, "42"), true);
eq("number with comma", G.gradeAnswer({ type: "number", value: 1200 }, "1,200"), true);
eq("number with dollar", G.gradeAnswer({ type: "number", value: 5.5 }, "$5.50"), true);
eq("number wrong", G.gradeAnswer(num, 41), false);
eq("number not a number", G.gradeAnswer(num, "lots"), false);

// Times
const pm = { type: "time", value: "4:00 PM" };
eq("time same", G.gradeAnswer(pm, "4:00 PM"), true);
eq("time lower case", G.gradeAnswer(pm, "4:00pm"), true);
eq("time with dots", G.gradeAnswer(pm, "4:00 p.m."), true);
eq("time 24 hour", G.gradeAnswer(pm, "16:00"), true);
eq("time no period", G.gradeAnswer(pm, "4:00"), true);
eq("time wrong period", G.gradeAnswer(pm, "4:00 AM"), false);
eq("time wrong hour", G.gradeAnswer(pm, "5:00 PM"), false);
eq("time ambiguous hour", G.gradeAnswer(pm, "04:00"), true);
eq("time noon", G.gradeAnswer({ type: "time", value: "12:00 PM" }, "12:00"), true);
eq("time midnight", G.gradeAnswer({ type: "time", value: "12:00 AM" }, "00:00"), true);
eq("time midnight is not noon", G.gradeAnswer({ type: "time", value: "12:00 AM" }, "12:00 PM"), false);
eq("time 11:36 AM", G.gradeAnswer({ type: "time", value: "11:36 AM" }, "11:36"), true);
eq("time nonsense", G.gradeAnswer(pm, "25:00"), false);
eq("time missing", G.gradeAnswer(pm, "soon"), false);

// Text
const merc = { type: "text", value: "Mercury" };
eq("text exact", G.gradeAnswer(merc, "Mercury"), true);
eq("text case", G.gradeAnswer(merc, "mercury."), true);
eq("text extra words", G.gradeAnswer(merc, "Planet Mercury"), true);
eq("text negated", G.gradeAnswer(merc, "not Mercury"), false);
eq("text negated isn't", G.gradeAnswer(merc, "it isn't Mercury"), false);
eq("text wrong", G.gradeAnswer(merc, "Venus"), false);
eq("text part of a word", G.gradeAnswer({ type: "text", value: "art" }, "Mars"), false);
eq("text ignores accents", G.gradeAnswer({ type: "text", value: "São Paulo" }, "sao paulo"), true);
eq("text unicode exact", G.gradeAnswer({ type: "text", value: "São Paulo" }, "São Paulo"), true);
eq("text unicode composed", G.gradeAnswer({ type: "text", value: "Café" }, "Café"), true);

// Choices
const choice = { type: "choice", value: "B", options: ["A", "B", "C"] };
eq("choice right", G.gradeAnswer(choice, "B"), true);
eq("choice case", G.gradeAnswer(choice, "b"), true);
eq("choice wrong", G.gradeAnswer(choice, "C"), false);
eq("choice no partial", G.gradeAnswer(choice, "B or C"), false);

// Fields
const fields = { type: "fields", fields: [{ name: "meet_time", type: "time", value: "11:36 AM" }, { name: "distance_km", type: "number", value: 156 }] };
eq("fields right", G.gradeAnswer(fields, { meet_time: "11:36", distance_km: 156 }), true);
eq("fields one wrong", G.gradeAnswer(fields, { meet_time: "11:36", distance_km: 150 }), false);
eq("fields missing", G.gradeAnswer(fields, { meet_time: "11:36" }), false);
eq("fields not an object", G.gradeAnswer(fields, "11:36, 156"), false);
eq("fields expected", G.expected(fields), "meet time: 11:36 AM, distance km: 156");

// Whole replies
eq("reply pass", G.gradeReply(num, reply(42)), { pass: true, got: "42" });
eq("reply fail", G.gradeReply(num, reply(7)), { pass: false, got: "7" });
eq("reply fields shown", G.gradeReply(fields, reply({ meet_time: "11:36", distance_km: 156 })).got, "meet time: 11:36, distance km: 156");
eq("reply not json", G.gradeReply(num, "The answer is 42"), { pass: false, got: null });
eq("reply cut off", G.gradeReply(num, '{"reasoning":"lon'), { pass: false, got: null });
eq("reply no answer", G.gradeReply(num, '{"reasoning":"r"}'), { pass: false, got: null });
eq("reply in a code block", G.gradeReply(num, "` + "```" + `json\n" + reply(42) + "\n` + "```" + `"), { pass: true, got: "42" });
eq("reply with a sentence around it", G.gradeReply(num, "Here you go: " + reply(42) + " Hope that helps."), { pass: true, got: "42" });
eq("reply plain text", G.gradeReply({ type: "text", value: "Au" }, "The chemical symbol for gold is **Au**."), { pass: false, got: null });
eq("prompt asks for the shape", G.answerPrompt("What is 6x7?", num).indexOf('"required":["reasoning","answer"]') !== -1, true);
eq("prompt keeps the question", G.answerPrompt("What is 6x7?", num).indexOf("What is 6x7?") === 0, true);

// Schemas
const s = G.answerSchema(num);
eq("schema requires both", s.required, ["reasoning", "answer"]);
eq("schema number", s.properties.answer, { type: "number" });
eq("schema caps reasoning", s.properties.reasoning.maxLength > 0, true);
eq("schema text is short", G.answerSchema(merc).properties.answer.maxLength, 30);
eq("schema choice", G.answerSchema(choice).properties.answer.enum, ["A", "B", "C"]);
eq("schema fields", G.answerSchema(fields).properties.answer.required, ["meet_time", "distance_km"]);

// Rules for free-form replies
const rule = (type, extra) => ({ check: Object.assign({ type: type }, extra) });
eq("clean thinking", G.cleanAnswer("<think>hmm</think> Paris "), "Paris");
eq("exact", G.check(rule("exact", { value: "Paris" }), "**Paris.**"), true);
eq("exact wrong", G.check(rule("exact", { value: "Paris" }), "Paris, France"), false);
eq("contains", G.check(rule("contains", { value: "ocean" }), "The Ocean is big"), true);
eq("word count", G.check(rule("word_count", { value: 3 }), "one two three"), true);
eq("word count wrong", G.check(rule("word_count", { value: 3 }), "one two"), false);
eq("word count unicode", G.check(rule("word_count", { value: 2 }), "über café"), true);
eq("json list", G.check(rule("json_list", { key: "colors", length: 3 }), 'Sure: {"colors":["a","b","c"]}'), true);
eq("json list wrong length", G.check(rule("json_list", { key: "colors", length: 3 }), '{"colors":["a"]}'), false);
eq("json list broken", G.check(rule("json_list", { key: "colors", length: 3 }), '{"colors":['), false);
eq("uppercase", G.check(rule("uppercase"), "HELLO WORLD!"), true);
eq("uppercase unicode", G.check(rule("uppercase"), "ÉCOLE"), true);
eq("uppercase mixed", G.check(rule("uppercase"), "HELLO world"), false);
eq("uppercase no letters", G.check(rule("uppercase"), "123"), false);
eq("excludes letter", G.check(rule("excludes_letter", { value: "e" }), "A cat sat"), true);
eq("excludes letter found", G.check(rule("excludes_letter", { value: "e" }), "A cat ate"), false);
eq("excludes letter empty", G.check(rule("excludes_letter", { value: "e" }), ""), false);
eq("code", G.check(rule("code", { all: ["def ", "return"], any: ["[::-1]", "reversed"] }), "def r(s):\n  return s[::-1]"), true);
eq("code missing", G.check(rule("code", { all: ["def ", "return"] }), "lambda s: s"), false);
eq("unknown rule", G.check(rule("nope"), "anything"), false);

// Tags match api.ValidateTag
eq("tag simple", G.isValidTag("cloud-test"), true);
eq("tag punctuation", G.isValidTag("team/eval:v2.1_a@b"), true);
eq("tag unicode", G.isValidTag("テスト"), true);
eq("tag 64", G.isValidTag("a".repeat(64)), true);
eq("tag 65", G.isValidTag("a".repeat(65)), false);
eq("tag empty", G.isValidTag(""), false);
eq("tag html", G.isValidTag("<b>"), false);
eq("tag quote", G.isValidTag('a"b'), false);
eq("tag newline", G.isValidTag("a\nb"), false);

if (fails.length) { console.log(fails.join("\n")); process.exit(1); }
console.log("ok");
`

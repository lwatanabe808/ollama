package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestStatsSuiteBuiltInIsValid(t *testing.T) {
	s, err := parseStatsSuite(statsSuiteBasicsJSON)
	if err != nil {
		t.Fatalf("built-in suite is invalid: %v", err)
	}
	if len(s.Performance) == 0 || len(s.FactChecks) == 0 {
		t.Fatalf("built-in suite should have both sections, got %d performance and %d fact checks", len(s.Performance), len(s.FactChecks))
	}

	// Performance scenarios must not grade answers; that's what fact checks are for.
	var raw struct {
		Performance []map[string]any `json:"performance"`
	}
	if err := json.Unmarshal(statsSuiteBasicsJSON, &raw); err != nil {
		t.Fatal(err)
	}
	for _, sc := range raw.Performance {
		if _, ok := sc["check"]; ok {
			t.Errorf("performance scenario %v should not have an answer check", sc["id"])
		}
	}

	// Each input and output size group changes only one thing.
	byID := map[string]statsSuiteScenario{}
	for _, sc := range s.Performance {
		byID[sc.ID] = sc
	}
	for _, id := range []string{"input-short", "input-medium", "input-long"} {
		if byID[id].OutputTokens != byID["input-short"].OutputTokens {
			t.Errorf("%s should cap output the same as the other input sizes", id)
		}
	}
	for _, id := range []string{"output-short", "output-medium", "output-long"} {
		if byID[id].Input != "tiny" || byID[id].Output != "count" {
			t.Errorf("%s should use the same tiny input and counting output", id)
		}
	}
}

func TestStatsSuiteValidation(t *testing.T) {
	valid := `{"name":"mine","title":"Mine","performance":[],"fact_checks":[{"id":"a","group":"Known answers","category":"Math","level":"easy","prompt":"2+2?","answer":{"type":"number","value":4}}]}`
	if _, err := parseStatsSuite([]byte(valid)); err != nil {
		t.Fatalf("valid suite rejected: %v", err)
	}
	for name, bad := range map[string]string{
		"bad name":     strings.Replace(valid, `"mine"`, `"My Suite"`, 1),
		"bad level":    strings.Replace(valid, `"easy"`, `"extreme"`, 1),
		"bad answer":   strings.Replace(valid, `"number"`, `"vibes"`, 1),
		"answer type":  strings.Replace(valid, `"value":4`, `"value":"four"`, 1),
		"both graders": strings.Replace(valid, `"answer":{`, `"check":{"type":"exact"},"answer":{`, 1),
		"no grader":    strings.Replace(valid, `,"answer":{"type":"number","value":4}`, ``, 1),
		"bad choice":   strings.Replace(valid, `{"type":"number","value":4}`, `{"type":"choice","value":"Cal","options":["Amy","Ben"]}`, 1),
		"bad field":    strings.Replace(valid, `{"type":"number","value":4}`, `{"type":"fields","fields":[{"name":"Bad Name","type":"number","value":1}]}`, 1),
		"empty prompt": strings.Replace(valid, `"2+2?"`, `" "`, 1),
		"unknown key":  strings.Replace(valid, `"title"`, `"titel"`, 1),
		"empty suite":  `{"name":"mine","title":"Mine","performance":[],"fact_checks":[]}`,
		"duplicate id": `{"name":"mine","title":"Mine","fact_checks":[{"id":"a","group":"G","category":"M","level":"easy","prompt":"x","check":{"type":"exact"}},{"id":"a","group":"G","category":"M","level":"easy","prompt":"y","check":{"type":"exact"}}]}`,
		"huge output":  `{"name":"mine","title":"Mine","performance":[{"id":"p","title":"P","group":"G","isolates":"x","input":"tiny","output_tokens":99999}]}`,
	} {
		if _, err := parseStatsSuite([]byte(bad)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestStatsSuitesHandlerLoadsUserSuites(t *testing.T) {
	gin.SetMode(gin.TestMode)
	home := t.TempDir()
	setTestHome(t, home)
	dir := filepath.Join(home, ".ollama", "suites")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	good := `{"name":"mine","title":"Mine","fact_checks":[{"id":"a","group":"Known answers","category":"Math","level":"easy","prompt":"2+2?","answer":{"type":"number","value":4}}]}`
	_ = os.WriteFile(filepath.Join(dir, "mine.json"), []byte(good), 0o600)
	_ = os.WriteFile(filepath.Join(dir, "broken.json"), []byte("{nope"), 0o600)
	_ = os.WriteFile(filepath.Join(dir, "dupe.json"), []byte(strings.Replace(good, `"mine"`, `"basics"`, 1)), 0o600)

	s := &Server{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/stats/suites", nil)
	s.StatsSuitesHandler(c)

	var resp struct {
		Suites   []statsSuite        `json:"suites"`
		Problems []map[string]string `json:"problems"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Suites) != 2 || resp.Suites[0].Name != "basics" || !resp.Suites[0].BuiltIn || resp.Suites[1].Name != "mine" {
		t.Errorf("suites = %+v, want built-in basics then mine", resp.Suites)
	}
	if len(resp.Problems) != 2 {
		t.Errorf("problems = %+v, want the broken file and the duplicate name reported", resp.Problems)
	}
}

func TestStatsSuiteRunsSaveAndList(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	now := time.Now()
	s := &Server{stats: newTestStats(t, &now)}
	router := gin.New()
	router.GET("/api/stats/suite-runs", s.StatsSuiteRunsHandler)
	router.POST("/api/stats/suite-runs", s.SaveStatsSuiteRunHandler)

	post := func(body, contentType string) int {
		req := httptest.NewRequest(http.MethodPost, "/api/stats/suite-runs", strings.NewReader(body))
		if contentType != "" {
			req.Header.Set("Content-Type", contentType)
		}
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		return w.Code
	}
	run := func(id string) string {
		return `{"id":"` + id + `","suite":"basics","tag":"basics:` + id + `","models":["llama3.2"],"sections":["fact_checks"],"results":[{"test":"a","pass":true}]}`
	}

	if code := post(run("run-1"), "application/json"); code != http.StatusCreated {
		t.Fatalf("save status = %d", code)
	}
	// Plain-text posts are how other websites could sneak a request in.
	if code := post(run("run-2"), "text/plain"); code != http.StatusUnsupportedMediaType {
		t.Errorf("non-JSON post status = %d, want 415", code)
	}
	for name, bad := range map[string]string{
		"bad id":     strings.Replace(run("x"), `"x"`, `"Bad ID!"`, 1),
		"no models":  strings.Replace(run("x"), `["llama3.2"]`, `[]`, 1),
		"bad tag":    strings.Replace(run("x"), `"basics:x"`, `"<b>"`, 1),
		"no results": strings.Replace(run("x"), `[{"test":"a","pass":true}]`, `{}`, 1),
	} {
		if code := post(bad, "application/json"); code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400", name, code)
		}
	}

	// Save 13 more runs: bulk-a through bulk-m. Only the newest 10 are kept.
	var ids []string
	for i := range statsSuiteRunsKept + 3 {
		id := "bulk-" + string(rune('a'+i))
		ids = append(ids, id)
		if code := post(run(id), "application/json"); code != http.StatusCreated {
			t.Fatalf("saving %s: status %d", id, code)
		}
	}
	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/stats/suite-runs", nil))
	var resp struct {
		Runs []statsSuiteRun `json:"runs"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, r := range resp.Runs {
		got = append(got, r.ID)
	}
	var want []string
	for i := len(ids) - 1; i >= len(ids)-statsSuiteRunsKept; i-- {
		want = append(want, ids[i])
	}
	if !slices.Equal(got, want) {
		t.Errorf("runs = %v, want the newest %d, newest first: %v", got, statsSuiteRunsKept, want)
	}

	// Saving a run again under the same id replaces it, instead of adding a
	// second copy.
	again := strings.Replace(run("bulk-m"), `"llama3.2"`, `"gemma3"`, 1)
	if code := post(again, "application/json"); code != http.StatusCreated {
		t.Fatalf("saving bulk-m again: status %d", code)
	}
	w = httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/stats/suite-runs", nil))
	resp.Runs = nil
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	copies := 0
	for _, r := range resp.Runs {
		if r.ID == "bulk-m" {
			copies++
			if len(r.Models) != 1 || r.Models[0] != "gemma3" {
				t.Errorf("bulk-m wasn't replaced: models %v", r.Models)
			}
		}
	}
	if copies != 1 || len(resp.Runs) != statsSuiteRunsKept {
		t.Errorf("%d copies of bulk-m among %d runs, want 1 among %d", copies, len(resp.Runs), statsSuiteRunsKept)
	}
}

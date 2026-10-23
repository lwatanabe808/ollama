package server

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

// A test suite is a fixed set of scenarios for comparing models on this
// machine. The dashboard's Tests page runs them through the normal chat API
// with a tag, so their speed also shows up in the usage stats. Performance
// scenarios change one thing at a time (input size, output size, thinking,
// and so on). Fact checks grade answers, and are kept separate so the two
// never mix.
//
// Known-answer fact checks use Ollama's structured outputs: the model must
// reply with JSON holding its reasoning and a typed answer, so grading is an
// exact comparison instead of searching free text. Following-instructions
// checks stay free-form, since the format is what they test.
//
// Ollama ships one suite, "basics". People can add their own as JSON files
// in ~/.ollama/suites.

//go:embed stats_suite_basics.json
var statsSuiteBasicsJSON []byte

const (
	statsSuiteRunsKept     = 10
	statsSuiteRunMaxSize   = 2 << 20 // largest saved run, 2 MiB
	statsSuiteFileMaxSize  = 2 << 20 // largest suite file, 2 MiB
	statsSuiteMaxTests     = 200
	statsSuiteRunMaxModels = 32
)

// statsSuiteSections are the sections a saved run can name. "accuracy" is
// the earlier name for fact checks, accepted so older runs still load.
var statsSuiteSections = map[string]bool{"performance": true, "fact_checks": true, "accuracy": true}

var (
	statsSuiteNamePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,31}$`)
	statsSuiteIDPattern   = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,47}$`)
)

type statsSuite struct {
	Name        string               `json:"name"`
	Title       string               `json:"title"`
	Description string               `json:"description,omitempty"`
	BuiltIn     bool                 `json:"built_in"`
	Performance []statsSuiteScenario `json:"performance"`
	FactChecks  []statsSuiteTest     `json:"fact_checks"`
}

// statsSuiteScenario measures one thing. The page builds the prompt from
// these settings, so the same text is used for every model.
type statsSuiteScenario struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	Group    string `json:"group"`
	Isolates string `json:"isolates"`

	// Input is "tiny" or "reasoning", or leave it empty and set InputWords.
	Input      string `json:"input,omitempty"`
	InputWords int    `json:"input_words,omitempty"`
	// Output "count" asks the model to count upward, so it writes exactly
	// OutputTokens tokens instead of stopping early.
	Output       string `json:"output,omitempty"`
	OutputTokens int    `json:"output_tokens"`

	Cold         bool  `json:"cold,omitempty"`  // unload the model first
	Reuse        bool  `json:"reuse,omitempty"` // send the same input twice
	Think        *bool `json:"think,omitempty"`
	ThinkingOnly bool  `json:"thinking_only,omitempty"`

	// Context scenarios fill a small window to a share of its size.
	Window int     `json:"window,omitempty"`
	Fill   float64 `json:"fill,omitempty"`
}

type statsSuiteTest struct {
	ID       string `json:"id"`
	Group    string `json:"group"`
	Category string `json:"category"`
	Level    string `json:"level"`
	Prompt   string `json:"prompt"`
	// Set exactly one: Answer for a known answer graded through structured
	// output, or Check for a rule run on a free-form reply.
	Answer *statsSuiteAnswer `json:"answer,omitempty"`
	Check  *statsSuiteCheck  `json:"check,omitempty"`
}

// statsSuiteAnswer is the expected answer for a known-answer test. The page
// turns it into a JSON schema, so the model's answer always has this shape.
type statsSuiteAnswer struct {
	Type    string            `json:"type"` // number, text, time, choice, fields
	Value   any               `json:"value,omitempty"`
	Options []string          `json:"options,omitempty"`
	Fields  []statsSuiteField `json:"fields,omitempty"`
}

type statsSuiteField struct {
	Name  string `json:"name"`
	Type  string `json:"type"` // number, text, time
	Value any    `json:"value"`
}

// statsSuiteCheck is a rule run on a free-form reply, for tests where the
// format itself is what's being tested.
type statsSuiteCheck struct {
	Type   string   `json:"type"`
	Value  any      `json:"value,omitempty"`
	Key    string   `json:"key,omitempty"`    // json_list
	Length int      `json:"length,omitempty"` // json_list
	All    []string `json:"all,omitempty"`    // code: must contain every one
	Any    []string `json:"any,omitempty"`    // code: must contain at least one
}

// statsSuiteCheckTypes lists the answer checks the Tests page knows how to run.
var statsSuiteCheckTypes = map[string]bool{
	"exact": true, "contains": true, "word_count": true, "json_list": true,
	"uppercase": true, "excludes_letter": true, "code": true,
}

var statsSuiteFieldName = regexp.MustCompile(`^[a-z][a-z0-9_]{0,31}$`)

func validateStatsSuiteValue(kind string, v any) error {
	switch kind {
	case "number":
		if _, ok := v.(float64); !ok {
			return errors.New("value must be a number")
		}
	case "text", "time":
		if s, ok := v.(string); !ok || strings.TrimSpace(s) == "" {
			return errors.New("value must be text")
		}
	default:
		return fmt.Errorf("unknown answer type %q", kind)
	}
	return nil
}

func (a statsSuiteAnswer) validate() error {
	switch a.Type {
	case "number", "text", "time":
		return validateStatsSuiteValue(a.Type, a.Value)
	case "choice":
		v, _ := a.Value.(string)
		if len(a.Options) < 2 || !slices.Contains(a.Options, v) {
			return errors.New("a choice needs at least 2 options, and the value must be one of them")
		}
	case "fields":
		if len(a.Fields) == 0 || len(a.Fields) > 8 {
			return errors.New("fields needs 1 to 8 fields")
		}
		seen := map[string]bool{}
		for _, f := range a.Fields {
			if !statsSuiteFieldName.MatchString(f.Name) || seen[f.Name] {
				return fmt.Errorf("field name %q must be unique lowercase letters, numbers, and underscores", f.Name)
			}
			seen[f.Name] = true
			if err := validateStatsSuiteValue(f.Type, f.Value); err != nil {
				return fmt.Errorf("field %q: %w", f.Name, err)
			}
		}
	default:
		return fmt.Errorf("unknown answer type %q", a.Type)
	}
	return nil
}

func (s statsSuite) validate() error {
	if !statsSuiteNamePattern.MatchString(s.Name) {
		return fmt.Errorf("name %q must be lowercase letters, numbers, and dashes, up to 32 characters", s.Name)
	}
	if strings.TrimSpace(s.Title) == "" {
		return errors.New("title is required")
	}
	if len(s.Performance)+len(s.FactChecks) == 0 {
		return errors.New("add at least one performance scenario or fact check")
	}
	if len(s.Performance)+len(s.FactChecks) > statsSuiteMaxTests {
		return fmt.Errorf("use %d tests or fewer", statsSuiteMaxTests)
	}

	seen := map[string]bool{}
	checkID := func(id string) error {
		if !statsSuiteIDPattern.MatchString(id) {
			return fmt.Errorf("id %q must be lowercase letters, numbers, and dashes", id)
		}
		if seen[id] {
			return fmt.Errorf("id %q is used twice", id)
		}
		seen[id] = true
		return nil
	}

	for _, sc := range s.Performance {
		if err := checkID(sc.ID); err != nil {
			return err
		}
		if sc.Title == "" || sc.Group == "" {
			return fmt.Errorf("scenario %q needs a title and a group", sc.ID)
		}
		if sc.OutputTokens < 1 || sc.OutputTokens > 4096 {
			return fmt.Errorf("scenario %q: output_tokens must be 1 to 4096", sc.ID)
		}
		switch {
		case sc.Window > 0:
			if sc.Window < 512 || sc.Window > 131072 || sc.Fill < 0.1 || sc.Fill > 2 {
				return fmt.Errorf("scenario %q: window must be 512 to 131072 and fill 0.1 to 2", sc.ID)
			}
		case sc.Input == "tiny" || sc.Input == "reasoning":
		case sc.Input == "" && sc.InputWords > 0 && sc.InputWords <= 20000:
		default:
			return fmt.Errorf("scenario %q: set input to tiny or reasoning, or input_words to 1 to 20000", sc.ID)
		}
		if sc.Output != "" && sc.Output != "count" {
			return fmt.Errorf("scenario %q: output must be empty or count", sc.ID)
		}
	}

	for _, t := range s.FactChecks {
		if err := checkID(t.ID); err != nil {
			return err
		}
		if t.Group == "" || t.Category == "" || strings.TrimSpace(t.Prompt) == "" {
			return fmt.Errorf("test %q needs a group, a category, and a prompt", t.ID)
		}
		switch t.Level {
		case "easy", "medium", "hard":
		default:
			return fmt.Errorf("test %q: level must be easy, medium, or hard", t.ID)
		}
		switch {
		case (t.Answer == nil) == (t.Check == nil):
			return fmt.Errorf("test %q: set exactly one of answer or check", t.ID)
		case t.Answer != nil:
			if err := t.Answer.validate(); err != nil {
				return fmt.Errorf("test %q: %w", t.ID, err)
			}
		case !statsSuiteCheckTypes[t.Check.Type]:
			return fmt.Errorf("test %q: unknown check type %q", t.ID, t.Check.Type)
		}
	}
	return nil
}

func parseStatsSuite(bts []byte) (statsSuite, error) {
	var s statsSuite
	dec := json.NewDecoder(bytes.NewReader(bts))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return s, err
	}
	// Send empty sections as [] rather than null.
	if s.Performance == nil {
		s.Performance = []statsSuiteScenario{}
	}
	if s.FactChecks == nil {
		s.FactChecks = []statsSuiteTest{}
	}
	return s, s.validate()
}

// readSmallFile reads a file only if it's at most max bytes, so a huge file
// (or something like /dev/zero) can't use up memory.
func readSmallFile(path string, max int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if info, err := f.Stat(); err == nil && info.Mode().IsRegular() && info.Size() > max {
		return nil, fmt.Errorf("file is larger than %d bytes", max)
	}
	bts, err := io.ReadAll(io.LimitReader(f, max+1))
	if err == nil && int64(len(bts)) > max {
		err = fmt.Errorf("file is larger than %d bytes", max)
	}
	return bts, err
}

func statsDir() (string, error) {
	path, err := defaultStatsPath()
	if err != nil {
		return "", err
	}
	return filepath.Dir(path), nil
}

// loadStatsSuites returns the built-in suite, then any valid suites in
// ~/.ollama/suites. Files that can't be used are reported, not fatal.
func loadStatsSuites() (suites []statsSuite, problems []gin.H) {
	builtIn, err := parseStatsSuite(statsSuiteBasicsJSON)
	if err != nil {
		// Covered by tests, so this means a broken build.
		slog.Error("built-in test suite is invalid", "error", err)
	} else {
		builtIn.BuiltIn = true
		suites = append(suites, builtIn)
	}

	dir, err := statsDir()
	if err != nil {
		return suites, problems
	}
	files, _ := filepath.Glob(filepath.Join(dir, "suites", "*.json"))
	sort.Strings(files)
	names := map[string]bool{builtIn.Name: true}
	for _, f := range files {
		bts, err := readSmallFile(f, statsSuiteFileMaxSize)
		var s statsSuite
		if err == nil {
			s, err = parseStatsSuite(bts)
		}
		if err == nil && names[s.Name] {
			err = fmt.Errorf("a suite named %q already exists", s.Name)
		}
		if err != nil {
			problems = append(problems, gin.H{"file": filepath.Base(f), "error": err.Error()})
			continue
		}
		names[s.Name] = true
		suites = append(suites, s)
	}
	return suites, problems
}

func (s *Server) StatsSuitesHandler(c *gin.Context) {
	suites, problems := loadStatsSuites()
	if problems == nil {
		problems = []gin.H{}
	}
	c.JSON(http.StatusOK, gin.H{"suites": suites, "problems": problems})
}

// statsSuiteRun is a finished (or cancelled) run, saved by the Tests page.
// The server checks the outline and size; the results themselves are the
// page's to describe.
type statsSuiteRun struct {
	ID         string          `json:"id"`
	Suite      string          `json:"suite"`
	Tag        string          `json:"tag"`
	StartedAt  time.Time       `json:"started_at"`
	FinishedAt time.Time       `json:"finished_at"`
	Cancelled  bool            `json:"cancelled,omitempty"`
	Models     []string        `json:"models"`
	Sections   []string        `json:"sections"`
	Results    json.RawMessage `json:"results"`
}

var statsSuiteRunsMu sync.Mutex

func statsSuiteRunsPath() (string, error) {
	dir, err := statsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "stats-suite-runs.jsonl"), nil
}

// readStatsSuiteRuns returns saved runs, oldest first, skipping bad lines.
func readStatsSuiteRuns(path string) []statsSuiteRun {
	bts, err := readSmallFile(path, (statsSuiteRunsKept+1)*statsSuiteRunMaxSize)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Warn("could not read saved test runs", "path", path, "error", err)
		}
		return nil
	}
	var runs []statsSuiteRun
	for line := range bytes.SplitSeq(bts, []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var r statsSuiteRun
		if err := json.Unmarshal(line, &r); err == nil && r.ID != "" {
			runs = append(runs, r)
		}
	}
	return runs
}

func (s *Server) StatsSuiteRunsHandler(c *gin.Context) {
	path, err := statsSuiteRunsPath()
	if err != nil || s.stats == nil {
		c.JSON(http.StatusOK, gin.H{"runs": []statsSuiteRun{}})
		return
	}
	statsSuiteRunsMu.Lock()
	runs := readStatsSuiteRuns(path)
	statsSuiteRunsMu.Unlock()

	// Newest first.
	out := make([]statsSuiteRun, 0, len(runs))
	for i := len(runs) - 1; i >= 0; i-- {
		out = append(out, runs[i])
	}
	c.JSON(http.StatusOK, gin.H{"runs": out})
}

func (s *Server) SaveStatsSuiteRunHandler(c *gin.Context) {
	// Saved runs are stats too, so the off switch covers them.
	if s.stats == nil {
		c.JSON(http.StatusConflict, statsOffError)
		return
	}
	// Requiring JSON means other websites can't post here without the
	// browser first asking permission, which Ollama's CORS rules deny.
	if mt, _, _ := mime.ParseMediaType(c.GetHeader("Content-Type")); mt != "application/json" {
		c.JSON(http.StatusUnsupportedMediaType, gin.H{"error": "send the run as application/json"})
		return
	}
	body, err := io.ReadAll(io.LimitReader(c.Request.Body, statsSuiteRunMaxSize+1))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if len(body) > statsSuiteRunMaxSize {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "run is too large"})
		return
	}

	var run statsSuiteRun
	if err := json.Unmarshal(body, &run); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	switch {
	case !statsSuiteIDPattern.MatchString(run.ID):
		err = errors.New("id must be lowercase letters, numbers, and dashes")
	case !statsSuiteNamePattern.MatchString(run.Suite):
		err = errors.New("suite name is not valid")
	case len(run.Models) == 0 || len(run.Models) > statsSuiteRunMaxModels:
		err = fmt.Errorf("a run needs 1 to %d models", statsSuiteRunMaxModels)
	case len(run.Results) == 0 || run.Results[0] != '[':
		err = errors.New("results must be a list")
	case run.Tag != "":
		err = api.ValidateTag(run.Tag)
	}
	for _, m := range run.Models {
		if err == nil && (m == "" || len(m) > 200) {
			err = errors.New("model names must be 1 to 200 characters")
		}
	}
	for _, sec := range run.Sections {
		if err == nil && !statsSuiteSections[sec] {
			err = fmt.Errorf("unknown section %q", sec)
		}
	}
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	path, err := statsSuiteRunsPath()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	statsSuiteRunsMu.Lock()
	defer statsSuiteRunsMu.Unlock()

	// A run saved again under the same id replaces the earlier copy.
	runs := slices.DeleteFunc(readStatsSuiteRuns(path), func(r statsSuiteRun) bool { return r.ID == run.ID })
	runs = append(runs, run)
	if len(runs) > statsSuiteRunsKept {
		runs = runs[len(runs)-statsSuiteRunsKept:]
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, r := range runs {
		if err := enc.Encode(r); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
	}
	if err := writeFileAtomic(path, buf.Bytes()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.Status(http.StatusCreated)
}

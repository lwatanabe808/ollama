package server

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

const (
	statsFileVersion  = 1
	statsSaveInterval = time.Minute
	// statsKeepDays is how many days of daily totals are kept, counting
	// today. All-time totals are kept forever.
	statsKeepDays  = 31
	statsDayLayout = "2006-01-02"
)

// statsFile is what is saved to ~/.ollama/stats.json.
type statsFile struct {
	Version int                                   `json:"version"`
	Since   time.Time                             `json:"since"`
	AllTime map[string]*api.ModelStats            `json:"all_time"`
	Days    map[string]map[string]*api.ModelStats `json:"days"`
}

func newStatsFile(now time.Time) statsFile {
	return statsFile{
		Version: statsFileVersion,
		Since:   now,
		AllTime: map[string]*api.ModelStats{},
		Days:    map[string]map[string]*api.ModelStats{},
	}
}

// usageStats keeps per-model usage totals for chat and generate requests,
// plus a log of each request. Both live in memory and are saved to disk every
// minute and on shutdown, so they survive restarts. Only counts and timings
// are kept, never prompts or answers.
type usageStats struct {
	path   string
	now    func() time.Time
	active atomic.Int64

	mu    sync.Mutex // guards data, dirty, and log
	data  statsFile
	dirty bool
	log   statsLog

	// saveMu makes saves run one at a time, so an older save can't finish
	// after a newer one (say, the minute timer's save landing after a reset)
	// and put old numbers back on disk.
	saveMu sync.Mutex

	stop     chan struct{}
	stopOnce sync.Once
}

func defaultStatsPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "stats.json"), nil
}

// newUsageStats loads saved totals from path. A missing file starts fresh.
// A file that can't be read is moved aside so it is not overwritten.
func newUsageStats(path string) *usageStats {
	u := &usageStats{path: path, now: time.Now, stop: make(chan struct{})}
	u.data = newStatsFile(u.now())
	u.log = loadStatsLog(statsLogPath(path), u.now())

	bts, err := os.ReadFile(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		return u
	case err != nil:
		slog.Warn("could not read usage stats, starting fresh", "path", path, "error", err)
		return u
	}

	var loaded statsFile
	if err := json.Unmarshal(bts, &loaded); err != nil || loaded.Version != statsFileVersion {
		bad := path + ".corrupt"
		slog.Warn("usage stats file is not valid, moving it aside and starting fresh", "path", path, "moved_to", bad, "error", err)
		_ = os.Rename(path, bad)
		return u
	}

	u.data = cleanStatsFile(loaded)
	u.pruneDaysLocked(u.now())
	return u
}

// cleanStatsFile drops empty entries a hand-edited or damaged file could
// hold, like {"all_time": {"m": null}}, so later code never meets a nil.
func cleanStatsFile(f statsFile) statsFile {
	if f.AllTime == nil {
		f.AllTime = map[string]*api.ModelStats{}
	}
	if f.Days == nil {
		f.Days = map[string]map[string]*api.ModelStats{}
	}
	for name, m := range f.AllTime {
		if m == nil {
			delete(f.AllTime, name)
		}
	}
	for day, models := range f.Days {
		if models == nil {
			delete(f.Days, day)
			continue
		}
		for name, m := range models {
			if m == nil {
				delete(models, name)
			}
		}
	}
	return f
}

// statsRequestInfo describes a request and the loaded model that answered it.
type statsRequestInfo struct {
	ContextLength int
	MemorySize    uint64
	MemoryVRAM    uint64
	Tag           string
	Test          bool // sent by a test tool; logged but not counted as usage
}

// statsInfoFor gathers what the stats need about a request: its tag, whether
// it's a test, and the model's window and memory from its runner. Call it
// before the answer starts, since the runner is busy after that.
func statsInfoFor(c *gin.Context, r interface {
	ContextLength() int
	MemorySize() (total, vram uint64)
},
) statsRequestInfo {
	total, vram := r.MemorySize()
	return statsRequestInfo{
		ContextLength: r.ContextLength(),
		MemorySize:    total,
		MemoryVRAM:    vram,
		Tag:           c.GetString(statsTagKey),
		Test:          statsIsTest(c),
	}
}

// requestStatsFor pulls one request's numbers out of its final metrics.
func requestStatsFor(m api.Metrics, info statsRequestInfo, at time.Time) api.RequestStats {
	r := api.RequestStats{
		At:             at,
		ContextLength:  int64(info.ContextLength),
		MemorySize:     int64(info.MemorySize),
		PromptTokens:   int64(m.PromptEvalCount),
		OutputTokens:   int64(m.EvalCount),
		LoadDuration:   m.LoadDuration,
		PromptDuration: m.PromptEvalDuration,
		OutputDuration: m.EvalDuration,
		TotalDuration:  m.TotalDuration,
	}
	if m.PromptEvalCachedCount != nil {
		r.PromptCachedTokens = int64(*m.PromptEvalCachedCount)
	}
	if info.MemorySize > 0 {
		vram := int64(info.MemoryVRAM)
		r.MemoryVRAM = &vram
	}
	return r
}

func copyInt64(p *int64) *int64 {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

// addRequest adds one request to a model's totals. Requests must be added
// oldest first, so the latest window, memory, and last request win. It's used
// both for the saved totals and for totals rebuilt from the log.
func addRequest(s *api.ModelStats, r api.RequestStats, failed bool) {
	s.Requests++
	if r.At.After(s.LastUsed) {
		s.LastUsed = r.At
	}
	if r.ContextLength > 0 {
		s.ContextLength = r.ContextLength
	}
	if r.MemorySize > 0 {
		s.MemorySize = r.MemorySize
		s.MemoryVRAM = copyInt64(r.MemoryVRAM)
	}
	if failed {
		s.Errors++
		return
	}
	s.TrackContextFill(r.ContextUsed(), r.ContextLength)
	last := r
	last.MemoryVRAM = copyInt64(r.MemoryVRAM)
	s.Last = &last
	s.PromptTokens += r.PromptTokens
	s.PromptCachedTokens += r.PromptCachedTokens
	s.OutputTokens += r.OutputTokens
	s.LoadDuration += r.LoadDuration
	s.PromptDuration += r.PromptDuration
	s.OutputDuration += r.OutputDuration
	s.TotalDuration += r.TotalDuration
}

// record adds one finished request to the log and, unless it's a test, to
// today's and the all-time totals. info describes the request and model.
func (u *usageStats) record(model string, m api.Metrics, info statsRequestInfo, failed bool) {
	if u == nil || model == "" {
		return
	}

	now := u.now()
	day := now.Format(statsDayLayout)
	r := requestStatsFor(m, info, now)

	u.mu.Lock()
	defer u.mu.Unlock()

	u.log.add(api.RequestLogEntry{RequestStats: r, Model: model, Tag: info.Tag, Error: failed, Test: info.Test}, now)
	u.dirty = true

	// Test requests are only logged, so test runs don't count as usage.
	if info.Test {
		return
	}

	all, ok := u.data.AllTime[model]
	if !ok {
		all = &api.ModelStats{Model: model}
		u.data.AllTime[model] = all
	}
	addRequest(all, r, failed)

	models, ok := u.data.Days[day]
	if !ok {
		models = map[string]*api.ModelStats{}
		u.data.Days[day] = models
		u.pruneDaysLocked(now)
	}
	today, ok := models[model]
	if !ok {
		today = &api.ModelStats{Model: model}
		models[model] = today
	}
	addRequest(today, r, failed)
}

// pruneDaysLocked keeps the last statsKeepDays days, counting today.
func (u *usageStats) pruneDaysLocked(now time.Time) {
	cutoff := now.AddDate(0, 0, -(statsKeepDays - 1)).Format(statsDayLayout)
	for day := range u.data.Days {
		// The day layout sorts the same as time, so a string compare works.
		if day < cutoff {
			delete(u.data.Days, day)
		}
	}
}

// sortedStats copies a set of model totals into a list, most used first.
// The copies share nothing with the saved totals.
func sortedStats(m map[string]*api.ModelStats) []api.ModelStats {
	out := make([]api.ModelStats, 0, len(m))
	for _, s := range m {
		c := *s
		c.MemoryVRAM = copyInt64(s.MemoryVRAM)
		if s.Last != nil {
			last := *s.Last
			last.MemoryVRAM = copyInt64(s.Last.MemoryVRAM)
			c.Last = &last
		}
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Requests != out[j].Requests {
			return out[i].Requests > out[j].Requests
		}
		return out[i].Model < out[j].Model
	})
	return out
}

func (u *usageStats) snapshot() api.StatsResponse {
	if u == nil {
		return api.StatsResponse{TodayModels: []api.ModelStats{}, AllTimeModels: []api.ModelStats{}, Days: []api.DayStats{}, Tags: []api.TagStats{}, Disabled: true}
	}

	now := u.now()
	today := now.Format(statsDayLayout)

	u.mu.Lock()
	defer u.mu.Unlock()

	u.pruneDaysLocked(now)
	resp := api.StatsResponse{
		Since:         u.data.Since,
		Today:         today,
		TodayModels:   sortedStats(u.data.Days[today]),
		AllTimeModels: sortedStats(u.data.AllTime),
		Active:        u.active.Load(),
		Days:          u.daysLocked(),
	}
	tags := u.log.tags()
	resp.TagsTotal = len(tags)
	resp.Tags = tags[:min(len(tags), statsTagsListed)]
	return resp
}

// logQuery looks up logged requests. An empty tag means all requests.
func (u *usageStats) logQuery(tag string, limit int) api.StatsLogResponse {
	if u == nil {
		return api.StatsLogResponse{Tag: tag, Models: []api.ModelStats{}, Requests: []api.RequestLogEntry{}, Disabled: true}
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.log.query(tag, limit)
}

// daysLocked adds up each saved day across all models, oldest first.
func (u *usageStats) daysLocked() []api.DayStats {
	days := make([]api.DayStats, 0, len(u.data.Days))
	for date, models := range u.data.Days {
		d := api.DayStats{Date: date}
		for _, m := range models {
			d.Requests += m.Requests
			d.Errors += m.Errors
			d.OutputTokens += m.OutputTokens
		}
		days = append(days, d)
	}
	sort.Slice(days, func(i, j int) bool { return days[i].Date < days[j].Date })
	return days
}

// reset clears all totals and the request log, and saves right away.
func (u *usageStats) reset() error {
	u.mu.Lock()
	u.data = newStatsFile(u.now())
	u.log.clear()
	u.dirty = true
	u.mu.Unlock()
	return u.save()
}

// save writes the totals and any new log lines to disk. The file is a backup
// copy: the stats stay in memory, where requests keep adding to them and
// readers get them from. Saves run one at a time; see saveMu.
func (u *usageStats) save() error {
	if u == nil {
		return nil
	}
	u.saveMu.Lock()
	defer u.saveMu.Unlock()
	return errors.Join(u.saveTotals(), u.saveLog())
}

// saveTotals writes the totals if anything changed since the last save.
func (u *usageStats) saveTotals() error {
	u.mu.Lock()
	if !u.dirty {
		u.mu.Unlock()
		return nil
	}
	bts, err := json.MarshalIndent(u.data, "", "  ")
	u.dirty = false
	u.mu.Unlock()
	if err != nil {
		return err
	}

	if err := writeFileAtomic(u.path, bts); err != nil {
		// Try again on the next save.
		u.mu.Lock()
		u.dirty = true
		u.mu.Unlock()
		return err
	}
	return nil
}

// writeFileAtomic replaces path in one step: it writes a temp file, flushes
// it to disk, then renames it over the old file. If Ollama or the computer
// stops partway, the old file is left as it was. CreateTemp makes the file
// readable only by its owner.
func writeFileAtomic(path string, bts []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.Write(bts); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// run saves the totals and new log lines every minute until close is called.
func (u *usageStats) run() {
	u.runEvery(statsSaveInterval)
}

func (u *usageStats) runEvery(interval time.Duration) {
	if u == nil {
		return
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if err := u.save(); err != nil {
				slog.Warn("could not save usage stats", "path", u.path, "error", err)
			}
		case <-u.stop:
			return
		}
	}
}

// close stops the save loop and does a final save. Safe to call more than once.
func (u *usageStats) close() error {
	if u == nil {
		return nil
	}
	u.stopOnce.Do(func() { close(u.stop) })
	if err := u.save(); err != nil {
		return fmt.Errorf("save usage stats: %w", err)
	}
	return nil
}

// observe is the stats meter. Like a meter on a pipe, it sits between a
// handler's response channel and the code that writes the response, passes
// every item through unchanged, and records the request once the final chunk
// (Done) or an error goes by.
//
// Only requests that reach the model are seen here. Requests rejected before
// that (unknown model, bad request) and cloud requests aren't counted.
func (u *usageStats) observe(ctx context.Context, model string, info statsRequestInfo, in chan any) chan any {
	if u == nil {
		return in
	}

	out := make(chan any)
	u.active.Add(1)
	go func() {
		defer close(out)
		defer u.active.Add(-1)

		recorded := false
		forward := true
		for v := range in {
			if !recorded {
				switch r := v.(type) {
				case api.GenerateResponse:
					if r.Done {
						u.record(model, r.Metrics, info, false)
						recorded = true
					}
				case api.ChatResponse:
					if r.Done {
						u.record(model, r.Metrics, info, false)
						recorded = true
					}
				case gin.H:
					if _, ok := r["error"]; ok {
						// If the caller went away (stopped the answer or
						// closed the app), the model didn't fail, so it's
						// not counted as an error.
						if ctx.Err() == nil {
							u.record(model, api.Metrics{}, info, true)
						}
						recorded = true
					}
				}
			}

			if !forward {
				// The reader is gone. Keep draining so the sender never blocks.
				continue
			}
			select {
			case out <- v:
			case <-ctx.Done():
				forward = false
			}
		}
	}()
	return out
}

// statsModelName prefers the resolved model name, so "llama3.2" and
// "llama3.2:latest" are counted as the same model.
func statsModelName(m *Model, requested string) string {
	if m != nil && m.ShortName != "" {
		return m.ShortName
	}
	return requested
}

//go:embed stats_dashboard.html
var statsDashboardHTML []byte

// statsDashboardFiles are the dashboard's styles and scripts, served at
// /stats/static/NAME. Keeping them out of the HTML keeps each one small and
// lets the page run with no inline scripts.
var statsDashboardFiles = map[string]struct {
	path, contentType string
}{
	"dashboard.css": {"stats_dashboard.css", "text/css; charset=utf-8"},
	"common.js":     {"stats_common.js", "text/javascript; charset=utf-8"},
	"grading.js":    {"stats_grading.js", "text/javascript; charset=utf-8"},
	"usage.js":      {"stats_usage.js", "text/javascript; charset=utf-8"},
	"tests.js":      {"stats_tests.js", "text/javascript; charset=utf-8"},
}

//go:embed stats_dashboard.css stats_common.js stats_grading.js stats_usage.js stats_tests.js
var statsDashboardFS embed.FS

// statsPageCSP lets the dashboard load only its own files and talk only to
// this server. Inline styles are still allowed, for small style attributes.
const statsPageCSP = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"

// StatsDashboardHandler serves the usage and tests dashboard. It needs no
// outside files; it reads /api/stats and related endpoints from this server.
func (s *Server) StatsDashboardHandler(c *gin.Context) {
	c.Header("Content-Security-Policy", statsPageCSP)
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("Cache-Control", "no-store")
	c.Data(http.StatusOK, "text/html; charset=utf-8", statsDashboardHTML)
}

// StatsDashboardFileHandler serves one of the dashboard's styles or scripts.
func (s *Server) StatsDashboardFileHandler(c *gin.Context) {
	f, ok := statsDashboardFiles[c.Param("file")]
	if !ok {
		c.Status(http.StatusNotFound)
		return
	}
	bts, err := statsDashboardFS.ReadFile(f.path)
	if err != nil {
		c.Status(http.StatusNotFound)
		return
	}
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("Cache-Control", "no-store")
	c.Data(http.StatusOK, f.contentType, bts)
}

func (s *Server) StatsHandler(c *gin.Context) {
	c.JSON(http.StatusOK, s.stats.snapshot())
}

// statsOffError is the response when stats are turned off with OLLAMA_NOSTATS.
var statsOffError = gin.H{"error": "usage stats are turned off on this server (OLLAMA_NOSTATS)"}

func (s *Server) ResetStatsHandler(c *gin.Context) {
	if s.stats == nil {
		c.JSON(http.StatusConflict, statsOffError)
		return
	}
	if err := s.stats.reset(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.Status(http.StatusOK)
}

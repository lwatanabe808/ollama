package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

const (
	// statsTagHeader lets any app tag a request, including apps that use the
	// OpenAI or Anthropic formats and can't add a "tag" field to the body.
	statsTagHeader = "X-Ollama-Tag"
	// statsKindHeader set to "test" marks a request from a test tool, so it's
	// logged but not counted as usage.
	statsKindHeader = "X-Ollama-Stats"
	// statsTagKey is where a request's tag is kept on its gin.Context, so it
	// never has to stay on the request body (which may be sent elsewhere).
	statsTagKey = "ollama.stats.tag"

	// The log keeps 90 days of requests, and at most this many, so the file
	// can't grow forever. Each line is about 350 bytes, so a full log is
	// about 17 MB.
	statsLogKeepDays   = 90
	statsLogMaxEntries = 50_000
	// statsLogMaxLine is far more than any real line; longer lines are
	// damage and are skipped.
	statsLogMaxLine = 64 * 1024

	statsLogDefaultLimit = 100
	statsLogMaxLimit     = 1000

	// statsTagsListed caps how many tags /api/stats lists, so the response
	// stays small no matter how many tags exist.
	statsTagsListed = 100
)

// statsLog holds one line per finished request. It lives in memory and is
// written to ~/.ollama/stats-log.jsonl whenever the stats are saved (every
// minute, on shutdown, and after a reset). New lines are appended; the whole
// file is only rewritten after a reset or once it holds many lines that have
// aged out.
type statsLog struct {
	path      string
	entries   []api.RequestLogEntry // oldest first
	pending   []api.RequestLogEntry // added since the last save
	rewrite   bool                  // the file must be rewritten from entries
	fileLines int                   // lines currently in the file

	// byTag keeps the tag list up to date as entries come and go, so
	// building it doesn't mean reading the whole log on every request.
	byTag map[string]*api.TagStats
}

// countTag adds (delta 1) or removes (delta -1) one entry from the tag list.
// Test requests and untagged requests aren't listed.
func (l *statsLog) countTag(e api.RequestLogEntry, delta int64) {
	if e.Tag == "" || e.Test {
		return
	}
	if l.byTag == nil {
		l.byTag = map[string]*api.TagStats{}
	}
	t, ok := l.byTag[e.Tag]
	if !ok {
		if delta < 0 {
			return
		}
		t = &api.TagStats{Tag: e.Tag}
		l.byTag[e.Tag] = t
	}
	t.Requests += delta
	if e.Error {
		t.Errors += delta
	}
	// Entries leave oldest first, so removing one never changes LastUsed.
	if delta > 0 && e.At.After(t.LastUsed) {
		t.LastUsed = e.At
	}
	if t.Requests <= 0 {
		delete(l.byTag, e.Tag)
	}
}

// statsLogPath puts the log next to the totals: stats.json -> stats-log.jsonl.
func statsLogPath(statsPath string) string {
	return strings.TrimSuffix(statsPath, filepath.Ext(statsPath)) + "-log.jsonl"
}

// readLogLine reads one line. A line longer than max is read past and
// reported as too long, so one damaged line can't stop the rest loading.
func readLogLine(rd *bufio.Reader, max int) (line []byte, tooLong bool, err error) {
	for {
		chunk, err := rd.ReadSlice('\n')
		if !tooLong {
			if len(line)+len(chunk) > max {
				tooLong, line = true, nil
			} else {
				line = append(line, chunk...)
			}
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		return line, tooLong, err
	}
}

// loadStatsLog reads the log file. Lines it can't read are skipped (and
// dropped the next time the file is rewritten), so one bad line never loses
// the rest.
func loadStatsLog(path string, now time.Time) statsLog {
	l := statsLog{path: path}

	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return l
	} else if err != nil {
		slog.Warn("could not read usage stats log, starting fresh", "path", path, "error", err)
		return l
	}
	defer f.Close()

	rd := bufio.NewReaderSize(f, 64*1024)
	skipped := 0
	for {
		line, tooLong, err := readLogLine(rd, statsLogMaxLine)
		if line = bytes.TrimSpace(line); len(line) > 0 || tooLong {
			l.fileLines++
			var e api.RequestLogEntry
			if tooLong || json.Unmarshal(line, &e) != nil || e.Model == "" {
				skipped++
			} else {
				l.entries = append(l.entries, e)
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Warn("could not read all of the usage stats log", "path", path, "error", err)
			}
			break
		}
	}
	if skipped > 0 {
		slog.Warn("skipped unreadable lines in the usage stats log", "path", path, "lines", skipped)
		l.rewrite = true
	}

	// Clock changes could leave lines out of order.
	sort.SliceStable(l.entries, func(i, j int) bool { return l.entries[i].At.Before(l.entries[j].At) })
	for _, e := range l.entries {
		l.countTag(e, 1)
	}
	l.trim(now)
	return l
}

func (l *statsLog) add(e api.RequestLogEntry, now time.Time) {
	l.entries = append(l.entries, e)
	l.countTag(e, 1)
	l.pending = append(l.pending, e)
	l.trim(now)
}

// trim drops requests older than the keep window or past the size cap.
func (l *statsLog) trim(now time.Time) {
	cutoff := now.AddDate(0, 0, -statsLogKeepDays)
	drop := 0
	for drop < len(l.entries) && l.entries[drop].At.Before(cutoff) {
		drop++
	}
	if over := len(l.entries) - drop - statsLogMaxEntries; over > 0 {
		drop += over
	}
	if drop > 0 {
		for _, e := range l.entries[:drop] {
			l.countTag(e, -1)
		}
		l.entries = l.entries[drop:]
	}

	// Appending is cheap, so only rewrite once the file holds a lot of
	// lines we no longer keep.
	if l.fileLines+len(l.pending) > len(l.entries)+statsLogMaxEntries/10 {
		l.rewrite = true
	}
}

func (l *statsLog) clear() {
	l.entries = nil
	l.pending = nil
	l.byTag = nil
	l.rewrite = true
}

// tags summarizes the tags in the log, most recently used first.
func (l *statsLog) tags() []api.TagStats {
	out := make([]api.TagStats, 0, len(l.byTag))
	for _, t := range l.byTag {
		out = append(out, *t)
	}
	sort.Slice(out, func(i, j int) bool {
		if !out[i].LastUsed.Equal(out[j].LastUsed) {
			return out[i].LastUsed.After(out[j].LastUsed)
		}
		return out[i].Tag < out[j].Tag
	})
	return out
}

// query returns the requests with the given tag (all requests when tag is
// empty), newest first, plus per-model totals for every match.
func (l *statsLog) query(tag string, limit int) api.StatsLogResponse {
	resp := api.StatsLogResponse{Tag: tag, Models: []api.ModelStats{}, Requests: []api.RequestLogEntry{}}
	matches := func(e api.RequestLogEntry) bool { return tag == "" || e.Tag == tag }

	// Totals are added oldest first, the same way the saved totals are.
	byModel := map[string]*api.ModelStats{}
	for _, e := range l.entries {
		if !matches(e) {
			continue
		}
		resp.Total++
		m, ok := byModel[e.Model]
		if !ok {
			m = &api.ModelStats{Model: e.Model}
			byModel[e.Model] = m
		}
		addRequest(m, e.RequestStats, e.Error)
	}
	resp.Models = sortedStats(byModel)

	for i := len(l.entries) - 1; i >= 0 && len(resp.Requests) < limit; i-- {
		if matches(l.entries[i]) {
			resp.Requests = append(resp.Requests, l.entries[i])
		}
	}
	return resp
}

// saveLog writes new log lines to disk. It appends when it can and rewrites
// the whole file (safely, through a temp file) only when needed.
func (u *usageStats) saveLog() error {
	u.mu.Lock()
	l := &u.log
	if !l.rewrite && len(l.pending) == 0 {
		u.mu.Unlock()
		return nil
	}
	rewrite := l.rewrite
	pending := l.pending
	lines := pending
	if rewrite {
		lines = slices.Clone(l.entries)
	}
	l.pending = nil
	l.rewrite = false
	u.mu.Unlock()

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, e := range lines {
		if err := enc.Encode(e); err != nil {
			return err
		}
	}

	var err error
	if rewrite {
		err = writeFileAtomic(l.path, buf.Bytes())
	} else {
		err = appendFile(l.path, buf.Bytes())
	}

	u.mu.Lock()
	defer u.mu.Unlock()
	if err != nil {
		// Try again on the next save. A rewrite writes every entry anyway,
		// so its pending lines aren't needed; otherwise keep them, capped,
		// so a disk that stays full can't make this list grow forever.
		if rewrite {
			l.rewrite = true
		} else if l.rewrite {
			l.pending = nil
		} else {
			l.pending = append(pending, l.pending...)
			if over := len(l.pending) - statsLogMaxEntries; over > 0 {
				l.pending = l.pending[over:]
			}
		}
		return err
	}
	if rewrite {
		l.fileLines = len(lines)
	} else {
		l.fileLines += len(lines)
	}
	return nil
}

func appendFile(path string, bts []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(bts); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// takeStatsTag reads a request's tag from the body's tag field, or else the
// X-Ollama-Tag header, and keeps it on the gin.Context for statsInfoFor. It
// always clears the body field, so the tag is never sent on to cloud models
// or remote hosts. A bad tag in the body fails the request, since the caller
// asked for it explicitly; a bad header is ignored, so an app's label can't
// break its chats. With stats off, tags are ignored. It returns false if it
// already sent an error response.
func (s *Server) takeStatsTag(c *gin.Context, bodyTag *string) bool {
	body := strings.TrimSpace(*bodyTag)
	*bodyTag = ""
	if s.stats == nil {
		return true
	}
	if body != "" {
		if err := api.ValidateTag(body); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return false
		}
		c.Set(statsTagKey, body)
		return true
	}
	if header := strings.TrimSpace(c.GetHeader(statsTagHeader)); header != "" {
		if err := api.ValidateTag(header); err != nil {
			slog.Debug("ignoring an invalid X-Ollama-Tag header", "error", err)
		} else {
			c.Set(statsTagKey, header)
		}
	}
	return true
}

// statsIsTest reports whether a request was sent by a test tool.
func statsIsTest(c *gin.Context) bool {
	return c.Request != nil && strings.EqualFold(strings.TrimSpace(c.GetHeader(statsKindHeader)), "test")
}

// StatsLogHandler returns logged requests, optionally only those with a tag.
func (s *Server) StatsLogHandler(c *gin.Context) {
	tag := strings.TrimSpace(c.Query("tag"))
	if tag != "" {
		if err := api.ValidateTag(tag); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}

	limit := statsLogDefaultLimit
	if v := c.Query("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "limit must be a positive number"})
			return
		}
		limit = min(n, statsLogMaxLimit)
	}

	c.JSON(http.StatusOK, s.stats.logQuery(tag, limit))
}

package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

// Cloud models run on Ollama's servers: Ollama passes the request through
// and copies the answer back as raw bytes. The cloud meter watches those
// bytes go by, the same way observe watches a local model's answer, and
// records the request once the answer ends.
//
// Times are measured on this computer, so the wait before the first word
// includes the trip over the network. Token counts and writing time come
// from the cloud's own numbers. Loading and memory don't apply to cloud.
//
// Only Ollama's own chat and generate endpoints are metered. The OpenAI and
// Anthropic compatible endpoints pass cloud requests straight through in
// their own formats, which aren't read here.

const (
	statsCloudMeterKey = "ollama.stats.cloud"

	// statsCloudMaxLine caps how much of one line the meter keeps. The final
	// line with the numbers is small; a longer line is an answer chunk, and
	// is skipped.
	statsCloudMaxLine = 1 << 20
)

type cloudMeter struct {
	u     *usageStats
	model string
	info  statsRequestInfo
	start time.Time

	line    []byte
	tooLong bool

	done    bool
	failed  bool
	metrics api.Metrics
}

// meterCloud asks the cloud proxy to measure this request. model is the name
// the caller used, like "gpt-oss:120b-cloud", so cloud models never share
// totals with a local model of the same base name.
func (s *Server) meterCloud(c *gin.Context, model string) {
	if s.stats == nil {
		return
	}
	info := statsRequestInfo{Tag: c.GetString(statsTagKey), Test: statsIsTest(c), Cloud: true}
	c.Set(statsCloudMeterKey, &cloudMeter{u: s.stats, model: model, info: info})
}

func cloudMeterFrom(c *gin.Context) *cloudMeter {
	v, ok := c.Get(statsCloudMeterKey)
	if !ok {
		return nil
	}
	m, _ := v.(*cloudMeter)
	return m
}

func (m *cloudMeter) begin() {
	m.start = time.Now()
	m.u.active.Add(1)
}

// writer wraps the response so the meter sees every byte on its way out.
func (m *cloudMeter) writer(w http.ResponseWriter) http.ResponseWriter {
	return &cloudMeterWriter{ResponseWriter: w, m: m}
}

// scan splits the answer into lines. Streamed answers are one JSON object
// per line; a non-streamed answer is a single object.
func (m *cloudMeter) scan(p []byte) {
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			m.keep(p)
			return
		}
		m.keep(p[:i])
		m.endLine()
		p = p[i+1:]
	}
}

func (m *cloudMeter) keep(p []byte) {
	if m.tooLong {
		return
	}
	if len(m.line)+len(p) > statsCloudMaxLine {
		m.tooLong, m.line = true, nil
		return
	}
	m.line = append(m.line, p...)
}

// endLine reads a finished line, but only if it could be the last one:
// "done":true, or an error. Answer chunks are skipped without parsing.
func (m *cloudMeter) endLine() {
	line := bytes.TrimSpace(m.line)
	m.line, m.tooLong = m.line[:0], false
	if !bytes.Contains(line, []byte(`"done":true`)) && !bytes.Contains(line, []byte(`"error"`)) {
		return
	}
	var v struct {
		Done  bool            `json:"done"`
		Error json.RawMessage `json:"error"`
		api.Metrics
	}
	if json.Unmarshal(line, &v) != nil {
		return
	}
	if len(v.Error) > 0 && string(v.Error) != "null" {
		m.failed = true
		return
	}
	if v.Done {
		m.done = true
		m.metrics = v.Metrics
	}
}

// finish records the request once the answer has been copied. status is the
// cloud's HTTP status, or 0 if the request never reached the cloud.
//
// Like local requests, a request the caller stopped isn't counted, and
// neither is one turned away before it reached a model (not signed in,
// cloud turned off, unknown model: 4xx). A server error, an error in the
// answer, or an answer that ends without finishing counts as an error.
func (m *cloudMeter) finish(c *gin.Context, status int) {
	defer m.u.active.Add(-1)
	m.endLine()
	if c.Request.Context().Err() != nil || status == 0 || (status >= 400 && status < 500) {
		return
	}

	if m.done && !m.failed && status < 300 {
		metrics := m.metrics
		metrics.TotalDuration = time.Since(m.start)
		metrics.LoadDuration = 0
		m.u.record(m.model, metrics, m.info, false)
		return
	}
	m.u.record(m.model, api.Metrics{}, m.info, true)
}

type cloudMeterWriter struct {
	http.ResponseWriter
	m *cloudMeter
}

func (w *cloudMeterWriter) Write(p []byte) (int, error) {
	w.m.scan(p)
	return w.ResponseWriter.Write(p)
}

func (w *cloudMeterWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

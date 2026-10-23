package cmd

import (
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// openBrowser is a variable so tests can check the URL without opening a browser.
var openBrowser = launch.OpenBrowser

func StatsHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	tag, _ := cmd.Flags().GetString("tag")
	tag = strings.TrimSpace(tag)
	if cmd.Flags().Changed("tag") && tag == "" {
		return errors.New("--tag needs a tag name, for example: ollama stats --tag cloud-test")
	}
	if cmd.Flags().Changed("limit") && tag == "" {
		return errors.New("--limit only works with --tag")
	}

	if reset, _ := cmd.Flags().GetBool("reset"); reset {
		if err := client.ResetStats(cmd.Context()); err != nil {
			return statsError(err)
		}
		fmt.Println("Usage stats reset.")
		return nil
	}

	if web, _ := cmd.Flags().GetBool("web"); web {
		// Check first, so an old server or stats being off is reported here
		// instead of as a broken page.
		st, err := client.Stats(cmd.Context())
		if err != nil {
			return statsError(err)
		}
		if st.Disabled {
			fmt.Println(statsOffMessage)
			return nil
		}
		dashboard := dashboardURL(envconfig.Host())
		fmt.Printf("Opening the stats dashboard at %s\n", dashboard)
		openBrowser(dashboard)
		return nil
	}

	if tag != "" {
		if err := api.ValidateTag(tag); err != nil {
			return fmt.Errorf("--tag: %w", err)
		}
		limit, _ := cmd.Flags().GetInt("limit")
		if limit < 1 {
			return errors.New("--limit must be at least 1")
		}
		log, err := client.StatsLog(cmd.Context(), tag, limit)
		if err != nil {
			return statsError(err)
		}
		if log.Disabled {
			fmt.Println(statsOffMessage)
			return nil
		}
		renderTagLog(os.Stdout, log)
		return nil
	}

	st, err := client.Stats(cmd.Context())
	if err != nil {
		return statsError(err)
	}
	if st.Disabled {
		fmt.Println(statsOffMessage)
		return nil
	}

	renderStats(os.Stdout, st)
	return nil
}

const statsOffMessage = "Usage stats are turned off on this server (OLLAMA_NOSTATS is set)."

// dashboardURL is the dashboard's address. A server listening on every
// address (0.0.0.0 or ::) can't be opened as is, so this computer's own
// address is used instead.
func dashboardURL(host *url.URL) string {
	u := *host
	switch u.Hostname() {
	case "0.0.0.0", "::":
		u.Host = net.JoinHostPort("127.0.0.1", u.Port())
	}
	return u.JoinPath("stats").String()
}

// renderTable prints rows in the same plain style as 'ollama list'.
func renderTable(w io.Writer, header []string, rows [][]string) {
	table := tablewriter.NewWriter(w)
	table.SetHeader(header)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetAutoFormatHeaders(false)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("    ")
	table.AppendBulk(rows)
	table.Render()
}

// renderTagLog shows the totals for one tag, then each tagged request.
func renderTagLog(w io.Writer, log *api.StatsLogResponse) {
	fmt.Fprintf(w, "Requests tagged %q: %s\n", log.Tag, format.HumanNumber(uint64(log.Total)))
	if log.Total == 0 {
		// Tags can't contain quotes, so wrapping one with spaces is safe.
		shellTag := log.Tag
		if strings.Contains(shellTag, " ") {
			shellTag = `"` + shellTag + `"`
		}
		fmt.Fprintf(w, "\nNo requests with this tag yet. Tag one with:\n  ollama run MODEL \"your prompt\" --tag %s\n", shellTag)
		return
	}

	fmt.Fprintf(w, "\nBY MODEL\n")
	renderStatsTable(w, log.Models)
	renderContextNotes(w, log.Models)
	renderTimeBreakdown(w, log.Models)

	fmt.Fprintf(w, "\nREQUESTS (newest first)\n")
	var data [][]string
	for _, e := range log.Requests {
		status, speed, wait, total, split := "error", "-", "-", "-", "-"
		if !e.Error {
			status = "ok"
			speed = formatSpeed(e.TokensPerSecond(), " tok/s")
			wait = statsDuration(e.Wait())
			total = statsDuration(e.TotalDuration)
			split = statsDuration(e.LoadDuration) + " / " + statsDuration(e.PromptDuration) + " / " + statsDuration(e.OutputDuration)
		}
		data = append(data, []string{
			e.At.Local().Format("Jan 2 15:04:05"),
			e.Model,
			wait,
			speed,
			total,
			split,
			format.HumanNumber(uint64(e.PromptTokens)),
			format.HumanNumber(uint64(e.PromptNewTokens())),
			format.HumanNumber(uint64(e.PromptCachedTokens)),
			format.HumanNumber(uint64(e.OutputTokens)),
			status,
		})
	}
	renderTable(w, []string{"TIME", "MODEL", "FIRST WORD", "SPEED", "TIME TAKEN", "LOAD / READ / WRITE", "TOTAL IN", "NEWLY READ", "REUSED", "OUT", "STATUS"}, data)

	if shown := len(log.Requests); shown < log.Total {
		fmt.Fprintf(w, "\nShowing the newest %d of %d. Use --limit to see more.\n", shown, log.Total)
	}
}

// statsError gives a clearer message when the server is too old to have
// stats, or has them turned off.
func statsError(err error) error {
	var serr api.StatusError
	if errors.As(err, &serr) {
		switch serr.StatusCode {
		case http.StatusNotFound:
			return errors.New("this Ollama server does not support usage stats. Update Ollama and restart it")
		case http.StatusConflict:
			return errors.New(statsOffMessage)
		}
	}
	return err
}

func renderStats(w io.Writer, st *api.StatsResponse) {
	header := "Ollama usage stats"
	if !st.Since.IsZero() {
		header += "    tracking since " + st.Since.Local().Format("Jan 2, 2006")
	}
	switch st.Active {
	case 0:
	case 1:
		header += "    1 request running now"
	default:
		header += fmt.Sprintf("    %d requests running now", st.Active)
	}
	fmt.Fprintln(w, header)

	fmt.Fprintf(w, "\nTODAY\n")
	if len(st.TodayModels) == 0 {
		fmt.Fprintln(w, "No requests yet today.")
	} else {
		renderStatsTable(w, st.TodayModels)
	}

	fmt.Fprintf(w, "\nALL TIME\n")
	if len(st.AllTimeModels) == 0 {
		fmt.Fprintln(w, "No requests yet. Chat with a model and its stats will show up here.")
	} else {
		renderStatsTable(w, st.AllTimeModels)
		renderContextNotes(w, st.AllTimeModels)
	}

	renderTimeBreakdown(w, st.AllTimeModels)
	renderLastRequests(w, st.AllTimeModels)
	renderTags(w, st.Tags, st.TagsTotal)

	if len(st.AllTimeModels) > 1 {
		var total api.ModelStats
		for _, m := range st.AllTimeModels {
			total.Requests += m.Requests
			total.Errors += m.Errors
			total.OutputTokens += m.OutputTokens
		}
		fmt.Fprintf(w, "\nTotal: %s requests, %s errors, %s tokens written.\n",
			format.HumanNumber(uint64(total.Requests)),
			format.HumanNumber(uint64(total.Errors)),
			format.HumanNumber(uint64(total.OutputTokens)))
	}
}

func renderStatsTable(w io.Writer, models []api.ModelStats) {
	var data [][]string
	for _, m := range models {
		// Memory the model needed when it last ran. Saved, so it shows
		// even after the model is unloaded.
		mem := "-"
		if m.MemorySize > 0 {
			mem = format.HumanBytes(m.MemorySize)
		}
		data = append(data, []string{
			m.Model,
			format.HumanNumber(uint64(m.Requests)),
			format.HumanNumber(uint64(m.Errors)),
			statsDuration(m.AvgWait()),
			formatSpeed(m.TokensPerSecond(), " tok/s"),
			statsDuration(m.AvgDuration()),
			format.HumanNumber(uint64(m.PromptTokens)),
			format.HumanNumber(uint64(m.PromptNewTokens())),
			format.HumanNumber(uint64(m.PromptCachedTokens)),
			format.HumanNumber(uint64(m.OutputTokens)),
			statsContext(m),
			mem,
			format.HumanTime(m.LastUsed, "-"),
		})
	}

	renderTable(w, []string{"MODEL", "REQUESTS", "ERRORS", "AVG FIRST WORD", "AVG SPEED", "AVG TIME", "TOTAL TOKENS IN", "NEWLY READ", "REUSED", "TOTAL TOKENS OUT", "MAX CONTEXT", "MEMORY NEEDED", "LAST USED"}, data)
}

// renderTimeBreakdown splits each model's average request time into
// loading the model, reading the prompt, and writing the answer, and shows
// how much of the prompt was reused and where the model ran.
func renderTimeBreakdown(w io.Writer, models []api.ModelStats) {
	var data [][]string
	var cpuNotes []string
	for _, m := range models {
		if m.Requests-m.Errors <= 0 {
			continue
		}
		reused := "-"
		if share, ok := m.PromptReused(); ok {
			reused = fmt.Sprintf("%.0f%%", share*100)
		}
		data = append(data, []string{
			m.Model,
			statsDuration(m.AvgLoad()),
			statsDuration(m.AvgPromptRead()),
			statsDuration(m.AvgWriting()),
			reused,
			statsProcessor(m),
		})
		if share, ok := m.GPUShare(); ok && share < 1 {
			cpuNotes = append(cpuNotes, fmt.Sprintf("Note: %s is running %.0f%% on the CPU, which is much slower than the GPU. A smaller model or a smaller context size may fit fully on the GPU.", m.Model, (1-share)*100))
		}
	}
	if len(data) == 0 {
		return
	}

	fmt.Fprintf(w, "\nWHERE THE TIME GOES (average per request)\n")
	renderTable(w, []string{"MODEL", "LOADING MODEL", "READING PROMPT", "WRITING ANSWER", "PROMPT REUSED", "PROCESSOR"}, data)
	for _, n := range cpuNotes {
		fmt.Fprintf(w, "\n%s\n", n)
	}
}

// statsProcessor shows where the model ran, the same way 'ollama ps' does.
func statsProcessor(m api.ModelStats) string {
	share, ok := m.GPUShare()
	switch {
	case !ok:
		return "-"
	case share >= 1:
		return "100% GPU"
	case share <= 0:
		return "100% CPU"
	default:
		cpu := math.Round((1 - share) * 100)
		return fmt.Sprintf("%.0f%%/%.0f%% CPU/GPU", cpu, 100-cpu)
	}
}

// statsTagsShown is how many tags the terminal lists before summing up the rest.
const statsTagsShown = 10

// renderTags lists the most recent tags, with a hint on how to see one.
func renderTags(w io.Writer, tags []api.TagStats, total int) {
	if len(tags) == 0 {
		return
	}
	total = max(total, len(tags))
	fmt.Fprintf(w, "\nTAGS (most recent first)\n")
	var data [][]string
	for _, t := range tags[:min(len(tags), statsTagsShown)] {
		data = append(data, []string{
			t.Tag,
			format.HumanNumber(uint64(t.Requests)),
			format.HumanNumber(uint64(t.Errors)),
			format.HumanTime(t.LastUsed, "-"),
		})
	}
	renderTable(w, []string{"TAG", "REQUESTS", "ERRORS", "LAST USED"}, data)
	if more := total - statsTagsShown; more > 0 {
		fmt.Fprintf(w, "...and %d more. ", more)
	}
	fmt.Fprintf(w, "See one tag's requests with: ollama stats --tag NAME\n")
}

// renderLastRequests shows each model's most recent request next to its
// all-time average, so you can tell if the last answer was typical.
func renderLastRequests(w io.Writer, models []api.ModelStats) {
	var data [][]string
	for _, m := range models {
		last := m.Last
		if last == nil {
			continue
		}

		ok := m.Requests - m.Errors
		avgIn, avgOut := int64(0), int64(0)
		if ok > 0 {
			avgIn, avgOut = m.PromptTokens/ok, m.OutputTokens/ok
		}

		lastSpeed := formatSpeed(last.TokensPerSecond(), " tok/s")
		avgSpeed := formatSpeed(m.TokensPerSecond(), "")

		data = append(data, []string{
			m.Model,
			format.HumanTime(last.At, "-"),
			fmt.Sprintf("%s (avg %s)", statsDuration(last.Wait()), statsDuration(m.AvgWait())),
			fmt.Sprintf("%s (avg %s)", lastSpeed, avgSpeed),
			fmt.Sprintf("%s (avg %s)", statsDuration(last.TotalDuration), statsDuration(m.AvgDuration())),
			fmt.Sprintf("%s / %s (avg %s / %s)",
				format.HumanNumber(uint64(last.PromptTokens)), format.HumanNumber(uint64(last.OutputTokens)),
				format.HumanNumber(uint64(avgIn)), format.HumanNumber(uint64(avgOut))),
			fmt.Sprintf("%s (max %s)",
				format.HumanNumber(uint64(last.ContextUsed())), format.HumanNumber(uint64(m.MaxContextUsed))),
		})
	}
	if len(data) == 0 {
		return
	}

	fmt.Fprintf(w, "\nLAST REQUEST (compared to all-time average)\n")
	renderTable(w, []string{"MODEL", "WHEN", "FIRST WORD", "SPEED", "TIME", "TOKENS IN/OUT", "CONTEXT"}, data)
}

// statsContextWarnAt is how full the context window has to get before we warn.
const statsContextWarnAt = 0.9

// statsContext shows the most context one request used, out of the model's
// context window, like "5K / 32K (16%)".
func statsContext(m api.ModelStats) string {
	usedTokens, window := m.MaxContextFill()
	if usedTokens == 0 {
		return "-"
	}
	used := format.HumanNumber(uint64(usedTokens))
	if window <= 0 {
		return used
	}
	pct := float64(usedTokens) / float64(window) * 100
	return fmt.Sprintf("%s / %s (%.0f%%)", used, contextSize(window), pct)
}

// renderContextNotes warns about models whose fullest request nearly filled
// its context window. When that happens, older parts of a chat get cut off
// without any error.
func renderContextNotes(w io.Writer, models []api.ModelStats) {
	for _, m := range models {
		used, window := m.MaxContextFill()
		if window > 0 && float64(used) >= statsContextWarnAt*float64(window) {
			fmt.Fprintf(w, "\nNote: %s used %s of its %s context window. In long chats, older messages may get cut off.\n",
				m.Model, format.HumanNumber(uint64(used)), contextSize(window))
		}
	}
}

// contextSize shows context window sizes the way people talk about them:
// 32768 tokens is "32K", not "33K".
func contextSize(n int64) string {
	if n >= 1024 && n%1024 == 0 {
		return fmt.Sprintf("%dK", n/1024)
	}
	return format.HumanNumber(uint64(n))
}

// formatSpeed shows tokens per second with one decimal, or "-" if unknown.
func formatSpeed(tps float64, unit string) string {
	if tps <= 0 {
		return "-"
	}
	return fmt.Sprintf("%.1f%s", tps, unit)
}

func statsDuration(d time.Duration) string {
	switch {
	case d <= 0:
		return "-"
	case d < 100*time.Millisecond:
		return fmt.Sprintf("%dms", d.Milliseconds())
	case d < time.Minute:
		return fmt.Sprintf("%.1fs", d.Seconds())
	default:
		return fmt.Sprintf("%dm %ds", int(d.Minutes()), int(d.Seconds())%60)
	}
}

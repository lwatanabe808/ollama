package server

import (
	"fmt"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// fullStats fills the log to its cap, to measure work that grows with it.
func fullStats(b *testing.B) *usageStats {
	now := time.Now()
	u := newUsageStats(b.TempDir() + "/stats.json")
	u.now = func() time.Time { return now }
	for i := range statsLogMaxEntries {
		u.record(fmt.Sprintf("model-%d", i%5), api.Metrics{EvalCount: 10, EvalDuration: time.Second, TotalDuration: time.Second},
			statsRequestInfo{Tag: fmt.Sprintf("tag-%d", i%300)}, false)
	}
	return u
}

func BenchmarkSnapshotFullLog(b *testing.B) {
	u := fullStats(b)
	b.ResetTimer()
	for b.Loop() {
		u.snapshot()
	}
}

func BenchmarkLogQueryFullLog(b *testing.B) {
	u := fullStats(b)
	b.ResetTimer()
	for b.Loop() {
		u.logQuery("tag-7", 200)
	}
}

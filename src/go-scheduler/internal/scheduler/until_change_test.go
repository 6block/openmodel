package scheduler

import (
	"context"
	"testing"

	"openmodel/go-scheduler/internal/lotus"
)

// After a deadline's proof completes while the deadline is still open, the
// scheduler stays AVAILABLE. That decision must not report a countdown of 0:
// the gateway reads 0 as "graceful yield starts now" and cut the worker to 5% routing
// weight for the rest of the deadline (seen 2026-11-09 after a real WindowPoSt).
func TestAvailableAfterProofReportsUnknownHorizon(t *testing.T) {
	s := New(&flakyLotus{}, *testPolicy(), testLogger())
	info := &lotus.DeadlineInfo{CurrentEpoch: 1005, PeriodStart: 1000, Index: 0, Open: 1000, Close: 1060}
	s.markProofCompleted(info.PeriodStart, info.Index)
	s.evaluateAndApply(context.Background(), info)
	d := s.LatestDecision()
	if d == nil || d.State != StateAvailable {
		t.Fatalf("want AVAILABLE after the proof, got %+v", d)
	}
	if d.SecondsUntilNextChange >= 0 && d.SecondsUntilNextChange < 60 {
		t.Fatalf("available-after-proof must not claim an imminent yield, got %d s", d.SecondsUntilNextChange)
	}
}

// Every AVAILABLE decision built outside the WindowPoSt policy carries the
// unknown marker, never a zero that reads as "yield now".
func TestWinningPostAvailableDecisionsHaveUnknownHorizon(t *testing.T) {
	p := testPolicy()
	for name, d := range map[string]YieldDecision{
		"not elected": p.EvaluateWinningPost(&lotus.MinerBaseInfo{EligibleForMining: false}),
		"nil info":    p.EvaluateWinningPost(nil),
	} {
		if d.SecondsUntilNextChange != untilChangeUnknown {
			t.Errorf("%s: SecondsUntilNextChange = %d, want unknown (-1)", name, d.SecondsUntilNextChange)
		}
	}
}

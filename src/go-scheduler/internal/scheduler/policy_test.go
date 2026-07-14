package scheduler

import (
	"testing"

	"openmodel/go-scheduler/internal/config"
	"openmodel/go-scheduler/internal/lotus"
	pb "openmodel/go-scheduler/proto/sidecar"
)

func testPolicy() *YieldPolicy {
	return &YieldPolicy{
		WindowPost: config.WindowPostPolicy{
			GracefulYieldThresholdSec: 300, // -5 min
			HardStopThresholdSec:      60,
			ResumeDelayAfterCloseSec:  30,
		},
		WinningPost: config.WinningPostPolicy{Enabled: true, TimeoutSec: 45},
	}
}

// dl builds a DeadlineInfo with `openIn` epochs until the window opens.
func dl(currentEpoch, open, close int64) *lotus.DeadlineInfo {
	return &lotus.DeadlineInfo{CurrentEpoch: currentEpoch, Open: open, Close: close}
}

func TestEvaluateWindowPost_SafeZone(t *testing.T) {
	d := testPolicy().EvaluateWindowPost(dl(0, 100, 120)) // 3000s to open > 300
	if d.State != StateAvailable {
		t.Errorf("state = %v, want Available", d.State)
	}
	if d.Urgency != pb.YieldUrgency_YIELD_URGENCY_NORMAL {
		t.Errorf("urgency = %v, want NORMAL", d.Urgency)
	}
	if d.Reason != pb.YieldReason_YIELD_REASON_RESUME {
		t.Errorf("reason = %v, want RESUME", d.Reason)
	}
}

func TestEvaluateWindowPost_GracefulZone(t *testing.T) {
	d := testPolicy().EvaluateWindowPost(dl(0, 8, 20)) // 240s in (60,300]
	if d.State != StateYielding {
		t.Errorf("state = %v, want Yielding (graceful)", d.State)
	}
	if d.Urgency != pb.YieldUrgency_YIELD_URGENCY_NORMAL {
		t.Errorf("graceful urgency = %v, want NORMAL (drain)", d.Urgency)
	}
	if d.Reason != pb.YieldReason_YIELD_REASON_WINDOW_POST_APPROACHING {
		t.Errorf("reason = %v, want APPROACHING", d.Reason)
	}
}

func TestEvaluateWindowPost_HardStopZone(t *testing.T) {
	d := testPolicy().EvaluateWindowPost(dl(0, 2, 20)) // 60s <= 60 hard stop
	if d.State != StateWindowPost {
		t.Errorf("state = %v, want WindowPost (hard stop)", d.State)
	}
	if d.Urgency != pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE {
		t.Errorf("hard-stop urgency = %v, want IMMEDIATE", d.Urgency)
	}
}

func TestEvaluateWindowPost_Active(t *testing.T) {
	d := testPolicy().EvaluateWindowPost(dl(5, 0, 10)) // CurrentEpoch in [Open,Close)
	if d.State != StateWindowPost {
		t.Errorf("state = %v, want WindowPost (active)", d.State)
	}
	if d.Urgency != pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE {
		t.Errorf("active urgency = %v, want IMMEDIATE", d.Urgency)
	}
	if d.Reason != pb.YieldReason_YIELD_REASON_WINDOW_POST_ACTIVE {
		t.Errorf("reason = %v, want ACTIVE", d.Reason)
	}
	// secondsUntilClose (5 epochs=150) + ResumeDelay (30) = 180
	if d.SecondsUntilNextChange != 180 {
		t.Errorf("SecondsUntilNextChange = %d, want 180", d.SecondsUntilNextChange)
	}
}

func TestEvaluateWindowPost_FailSafeOnDisconnect(t *testing.T) {
	p := testPolicy()
	p.FailSafeOnDisconnect = true
	d := p.EvaluateWindowPost(nil)
	if d.State != StateWindowPost || d.Urgency != pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE {
		t.Errorf("fail-safe must IMMEDIATE-yield to WindowPost, got state=%v urgency=%v", d.State, d.Urgency)
	}
	if d.Reason != pb.YieldReason_YIELD_REASON_LOTUS_DISCONNECTED {
		t.Errorf("reason = %v, want LOTUS_DISCONNECTED", d.Reason)
	}
}

func TestEvaluateWindowPost_NoFailSafe(t *testing.T) {
	p := testPolicy()
	p.FailSafeOnDisconnect = false
	d := p.EvaluateWindowPost(nil)
	if d.State != StateAvailable || d.Urgency != pb.YieldUrgency_YIELD_URGENCY_NORMAL {
		t.Errorf("no-fail-safe should stay Available/NORMAL, got state=%v urgency=%v", d.State, d.Urgency)
	}
}

// TestEvaluateWindowPost_Boundaries guards the threshold comparisons (off-by-one).
func TestEvaluateWindowPost_Boundaries(t *testing.T) {
	p := testPolicy()
	if got := p.EvaluateWindowPost(dl(0, 10, 50)).State; got != StateYielding { // exactly 300s
		t.Errorf("at graceful boundary (300s): state = %v, want Yielding", got)
	}
	if got := p.EvaluateWindowPost(dl(0, 11, 50)).State; got != StateAvailable { // 330s
		t.Errorf("just above graceful (330s): state = %v, want Available", got)
	}
	if got := p.EvaluateWindowPost(dl(0, 2, 50)).State; got != StateWindowPost { // exactly 60s
		t.Errorf("at hard-stop boundary (60s): state = %v, want WindowPost", got)
	}
}

func TestEvaluateWinningPost_Eligible(t *testing.T) {
	d := testPolicy().EvaluateWinningPost(&lotus.MinerBaseInfo{EligibleForMining: true})
	if d.State != StateWinningPost || d.Urgency != pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE {
		t.Errorf("eligible must IMMEDIATE-yield to WinningPost, got state=%v urgency=%v", d.State, d.Urgency)
	}
	if d.Reason != pb.YieldReason_YIELD_REASON_WINNING_POST {
		t.Errorf("reason = %v, want WINNING_POST", d.Reason)
	}
	if d.SecondsUntilNextChange != 45 {
		t.Errorf("SecondsUntilNextChange = %d, want 45 (TimeoutSec)", d.SecondsUntilNextChange)
	}
}

func TestEvaluateWinningPost_NotElected(t *testing.T) {
	d := testPolicy().EvaluateWinningPost(&lotus.MinerBaseInfo{EligibleForMining: false})
	if d.State != StateAvailable {
		t.Errorf("not-elected state = %v, want Available", d.State)
	}
}

func TestEvaluateWinningPost_Disabled(t *testing.T) {
	p := testPolicy()
	p.WinningPost.Enabled = false
	// even though eligible, disabled policy must not yield
	d := p.EvaluateWinningPost(&lotus.MinerBaseInfo{EligibleForMining: true})
	if d.State != StateAvailable {
		t.Errorf("disabled WinningPost must stay Available, got %v", d.State)
	}
}

func TestEvaluateWinningPost_NilBaseInfo(t *testing.T) {
	if d := testPolicy().EvaluateWinningPost(nil); d.State != StateAvailable {
		t.Errorf("nil baseInfo state = %v, want Available", d.State)
	}
}

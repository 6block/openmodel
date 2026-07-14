package scheduler

import (
	"testing"

	pb "openmodel/go-scheduler/proto/sidecar"
)

// S4 regression: the GPU state must default to UNKNOWN (conservative), not
// AVAILABLE — otherwise inference connecting before the first Lotus evaluation
// would load the model and contend with active mining.
func TestStateMachineStartsUnknown(t *testing.T) {
	sm := NewStateMachine()
	if sm.Current() != StateUnknown {
		t.Fatalf("S4 regression: state machine starts in %v, want UNKNOWN", sm.Current())
	}
}

func TestSchedulerStartsUnknown(t *testing.T) {
	s := New(&flakyLotus{}, YieldPolicy{}, testLogger())
	if s.CurrentState() != StateUnknown {
		t.Fatalf("S4 regression: scheduler starts in %v, want UNKNOWN", s.CurrentState())
	}
	if s.CurrentState() == pb.GpuState_GPU_STATE_AVAILABLE {
		t.Fatal("scheduler must not advertise AVAILABLE before first evaluation")
	}
}

func TestFirstDecisionTransitionsFromUnknown(t *testing.T) {
	s := New(&flakyLotus{}, YieldPolicy{}, testLogger())
	s.applyDecision(&YieldDecision{State: StateAvailable})
	if s.CurrentState() != StateAvailable {
		t.Fatalf("expected AVAILABLE after first decision, got %v", s.CurrentState())
	}
}

// TestTransitionUnless verifies the atomic guarded transition that backs the
// WINNING_POST-clobber fix.
func TestTransitionUnless(t *testing.T) {
	sm := NewStateMachine()
	if !sm.TransitionUnless(StateAvailable, StateWinningPost) {
		t.Fatal("expected UNKNOWN -> AVAILABLE (not blocked)")
	}
	sm.Transition(StateWinningPost)
	if sm.TransitionUnless(StateAvailable, StateWinningPost) {
		t.Fatal("guarded transition must be refused while in WINNING_POST")
	}
	if sm.Current() != StateWinningPost {
		t.Fatalf("state clobbered despite guard: %v", sm.Current())
	}
	// An unconditional transition (the winning timer's own resume) still works.
	if !sm.Transition(StateAvailable) {
		t.Fatal("unconditional transition should still apply")
	}
}

// TestProofResumeDoesNotClobberWinningPost is the regression for the audit HIGH:
// a WindowPoSt-path resume (proof monitor / checkWindowPost stray AVAILABLE) must
// never overwrite an active WINNING_POST and pull inference back onto the GPU during
// block production.
func TestProofResumeDoesNotClobberWinningPost(t *testing.T) {
	s := New(&flakyLotus{}, YieldPolicy{}, testLogger())

	// Active WinningPoSt yield.
	s.applyDecision(&YieldDecision{State: StateWinningPost, Reason: pb.YieldReason_YIELD_REASON_WINNING_POST})
	if s.CurrentState() != StateWinningPost {
		t.Fatalf("setup: want WINNING_POST, got %v", s.CurrentState())
	}

	// A WindowPoSt-path resume must be refused while WINNING_POST is active.
	s.applyDecisionUnlessWinning(&YieldDecision{State: StateAvailable, Reason: pb.YieldReason_YIELD_REASON_RESUME})
	if s.CurrentState() != StateWinningPost {
		t.Fatalf("HIGH regression: WINNING_POST clobbered by WindowPoSt-path resume -> %v", s.CurrentState())
	}

	// WinningPoSt's own timer resume (unconditional) does apply.
	s.applyDecision(&YieldDecision{State: StateAvailable, Reason: pb.YieldReason_YIELD_REASON_RESUME})
	if s.CurrentState() != StateAvailable {
		t.Fatalf("want AVAILABLE after WinningPoSt's own resume, got %v", s.CurrentState())
	}

	// From a non-winning state, a guarded resume DOES apply (normal WindowPoSt end).
	s.applyDecision(&YieldDecision{State: StateWindowPost})
	s.applyDecisionUnlessWinning(&YieldDecision{State: StateAvailable, Reason: pb.YieldReason_YIELD_REASON_RESUME})
	if s.CurrentState() != StateAvailable {
		t.Fatalf("guarded resume from WINDOW_POST should apply, got %v", s.CurrentState())
	}
}

// TestNoopDecisionRefreshesLatest is the B1 staleness regression: every 15s poll in
// steady AVAILABLE produces an AVAILABLE→AVAILABLE no-op carrying the live WindowPoSt
// countdown. The old code discarded no-ops entirely, so /ready's seconds_until_change
// froze at whatever the LAST state change stored (a resume decision carries none → the
// field read 0 for hours on a real miner) and the gateway's predictive de-prioritization
// never saw a live countdown while the worker was servable.
func TestNoopDecisionRefreshesLatest(t *testing.T) {
	s := New(&flakyLotus{}, YieldPolicy{}, testLogger())

	// The resume that accompanies a real state change carries no countdown → 0.
	s.applyDecision(&YieldDecision{State: StateAvailable, Reason: pb.YieldReason_YIELD_REASON_RESUME})
	if d := s.LatestDecision(); d == nil || d.SecondsUntilNextChange != 0 {
		t.Fatalf("setup: resume decision should carry no countdown, got %+v", d)
	}

	// Next poll: same state, but with the real look-ahead countdown. Must be stored.
	s.applyDecisionUnlessWinning(&YieldDecision{
		State: StateAvailable, Reason: pb.YieldReason_YIELD_REASON_RESUME,
		SecondsUntilNextChange: 4242,
	})
	if s.CurrentState() != StateAvailable {
		t.Fatalf("no-op must not change state, got %v", s.CurrentState())
	}
	if d := s.LatestDecision(); d == nil || d.SecondsUntilNextChange != 4242 {
		t.Fatalf("B1 regression: no-op poll must refresh the countdown, got %+v", d)
	}

	// While WINNING_POST is active the guarded path must NOT touch the decision either
	// (the winning decision's countdown feeds the honest Retry-After).
	s.applyDecision(&YieldDecision{
		State: StateWinningPost, Reason: pb.YieldReason_YIELD_REASON_WINNING_POST,
		SecondsUntilNextChange: 30,
	})
	s.applyDecisionUnlessWinning(&YieldDecision{
		State: StateAvailable, Reason: pb.YieldReason_YIELD_REASON_RESUME,
		SecondsUntilNextChange: 99,
	})
	if d := s.LatestDecision(); d == nil || d.State != StateWinningPost || d.SecondsUntilNextChange != 30 {
		t.Fatalf("guarded no-op must not clobber the WINNING_POST decision, got %+v", d)
	}
}

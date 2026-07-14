package scheduler

import (
	"context"
	"testing"
	"time"
)

// S3 regression: when two WinningPoSt wins occur within the resume delay, the
// earlier win's resume timer must NOT resume during the later win's active
// window — only the latest win's timer resumes.
func TestWinningPostDoubleTimerDoesNotResumeEarly(t *testing.T) {
	s := New(&flakyLotus{}, YieldPolicy{}, testLogger())
	s.winningResumeDelay = 150 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Win #1 at t=0 (its timer fires ~t=150ms).
	s.triggerWinningYield(ctx, "win 1")
	if s.CurrentState() != StateWinningPost {
		t.Fatalf("expected WINNING_POST after first win, got %v", s.CurrentState())
	}

	// Win #2 at t≈80ms (its timer fires ~t=230ms).
	time.Sleep(80 * time.Millisecond)
	s.triggerWinningYield(ctx, "win 2")

	// At t≈180ms: win #1's timer has fired but must have SKIPPED resume because
	// win #2 superseded it. State must still be WINNING_POST.
	time.Sleep(100 * time.Millisecond) // ~t=180ms
	if s.CurrentState() != StateWinningPost {
		t.Fatalf("S3 regression: resumed during active second win — state=%v", s.CurrentState())
	}

	// After win #2's timer fires (~t=230ms), state resumes to AVAILABLE.
	time.Sleep(150 * time.Millisecond) // ~t=330ms
	if s.CurrentState() != StateAvailable {
		t.Fatalf("expected AVAILABLE after second win's timer, got %v", s.CurrentState())
	}
}

// A single win resumes normally after its delay.
func TestWinningPostSingleResume(t *testing.T) {
	s := New(&flakyLotus{}, YieldPolicy{}, testLogger())
	s.winningResumeDelay = 80 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.triggerWinningYield(ctx, "win")
	if s.CurrentState() != StateWinningPost {
		t.Fatalf("expected WINNING_POST, got %v", s.CurrentState())
	}
	time.Sleep(160 * time.Millisecond)
	if s.CurrentState() != StateAvailable {
		t.Fatalf("expected AVAILABLE after resume delay, got %v", s.CurrentState())
	}
}

package scheduler

import "testing"

// S5: completed-proof tracking must remember MULTIPLE deadlines within the same
// proving period (the old single-slot tracker forgot earlier ones).
func TestProofTrackingMultipleDeadlines(t *testing.T) {
	s := New(&flakyLotus{}, YieldPolicy{}, testLogger())
	const period int64 = 1000

	s.markProofCompleted(period, 5)
	s.markProofCompleted(period, 9)

	if !s.isProofAlreadyCompleted(period, 5) {
		t.Error("S5 regression: deadline 5 forgotten after marking deadline 9")
	}
	if !s.isProofAlreadyCompleted(period, 9) {
		t.Error("deadline 9 not tracked")
	}
	if s.isProofAlreadyCompleted(period, 7) {
		t.Error("deadline 7 was never completed but reported as complete")
	}
	if s.isProofAlreadyCompleted(period+1, 5) {
		t.Error("deadline 5 of a different period must not match")
	}
}

func TestProofTrackingBounded(t *testing.T) {
	s := New(&flakyLotus{}, YieldPolicy{}, testLogger())
	// Exceed the cap; must not grow unboundedly and must not panic.
	for i := int64(0); i < 200; i++ {
		s.markProofCompleted(i, uint64(i%48))
	}
	s.completedProofMu.RLock()
	n := len(s.completedProofs)
	s.completedProofMu.RUnlock()
	if n > 129 {
		t.Fatalf("completedProofs grew unbounded: %d entries", n)
	}
}

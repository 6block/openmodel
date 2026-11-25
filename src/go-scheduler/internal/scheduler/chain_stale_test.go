package scheduler

import (
	"context"
	"sync"
	"testing"
	"time"

	"openmodel/go-scheduler/internal/lotus"

	pb "openmodel/go-scheduler/proto/sidecar"
)

// frozenLotus answers every RPC successfully but reports whatever epoch the test
// sets. It models the failure this check exists for: a node that missed a network
// upgrade keeps serving its last good head, so it looks healthy while every
// deadline it reports is frozen. Deadlines are far away and every deadline has
// sectors, so without the staleness check the scheduler would stay AVAILABLE.
type frozenLotus struct {
	mu    sync.Mutex
	epoch int64
}

func (f *frozenLotus) set(epoch int64) {
	f.mu.Lock()
	f.epoch = epoch
	f.mu.Unlock()
}

func (f *frozenLotus) GetProvingDeadline(ctx context.Context) (*lotus.DeadlineInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return &lotus.DeadlineInfo{
		CurrentEpoch:         f.epoch,
		PeriodStart:          f.epoch - 100,
		Index:                3,
		Open:                 f.epoch + 1000, // far from the yield thresholds
		Close:                f.epoch + 1060,
		WPoStPeriodDeadlines: 48,
		WPoStProvingPeriod:   2880,
		WPoStChallengeWindow: 60,
	}, nil
}
func (f *frozenLotus) GetDeadlineSectors(ctx context.Context, idx uint64) (*lotus.DeadlineSectors, error) {
	return &lotus.DeadlineSectors{Sectors: 1, Partitions: 1}, nil
}
func (f *frozenLotus) GetMinerBaseInfo(ctx context.Context, epoch int64, tsk []lotus.TipsetCID) (*lotus.MinerBaseInfo, error) {
	return &lotus.MinerBaseInfo{}, nil
}
func (f *frozenLotus) SubscribeChainHead(ctx context.Context) (<-chan *lotus.ChainHead, error) {
	return make(chan *lotus.ChainHead), nil
}
func (f *frozenLotus) Close() error { return nil }

// staleScheduler builds a scheduler on a frozen node with a hand-driven clock.
func staleScheduler(failSafe bool, after time.Duration) (*Scheduler, *frozenLotus, *time.Time) {
	node := &frozenLotus{epoch: 4110312}
	s := New(node, YieldPolicy{FailSafeOnDisconnect: failSafe, ChainStaleAfter: after}, testLogger())
	clock := time.Date(2026, 9, 29, 6, 49, 0, 0, time.UTC)
	s.now = func() time.Time { return clock }
	return s, node, &clock
}

// TestChainStaleYieldsUnderFailSafe: the stale-chain incident. Lotus kept answering with
// epoch 4110312 for 16 days; with the check, the scheduler yields once the head has
// been still for longer than the threshold.
func TestChainStaleYieldsUnderFailSafe(t *testing.T) {
	s, _, clock := staleScheduler(true, 10*time.Minute)
	ctx := context.Background()

	s.checkWindowPost(ctx)
	if s.CurrentState() != StateAvailable {
		t.Fatalf("a fresh, far-from-deadline node must be AVAILABLE, got %s", s.CurrentState())
	}
	if s.ChainStale() {
		t.Fatal("the first observation must only arm the tracker, never report stale")
	}

	*clock = clock.Add(9 * time.Minute)
	s.checkWindowPost(ctx)
	if s.CurrentState() != StateAvailable || s.ChainStale() {
		t.Fatalf("below threshold must stay AVAILABLE, got %s stale=%v", s.CurrentState(), s.ChainStale())
	}

	*clock = clock.Add(2 * time.Minute) // 11 min without a new epoch
	s.checkWindowPost(ctx)
	if !s.ChainStale() {
		t.Fatal("a head stuck past the threshold must be reported stale")
	}
	if s.CurrentState() != StateWindowPost {
		t.Fatalf("a stale chain under fail-safe must yield, got %s", s.CurrentState())
	}
	d := s.LatestDecision()
	if d == nil || d.Reason != pb.YieldReason_YIELD_REASON_LOTUS_DISCONNECTED {
		t.Fatalf("yield reason must be LOTUS_DISCONNECTED, got %+v", d)
	}
}

// TestChainStaleRecoversWhenHeadAdvances: once the node moves again the flag clears
// and the normal deadline evaluation takes over.
func TestChainStaleRecoversWhenHeadAdvances(t *testing.T) {
	s, node, clock := staleScheduler(true, 10*time.Minute)
	ctx := context.Background()
	s.checkWindowPost(ctx)
	*clock = clock.Add(30 * time.Minute)
	s.checkWindowPost(ctx)
	if !s.ChainStale() || s.CurrentState() != StateWindowPost {
		t.Fatalf("setup: expected stale + yielded, got stale=%v %s", s.ChainStale(), s.CurrentState())
	}

	node.set(4110313)
	*clock = clock.Add(30 * time.Second)
	s.checkWindowPost(ctx)
	if s.ChainStale() {
		t.Fatal("an advancing head must clear the stale flag")
	}
	if s.CurrentState() != StateAvailable {
		t.Fatalf("after recovery the far-away deadline must leave the GPU AVAILABLE, got %s", s.CurrentState())
	}
}

// TestChainStaleWithoutFailSafeOnlyFlags: an operator who disabled fail-safe has
// chosen not to yield on Lotus trouble. The staleness must still be detected (and
// visible on /ready), but no yield is forced.
func TestChainStaleWithoutFailSafeOnlyFlags(t *testing.T) {
	s, _, clock := staleScheduler(false, 10*time.Minute)
	ctx := context.Background()
	s.checkWindowPost(ctx)
	*clock = clock.Add(time.Hour)
	s.checkWindowPost(ctx)
	if !s.ChainStale() {
		t.Fatal("staleness must be detected even without fail-safe")
	}
	if s.CurrentState() == StateWindowPost {
		t.Fatal("without fail-safe the scheduler must not force a yield")
	}
}

// TestChainStaleDisabled: a non-positive threshold turns the check off entirely.
func TestChainStaleDisabled(t *testing.T) {
	s, _, clock := staleScheduler(true, 0)
	ctx := context.Background()
	s.checkWindowPost(ctx)
	*clock = clock.Add(48 * time.Hour)
	s.checkWindowPost(ctx)
	if s.ChainStale() || s.CurrentState() != StateAvailable {
		t.Fatalf("disabled check must never fire, got stale=%v %s", s.ChainStale(), s.CurrentState())
	}
}

// TestChainStaleIgnoresBackwardEpoch: a reorg can briefly report a LOWER head. That
// must not reset the "last advanced" time — otherwise a node oscillating between two
// stuck heads would never be flagged.
func TestChainStaleIgnoresBackwardEpoch(t *testing.T) {
	s, node, clock := staleScheduler(true, 10*time.Minute)
	ctx := context.Background()
	s.checkWindowPost(ctx)
	*clock = clock.Add(6 * time.Minute)
	node.set(4110300)
	s.checkWindowPost(ctx)
	*clock = clock.Add(6 * time.Minute)
	node.set(4110312)
	s.checkWindowPost(ctx)
	if !s.ChainStale() {
		t.Fatal("a head that never exceeded its previous maximum for 12 minutes is stale")
	}
}

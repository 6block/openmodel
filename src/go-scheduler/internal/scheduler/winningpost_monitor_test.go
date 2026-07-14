package scheduler

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync/atomic"
	"testing"
	"time"

	"openmodel/go-scheduler/internal/curio"
	"openmodel/go-scheduler/internal/lotus"
)

// flakyLotus fails GetProvingDeadline the first `failN` times, then succeeds.
type flakyLotus struct {
	failN  int32
	calls  atomic.Int32
}

func (f *flakyLotus) GetProvingDeadline(ctx context.Context) (*lotus.DeadlineInfo, error) {
	n := f.calls.Add(1)
	if n <= f.failN {
		return nil, errors.New("lotus unavailable")
	}
	return &lotus.DeadlineInfo{CurrentEpoch: 100000}, nil
}
func (f *flakyLotus) GetDeadlineSectors(ctx context.Context, idx uint64) (*lotus.DeadlineSectors, error) {
	return &lotus.DeadlineSectors{Sectors: 1, Partitions: 1}, nil
}
func (f *flakyLotus) GetMinerBaseInfo(ctx context.Context, epoch int64, tsk []lotus.TipsetCID) (*lotus.MinerBaseInfo, error) {
	return &lotus.MinerBaseInfo{}, nil
}
func (f *flakyLotus) SubscribeChainHead(ctx context.Context) (<-chan *lotus.ChainHead, error) {
	return make(chan *lotus.ChainHead), nil
}
func (f *flakyLotus) Close() error { return nil }

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// S1 regression: a transient Lotus failure at startup must NOT permanently
// disable the WinningPoSt monitor — it must retry the initial epoch fetch.
func TestWinningPostMonitorRetriesInitialFetch(t *testing.T) {
	lotusMock := &flakyLotus{failN: 3}
	s := New(lotusMock, YieldPolicy{}, testLogger())
	s.initRetryInterval = 5 * time.Millisecond
	s.SetProofMonitor(curio.NewMockProofMonitor(0, testLogger()), 182063, curio.WaitConfig{
		PollInterval: time.Second, MaxWait: time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan struct{})
	go func() {
		s.monitorWinningPost(ctx)
		close(done)
	}()

	// Give it time to retry past the 3 failures (3 * 5ms + slack).
	time.Sleep(200 * time.Millisecond)

	calls := lotusMock.calls.Load()
	if calls < 4 {
		t.Fatalf("S1 regression: expected >= 4 GetProvingDeadline calls (retries), got %d", calls)
	}

	// The monitor should still be RUNNING (not returned early). Cancelling must
	// make it exit cleanly.
	select {
	case <-done:
		t.Fatal("monitor exited before context cancel — did not stay running")
	default:
	}

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("monitor did not exit after context cancel")
	}
}

// Sanity: when Lotus works immediately, the monitor starts without extra retries.
func TestWinningPostMonitorNoRetryWhenHealthy(t *testing.T) {
	lotusMock := &flakyLotus{failN: 0}
	s := New(lotusMock, YieldPolicy{}, testLogger())
	s.initRetryInterval = 5 * time.Millisecond
	s.SetProofMonitor(curio.NewMockProofMonitor(0, testLogger()), 182063, curio.WaitConfig{
		PollInterval: time.Second, MaxWait: time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())
	go s.monitorWinningPost(ctx)
	time.Sleep(50 * time.Millisecond)
	cancel()

	if got := lotusMock.calls.Load(); got != 1 {
		t.Fatalf("expected exactly 1 initial fetch when healthy, got %d", got)
	}
}

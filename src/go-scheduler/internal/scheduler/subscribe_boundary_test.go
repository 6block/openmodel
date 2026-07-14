package scheduler

import (
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	pb "openmodel/go-scheduler/proto/sidecar"
)

func subLog() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func newSchedForSubTest() *Scheduler {
	return New(nil, YieldPolicy{}, subLog())
}

// TestSubscriberIsolationSlowConsumer verifies the C7 fan-out safety property: a slow
// subscriber whose 32-deep buffer fills up has events DROPPED (non-blocking send),
// and that this never blocks the broadcaster or starves a healthy subscriber. A
// stuck Python client must not be able to wedge the scheduler's event loop.
func TestSubscriberIsolationSlowConsumer(t *testing.T) {
	s := newSchedForSubTest()

	_, slowCh := s.Subscribe()  // never drained → buffer fills
	idFast, fastCh := s.Subscribe()
	_ = slowCh

	// Broadcast many more events than the 32-deep buffer can hold.
	const n = 100
	done := make(chan struct{})
	go func() {
		for i := 0; i < n; i++ {
			s.broadcastEvent(&YieldDecision{State: StateAvailable, Message: "ev"})
		}
		close(done)
	}()

	// Drain the fast subscriber concurrently; it must keep receiving despite the slow
	// one being stuck.
	got := 0
	drainDone := make(chan struct{})
	go func() {
		for range fastCh {
			got++
			if got >= 32 { // fast consumer got a healthy share without deadlock
				break
			}
		}
		close(drainDone)
	}()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("broadcast blocked on a slow subscriber (fan-out not isolated)")
	}
	<-drainDone
	s.Unsubscribe(idFast)
	if got == 0 {
		t.Fatal("fast subscriber received nothing; slow consumer starved the fan-out")
	}
}

// TestUnsubscribeStopsDelivery verifies an unsubscribed channel is closed and stops
// receiving — no events delivered to a gone subscriber, no leak.
func TestUnsubscribeStopsDelivery(t *testing.T) {
	s := newSchedForSubTest()
	id, ch := s.Subscribe()

	s.broadcastEvent(&YieldDecision{State: StateAvailable, Message: "before"})
	if e, ok := <-ch; !ok || e == nil {
		t.Fatal("expected to receive the event before unsubscribe")
	}

	s.Unsubscribe(id)
	// After unsubscribe the channel is closed; a further read must report closed.
	if _, ok := <-ch; ok {
		t.Fatal("channel should be closed after Unsubscribe")
	}

	// Broadcasting after unsubscribe must not panic (no send on closed channel).
	s.broadcastEvent(&YieldDecision{State: StateWinningPost, Message: "after"})
}

// TestMultiSubscriberEachGetsEvent verifies every active subscriber independently
// receives a broadcast event (fan-out correctness).
func TestMultiSubscriberEachGetsEvent(t *testing.T) {
	s := newSchedForSubTest()
	const subs = 5
	chans := make([]<-chan *pb.ScheduleEvent, subs)
	ids := make([]int, subs)
	for i := 0; i < subs; i++ {
		ids[i], chans[i] = s.Subscribe()
	}

	s.broadcastEvent(&YieldDecision{State: StateWindowPost, Message: "fanout"})

	var wg sync.WaitGroup
	for i := 0; i < subs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			e, ok := <-chans[i]
			if !ok || e == nil || e.Message != "fanout" {
				t.Errorf("subscriber %d did not receive the broadcast event", i)
			}
		}(i)
	}
	wg.Wait()
	for _, id := range ids {
		s.Unsubscribe(id)
	}
}

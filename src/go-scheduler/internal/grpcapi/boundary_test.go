package grpcapi

import (
	"context"
	"strings"
	"testing"
	"time"

	"openmodel/go-scheduler/internal/scheduler"
	pb "openmodel/go-scheduler/proto/sidecar"
)

// C7 — gRPC/protobuf boundary tests for the scheduler↔inference channel.
// These exercise malformed/edge-case messages, unknown enum values, oversized
// fields, stream cancellation, and multi-subscriber isolation — none of which need
// a real socket (the handler and scheduler are driven directly).

// TestReportInferenceStatusEdgeValues feeds boundary and nonsensical field values
// (negative counts, far-future/negative timestamps, huge model string, unknown enum
// passthrough) and asserts the handler tolerates them without panicking and still
// returns a valid response. A misbehaving or version-skewed Python client must not be
// able to crash the scheduler.
func TestReportInferenceStatusEdgeValues(t *testing.T) {
	h := newHandler()
	cases := []*pb.InferenceStatusReport{
		nil, // nil message (a decode glitch / empty frame)
		{IsRunning: true, ActiveRequests: -5, TimestampUnix: -1},
		{ActiveRequests: 1 << 30, GpuUtilizationPct: -999.5, TimestampUnix: 1 << 62},
		{LoadedModel: strings.Repeat("x", 1<<20)}, // 1 MiB model name
		{GpuUtilizationPct: 100000},
	}
	for i, rpt := range cases {
		resp, err := h.ReportInferenceStatus(context.Background(), rpt)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}
		if resp == nil {
			t.Errorf("case %d: nil response", i)
		}
	}
}

// TestGetGpuScheduleNilRequest verifies a nil ScheduleRequest (an empty/garbled
// frame) does not panic the read RPC.
func TestGetGpuScheduleNilRequest(t *testing.T) {
	resp, err := newHandler().GetGpuSchedule(context.Background(), nil)
	if err != nil {
		t.Fatalf("nil request should be tolerated, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected a non-nil response")
	}
}

// TestSubscribeUnknownEnumPassthrough verifies that an out-of-range enum value set on
// a decision is streamed through as-is rather than crashing the send path. proto3
// open enums must round-trip unknown numbers (forward-compat with a newer peer).
func TestSubscribeUnknownEnumPassthrough(t *testing.T) {
	sched := scheduler.New(fakeLotus{}, scheduler.YieldPolicy{}, gLog())
	h := NewHandler(sched, gLog())

	// Inject a decision carrying an unknown enum number (999) via the test seam.
	sched.SetLatestDecisionForTest(&scheduler.YieldDecision{
		State:   pb.GpuState(999),
		Reason:  pb.YieldReason(888),
		Urgency: pb.YieldUrgency(777),
		Message: "future peer",
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // exit right after the initial send
	fs := &fakeStream{ctx: ctx}
	if err := h.SubscribeScheduleEvents(&pb.ScheduleRequest{}, fs); err != nil {
		t.Fatal(err)
	}
	if len(fs.sent) != 1 {
		t.Fatalf("expected 1 initial event, got %d", len(fs.sent))
	}
	// The initial event copies Reason/Urgency from the latest decision; an unknown
	// enum number must pass through unchanged (proto3 open-enum forward-compat).
	if fs.sent[0].Reason != pb.YieldReason(888) {
		t.Errorf("unknown Reason enum should pass through, got %v", fs.sent[0].Reason)
	}
	if fs.sent[0].Urgency != pb.YieldUrgency(777) {
		t.Errorf("unknown Urgency enum should pass through, got %v", fs.sent[0].Urgency)
	}
}

// TestSubscribeStreamCancelStopsPromptly verifies that cancelling the stream context
// returns from SubscribeScheduleEvents promptly (no goroutine leak / hang) even when
// no events are flowing.
func TestSubscribeStreamCancelStopsPromptly(t *testing.T) {
	h := newHandler()
	ctx, cancel := context.WithCancel(context.Background())
	fs := &fakeStream{ctx: ctx}

	done := make(chan error, 1)
	go func() { done <- h.SubscribeScheduleEvents(&pb.ScheduleRequest{}, fs) }()

	// Let the initial event send, then cancel.
	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("clean cancel should return nil, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("SubscribeScheduleEvents did not return promptly after context cancel")
	}
}

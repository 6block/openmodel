package grpcapi

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"

	"google.golang.org/grpc/metadata"

	"openmodel/go-scheduler/internal/lotus"
	"openmodel/go-scheduler/internal/scheduler"
	pb "openmodel/go-scheduler/proto/sidecar"
)

func gLog() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// fakeLotus satisfies lotus.Client without touching a network (scheduler.New only
// stores it; these tests don't run the polling loop).
type fakeLotus struct{}

func (fakeLotus) GetProvingDeadline(context.Context) (*lotus.DeadlineInfo, error) { return nil, nil }
func (fakeLotus) GetDeadlineSectors(context.Context, uint64) (*lotus.DeadlineSectors, error) {
	return nil, nil
}
func (fakeLotus) GetMinerBaseInfo(context.Context, int64, []lotus.TipsetCID) (*lotus.MinerBaseInfo, error) {
	return nil, nil
}
func (fakeLotus) SubscribeChainHead(context.Context) (<-chan *lotus.ChainHead, error) {
	return nil, nil
}
func (fakeLotus) Close() error { return nil }

// fakeStream implements pb.SchedulerService_SubscribeScheduleEventsServer.
type fakeStream struct {
	ctx     context.Context
	sent    []*pb.ScheduleEvent
	sendErr error
}

func (f *fakeStream) Send(e *pb.ScheduleEvent) error {
	if f.sendErr != nil {
		return f.sendErr
	}
	f.sent = append(f.sent, e)
	return nil
}
func (f *fakeStream) Context() context.Context     { return f.ctx }
func (f *fakeStream) SetHeader(metadata.MD) error  { return nil }
func (f *fakeStream) SendHeader(metadata.MD) error { return nil }
func (f *fakeStream) SetTrailer(metadata.MD)       {}
func (f *fakeStream) SendMsg(interface{}) error    { return nil }
func (f *fakeStream) RecvMsg(interface{}) error    { return nil }

func newHandler() *Handler {
	return NewHandler(scheduler.New(fakeLotus{}, scheduler.YieldPolicy{}, gLog()), gLog())
}

func TestGetGpuSchedule(t *testing.T) {
	resp, err := newHandler().GetGpuSchedule(context.Background(), &pb.ScheduleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.CurrentState != scheduler.StateUnknown {
		t.Errorf("initial CurrentState = %v, want UNKNOWN", resp.CurrentState)
	}
}

func TestReportInferenceStatus(t *testing.T) {
	resp, err := newHandler().ReportInferenceStatus(context.Background(),
		&pb.InferenceStatusReport{IsRunning: true, ActiveRequests: 2, TimestampUnix: 1000})
	if err != nil {
		t.Fatal(err)
	}
	if resp.CurrentState != scheduler.StateUnknown {
		t.Errorf("CurrentState = %v, want UNKNOWN", resp.CurrentState)
	}
}

func TestSubscribeSendsInitialEvent(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel so the stream loop exits right after the initial send
	fs := &fakeStream{ctx: ctx}
	if err := newHandler().SubscribeScheduleEvents(&pb.ScheduleRequest{}, fs); err != nil {
		t.Fatal(err)
	}
	if len(fs.sent) != 1 {
		t.Fatalf("expected exactly 1 initial event, got %d", len(fs.sent))
	}
	if fs.sent[0].Message != "initial state" {
		t.Errorf("initial event message = %q, want %q", fs.sent[0].Message, "initial state")
	}
}

func TestSubscribePropagatesSendError(t *testing.T) {
	fs := &fakeStream{ctx: context.Background(), sendErr: errors.New("broken pipe")}
	if err := newHandler().SubscribeScheduleEvents(&pb.ScheduleRequest{}, fs); err == nil {
		t.Fatal("expected the send error to propagate out of SubscribeScheduleEvents")
	}
}

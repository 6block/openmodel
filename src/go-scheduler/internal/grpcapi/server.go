package grpcapi

import (
	"context"
	"log/slog"
	"time"

	"openmodel/go-scheduler/internal/scheduler"
	pb "openmodel/go-scheduler/proto/sidecar"
)

// Handler implements the gRPC SchedulerService.
type Handler struct {
	pb.UnimplementedSchedulerServiceServer
	sched  *scheduler.Scheduler
	logger *slog.Logger
}

// NewHandler creates a new gRPC handler.
func NewHandler(sched *scheduler.Scheduler, logger *slog.Logger) *Handler {
	return &Handler{
		sched:  sched,
		logger: logger,
	}
}

// GetGpuSchedule returns the current GPU availability state.
func (h *Handler) GetGpuSchedule(ctx context.Context, req *pb.ScheduleRequest) (*pb.ScheduleResponse, error) {
	state := h.sched.CurrentState()
	decision := h.sched.LatestDecision()

	resp := &pb.ScheduleResponse{
		CurrentState: state,
	}

	if decision != nil {
		resp.SecondsUntilNextChange = decision.SecondsUntilNextChange
		resp.Message = decision.Message
	}

	return resp, nil
}

// SubscribeScheduleEvents streams GPU schedule events to the Python inference service.
func (h *Handler) SubscribeScheduleEvents(req *pb.ScheduleRequest, stream pb.SchedulerService_SubscribeScheduleEventsServer) error {
	subID, ch := h.sched.Subscribe()
	defer h.sched.Unsubscribe(subID)

	h.logger.Info("new event subscriber", "subscriber_id", subID)

	// Send initial state
	state := h.sched.CurrentState()
	decision := h.sched.LatestDecision()
	initialEvent := &pb.ScheduleEvent{
		State:   state,
		Reason:  pb.YieldReason_YIELD_REASON_RESUME,
		Message: "initial state",
	}
	if decision != nil {
		initialEvent.Urgency = decision.Urgency
		initialEvent.SecondsUntilDeadline = decision.SecondsUntilNextChange
		initialEvent.Reason = decision.Reason
		initialEvent.Message = decision.Message
	}
	if err := stream.Send(initialEvent); err != nil {
		return err
	}

	// Stream events
	for {
		select {
		case <-stream.Context().Done():
			h.logger.Info("subscriber disconnected", "subscriber_id", subID)
			return nil
		case event, ok := <-ch:
			if !ok {
				return nil
			}
			if err := stream.Send(event); err != nil {
				h.logger.Error("send event error", "subscriber_id", subID, "error", err)
				return err
			}
		}
	}
}

// ReportInferenceStatus receives status reports from the Python inference service.
func (h *Handler) ReportInferenceStatus(ctx context.Context, report *pb.InferenceStatusReport) (*pb.ScheduleResponse, error) {
	// Guard against a nil/empty message (a decode glitch or a misbehaving client
	// sending an empty frame): dereferencing it would panic the RPC and could take
	// down the scheduler. Treat nil as an empty report.
	if report == nil {
		report = &pb.InferenceStatusReport{}
	}
	h.logger.Debug("inference status report",
		"running", report.IsRunning,
		"active_requests", report.ActiveRequests,
		"gpu_utilization", report.GpuUtilizationPct,
		"model", report.LoadedModel,
		"timestamp", time.Unix(report.TimestampUnix, 0),
	)

	state := h.sched.CurrentState()
	decision := h.sched.LatestDecision()

	resp := &pb.ScheduleResponse{
		CurrentState: state,
	}
	if decision != nil {
		resp.SecondsUntilNextChange = decision.SecondsUntilNextChange
		resp.Message = decision.Message
	}

	return resp, nil
}

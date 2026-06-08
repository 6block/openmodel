package scheduler

import (
	"openmodel/go-scheduler/internal/config"
	"openmodel/go-scheduler/internal/lotus"

	pb "openmodel/go-scheduler/proto/sidecar"
)

// YieldPolicy configures thresholds for yielding GPU to mining operations.
type YieldPolicy struct {
	WindowPost           config.WindowPostPolicy
	WinningPost          config.WinningPostPolicy
	FailSafeOnDisconnect bool
}

// YieldDecision represents the scheduler's decision about GPU state.
type YieldDecision struct {
	State   GpuState
	Urgency pb.YieldUrgency
	Reason  pb.YieldReason
	Message string
	// SecondsUntilNextChange is the estimated time until the state might change.
	SecondsUntilNextChange int64
}

// EvaluateWindowPost determines the GPU state based on WindowPoSt deadline info.
func (p *YieldPolicy) EvaluateWindowPost(info *lotus.DeadlineInfo) YieldDecision {
	if info == nil {
		if p.FailSafeOnDisconnect {
			return YieldDecision{
				State:   StateWindowPost,
				Urgency: pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE,
				Reason:  pb.YieldReason_YIELD_REASON_LOTUS_DISCONNECTED,
				Message: "lotus disconnected, fail-safe: yielding to mining",
			}
		}
		return YieldDecision{
			State:   StateAvailable,
			Urgency: pb.YieldUrgency_YIELD_URGENCY_NORMAL,
			Reason:  pb.YieldReason_YIELD_REASON_UNKNOWN,
			Message: "lotus disconnected, no fail-safe",
		}
	}

	secondsUntilOpen := info.SecondsUntilOpen()

	// Deadline is currently active
	if info.IsOpen() {
		secondsUntilClose := info.SecondsUntilClose()
		return YieldDecision{
			State:                  StateWindowPost,
			Urgency:                pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE,
			Reason:                 pb.YieldReason_YIELD_REASON_WINDOW_POST_ACTIVE,
			Message:                "WindowPoSt deadline is active",
			SecondsUntilNextChange: secondsUntilClose + int64(p.WindowPost.ResumeDelayAfterCloseSec),
		}
	}

	gracefulThreshold := int64(p.WindowPost.GracefulYieldThresholdSec)
	hardStopThreshold := int64(p.WindowPost.HardStopThresholdSec)

	// Hard stop zone: very close to deadline
	if secondsUntilOpen <= hardStopThreshold {
		return YieldDecision{
			State:                  StateWindowPost,
			Urgency:                pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE,
			Reason:                 pb.YieldReason_YIELD_REASON_WINDOW_POST_APPROACHING,
			Message:                "WindowPoSt deadline imminent, hard stop",
			SecondsUntilNextChange: secondsUntilOpen,
		}
	}

	// Graceful yield zone: approaching deadline
	if secondsUntilOpen <= gracefulThreshold {
		return YieldDecision{
			State:                  StateYielding,
			Urgency:                pb.YieldUrgency_YIELD_URGENCY_NORMAL,
			Reason:                 pb.YieldReason_YIELD_REASON_WINDOW_POST_APPROACHING,
			Message:                "WindowPoSt deadline approaching, graceful yield",
			SecondsUntilNextChange: secondsUntilOpen - hardStopThreshold,
		}
	}

	// Safe zone: far from deadline
	return YieldDecision{
		State:                  StateAvailable,
		Urgency:                pb.YieldUrgency_YIELD_URGENCY_NORMAL,
		Reason:                 pb.YieldReason_YIELD_REASON_RESUME,
		Message:                "GPU available for inference",
		SecondsUntilNextChange: secondsUntilOpen - gracefulThreshold,
	}
}

// EvaluateWinningPost determines if a WinningPoSt event requires GPU yield.
func (p *YieldPolicy) EvaluateWinningPost(baseInfo *lotus.MinerBaseInfo) YieldDecision {
	if !p.WinningPost.Enabled || baseInfo == nil {
		return YieldDecision{
			State:   StateAvailable,
			Reason:  pb.YieldReason_YIELD_REASON_RESUME,
			Message: "WinningPoSt check: not eligible or disabled",
		}
	}

	if baseInfo.EligibleForMining {
		return YieldDecision{
			State:                  StateWinningPost,
			Urgency:                pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE,
			Reason:                 pb.YieldReason_YIELD_REASON_WINNING_POST,
			Message:                "WinningPoSt: miner elected for block production",
			SecondsUntilNextChange: int64(p.WinningPost.TimeoutSec),
		}
	}

	return YieldDecision{
		State:   StateAvailable,
		Reason:  pb.YieldReason_YIELD_REASON_RESUME,
		Message: "WinningPoSt check: not elected",
	}
}

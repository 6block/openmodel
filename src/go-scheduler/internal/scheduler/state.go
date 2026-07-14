package scheduler

import (
	"sync"

	pb "openmodel/go-scheduler/proto/sidecar"
)

// GpuState represents the current GPU allocation state.
type GpuState = pb.GpuState

const (
	StateUnknown     = pb.GpuState_GPU_STATE_UNKNOWN
	StateAvailable   = pb.GpuState_GPU_STATE_AVAILABLE
	StateYielding    = pb.GpuState_GPU_STATE_YIELDING
	StateWindowPost  = pb.GpuState_GPU_STATE_WINDOW_POST
	StateWinningPost = pb.GpuState_GPU_STATE_WINNING_POST
)

// StateMachine manages GPU state transitions with thread safety.
type StateMachine struct {
	mu      sync.RWMutex
	current GpuState
}

// NewStateMachine creates a state machine starting in UNKNOWN state.
//
// UNKNOWN is a conservative default: until the scheduler completes its first
// Lotus evaluation we cannot know whether the GPU is safe for inference, so we
// must NOT advertise AVAILABLE. The inference service treats any non-AVAILABLE
// state as "do not load the model" (start_paused), so a startup that coincides
// with active mining will not contend for the GPU. The first checkWindowPost
// transitions UNKNOWN -> AVAILABLE/WINDOW_POST within one poll.
func NewStateMachine() *StateMachine {
	return &StateMachine{
		current: StateUnknown,
	}
}

// Current returns the current GPU state.
func (sm *StateMachine) Current() GpuState {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.current
}

// Transition attempts to move to a new state. Returns true if the transition occurred.
// Valid transitions:
//   AVAILABLE -> YIELDING       (WindowPoSt approaching)
//   AVAILABLE -> WINNING_POST   (WinningPoSt triggered)
//   YIELDING  -> WINDOW_POST    (WindowPoSt imminent/active)
//   YIELDING  -> WINNING_POST   (WinningPoSt during yield)
//   WINDOW_POST -> AVAILABLE    (WindowPoSt done)
//   WINNING_POST -> AVAILABLE   (WinningPoSt done)
//   WINNING_POST -> YIELDING    (WinningPoSt done but WindowPoSt approaching)
//   WINNING_POST -> WINDOW_POST (WinningPoSt done but WindowPoSt active)
//   Any -> WINDOW_POST          (fail-safe)
//   Any -> AVAILABLE            (resume)
//
// NOTE: Transition is intentionally permissive (it accepts any target state).
// The documented transition graph is advisory; enforcing it here would risk
// blocking a legitimate fail-safe ("Any -> WINDOW_POST") or resume. Safety is
// instead enforced upstream in the scheduler: checkWindowPost refuses to
// override an active WINNING_POST, the proofMonitorActive guard prevents resume
// races, and the WinningPoSt resume-timer generation guard prevents a stale
// timer from resuming during a newer win.
func (sm *StateMachine) Transition(newState GpuState) bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.current == newState {
		return false // No change
	}

	sm.current = newState
	return true
}

// TransitionUnless moves to newState UNLESS the current state is in `blocked`. The
// check and the set happen atomically under the lock, so it cannot clobber a state
// a concurrent goroutine set in between — e.g. a WindowPoSt-path resume that raced
// a WinningPoSt win cannot overwrite the just-set WINNING_POST (audit HIGH fix for
// the checkWindowPost TOCTOU and the proof-monitor's unconditional resume). Returns
// true only if the transition actually occurred.
func (sm *StateMachine) TransitionUnless(newState GpuState, blocked ...GpuState) bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	for _, b := range blocked {
		if sm.current == b {
			return false
		}
	}
	if sm.current == newState {
		return false
	}
	sm.current = newState
	return true
}

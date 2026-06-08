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

// NewStateMachine creates a state machine starting in AVAILABLE state.
func NewStateMachine() *StateMachine {
	return &StateMachine{
		current: StateAvailable,
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
func (sm *StateMachine) Transition(newState GpuState) bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.current == newState {
		return false // No change
	}

	sm.current = newState
	return true
}

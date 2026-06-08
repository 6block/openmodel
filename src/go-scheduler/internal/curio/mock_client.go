package curio

import (
	"context"
	"log/slog"
	"time"
)

// MockProofMonitor simulates proof completion after a configurable delay.
// Used in dev mode when no real Curio database is available.
type MockProofMonitor struct {
	proofDelay time.Duration
	logger     *slog.Logger
}

// NewMockProofMonitor creates a mock monitor that simulates proof completion.
func NewMockProofMonitor(proofDelay time.Duration, logger *slog.Logger) *MockProofMonitor {
	return &MockProofMonitor{
		proofDelay: proofDelay,
		logger:     logger,
	}
}

func (m *MockProofMonitor) IsProofComplete(ctx context.Context, spID int64, periodStart int64, deadline uint64) (bool, error) {
	return false, nil // Mock always says "not yet" until WaitForProofComplete
}

func (m *MockProofMonitor) WaitForProofComplete(ctx context.Context, spID int64, periodStart int64, deadline uint64, cfg WaitConfig) bool {
	m.logger.Info("mock: simulating proof computation",
		"delay", m.proofDelay,
		"deadline", deadline,
	)

	select {
	case <-ctx.Done():
		return false
	case <-time.After(m.proofDelay):
		m.logger.Info("mock: proof computation complete",
			"deadline", deadline,
		)
		return true
	}
}

func (m *MockProofMonitor) CheckWinningPost(ctx context.Context, spID int64, sinceEpoch int64) (*WinningPostWin, error) {
	return nil, nil // Mock never wins
}

func (m *MockProofMonitor) Close() error {
	return nil
}

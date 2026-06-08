package lotus

import (
	"context"
	"sync"
	"time"
)

// MockClient simulates a Lotus Miner node for local development.
type MockClient struct {
	mu            sync.Mutex
	currentEpoch  int64
	deadlineIndex uint64
	deadlineOpen  int64
	deadlineClose int64

	// WinningPoSt simulation: set to true to trigger a winning event at next check.
	winningEligible bool

	// Per-deadline sector simulation.
	// Key: deadline index, Value: number of sectors.
	// Only deadlines in this map have sectors; others return 0.
	deadlineSectors map[uint64]int

	// Auto-advance epoch every 30 seconds
	startTime  time.Time
	startEpoch int64
}

// NewMockClient creates a mock Lotus client that simulates epoch progression.
func NewMockClient() *MockClient {
	now := time.Now()
	startEpoch := int64(100000)

	return &MockClient{
		currentEpoch:  startEpoch,
		deadlineIndex: 12,
		deadlineOpen:  startEpoch + 40,  // ~20 min away — enough time to test WinningPoSt first
		deadlineClose: startEpoch + 100, // 30-minute window (60 epochs)
		// nil = all deadlines have sectors (always trigger yield for testing)
		// Set via SetDeadlineSectors() to simulate sparse sectors like a real miner.
		deadlineSectors: nil,
		startTime:       now,
		startEpoch:      startEpoch,
	}
}

func (m *MockClient) advanceEpoch() {
	elapsed := time.Since(m.startTime).Seconds()
	m.currentEpoch = m.startEpoch + int64(elapsed/EpochDuration)
}

func (m *MockClient) GetProvingDeadline(ctx context.Context) (*DeadlineInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.advanceEpoch()

	// If past current deadline, advance to next one (48 deadlines per period)
	if m.currentEpoch >= m.deadlineClose {
		period := int64(120)  // shortened for dev testing (~60 min cycle)
		window := int64(60)   // 30 minutes = 60 epochs
		m.deadlineOpen = m.deadlineClose + period - window
		m.deadlineClose = m.deadlineOpen + window
		m.deadlineIndex = (m.deadlineIndex + 1) % 48
	}

	return &DeadlineInfo{
		CurrentEpoch:          m.currentEpoch,
		PeriodStart:           m.deadlineOpen - int64(m.deadlineIndex)*60,
		Index:                 m.deadlineIndex,
		Open:                  m.deadlineOpen,
		Close:                 m.deadlineClose,
		Challenge:             m.deadlineOpen,
		FaultCutoff:           m.deadlineOpen - 70,
		WPoStPeriodDeadlines:  48,
		WPoStProvingPeriod:    2880,
		WPoStChallengeWindow:  60,
		WPoStChallengeLookback: 20,
	}, nil
}

func (m *MockClient) GetDeadlineSectors(ctx context.Context, deadlineIdx uint64) (*DeadlineSectors, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// If deadlineSectors is nil, all deadlines have sectors (always trigger yield).
	// If set, only deadlines in the map have sectors.
	sectors := 1 // default: every deadline has sectors
	if m.deadlineSectors != nil {
		sectors = m.deadlineSectors[deadlineIdx] // 0 if not in map
	}
	partitions := 0
	if sectors > 0 {
		partitions = 1
	}

	return &DeadlineSectors{
		Deadline:   deadlineIdx,
		Partitions: partitions,
		Sectors:    sectors,
		Faults:     0,
	}, nil
}

// SetDeadlineSectors configures which deadlines have sectors for testing.
func (m *MockClient) SetDeadlineSectors(sectorMap map[uint64]int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deadlineSectors = sectorMap
}

func (m *MockClient) GetMinerBaseInfo(ctx context.Context, epoch int64, tsk []TipsetCID) (*MinerBaseInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	eligible := m.winningEligible
	// Reset after check (one-shot simulation)
	m.winningEligible = false

	return &MinerBaseInfo{
		HasMinPower:       true,
		EligibleForMining: eligible,
	}, nil
}

// SetWinningEligible triggers a WinningPoSt event on the next MinerGetBaseInfo call.
func (m *MockClient) SetWinningEligible(eligible bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.winningEligible = eligible
}

// SetDeadline allows tests to manually set the next deadline.
func (m *MockClient) SetDeadline(open, close int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deadlineOpen = open
	m.deadlineClose = close
}

func (m *MockClient) SubscribeChainHead(ctx context.Context) (<-chan *ChainHead, error) {
	ch := make(chan *ChainHead, 16)

	go func() {
		defer close(ch)
		ticker := time.NewTicker(EpochDuration * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				m.mu.Lock()
				m.advanceEpoch()
				epoch := m.currentEpoch
				m.mu.Unlock()

				select {
				case ch <- &ChainHead{Height: epoch}:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return ch, nil
}

func (m *MockClient) Close() error {
	return nil
}

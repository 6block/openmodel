package scheduler

import (
	"testing"

	"openmodel/go-scheduler/internal/config"
)

// S2: the sector-cache refresh interval must be configurable (so newly-sealed
// sectors are detected sooner) with a sane fallback.
func TestSectorCacheRefreshIntervalConfigurable(t *testing.T) {
	// Explicit policy value is honored.
	s := New(&flakyLotus{}, YieldPolicy{
		WindowPost: config.WindowPostPolicy{SectorCacheRefreshEpochs: 360},
	}, testLogger())
	if got := s.sectorCacheRefreshInterval(); got != 360 {
		t.Fatalf("expected configured 360, got %d", got)
	}

	// Zero-value policy falls back to the conservative 2880.
	s2 := New(&flakyLotus{}, YieldPolicy{}, testLogger())
	if got := s2.sectorCacheRefreshInterval(); got != 2880 {
		t.Fatalf("expected fallback 2880, got %d", got)
	}
}


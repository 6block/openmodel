package config

import (
	"os"
	"path/filepath"
	"testing"
)

// S2: the loader must default the sector-cache refresh interval to ~3h (360
// epochs), not the old hardcoded ~24h (2880), so an actively-sealing miner
// detects newly-sealed sectors and yields for their WindowPoSt in time.
func TestSectorCacheRefreshDefault(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.yaml")
	if err := os.WriteFile(path, []byte("mode: dev\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := cfg.Scheduler.WindowPost.SectorCacheRefreshEpochs; got != 360 {
		t.Fatalf("expected default 360, got %d", got)
	}
}

func TestSectorCacheRefreshHonored(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.yaml")
	yaml := "mode: dev\nscheduler:\n  window_post:\n    sector_cache_refresh_epochs: 120\n"
	if err := os.WriteFile(path, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := cfg.Scheduler.WindowPost.SectorCacheRefreshEpochs; got != 120 {
		t.Fatalf("expected configured 120, got %d", got)
	}
}

package lotus

import "testing"

func TestSecondsUntilOpen(t *testing.T) {
	if got := (&DeadlineInfo{CurrentEpoch: 10, Open: 20}).SecondsUntilOpen(); got != 300 {
		t.Errorf("SecondsUntilOpen = %d, want 300 (10 epochs x 30s)", got)
	}
	// already past the open epoch → clamp to 0 (not negative)
	if got := (&DeadlineInfo{CurrentEpoch: 30, Open: 20}).SecondsUntilOpen(); got != 0 {
		t.Errorf("past-open SecondsUntilOpen = %d, want 0", got)
	}
}

func TestSecondsUntilClose(t *testing.T) {
	if got := (&DeadlineInfo{CurrentEpoch: 5, Close: 15}).SecondsUntilClose(); got != 300 {
		t.Errorf("SecondsUntilClose = %d, want 300", got)
	}
	if got := (&DeadlineInfo{CurrentEpoch: 20, Close: 15}).SecondsUntilClose(); got != 0 {
		t.Errorf("past-close SecondsUntilClose = %d, want 0", got)
	}
}

func TestIsOpen(t *testing.T) {
	if !(&DeadlineInfo{CurrentEpoch: 5, Open: 0, Close: 10}).IsOpen() {
		t.Error("epoch 5 in [0,10) should be open")
	}
	if (&DeadlineInfo{CurrentEpoch: 10, Open: 0, Close: 10}).IsOpen() {
		t.Error("Close is exclusive: epoch 10 in [0,10) should be closed")
	}
	if (&DeadlineInfo{CurrentEpoch: 0, Open: 0, Close: 10}).IsOpen() != true {
		t.Error("Open is inclusive: epoch 0 in [0,10) should be open")
	}
	if (&DeadlineInfo{CurrentEpoch: 0, Open: 5, Close: 10}).IsOpen() {
		t.Error("before open: epoch 0 < 5 should be closed")
	}
}

func TestHasSectors(t *testing.T) {
	if !(&DeadlineSectors{Sectors: 1}).HasSectors() {
		t.Error("Sectors=1 should HasSectors")
	}
	if (&DeadlineSectors{Sectors: 0}).HasSectors() {
		t.Error("Sectors=0 should not HasSectors")
	}
}

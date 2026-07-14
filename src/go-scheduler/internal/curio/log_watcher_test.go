package curio

import (
	"context"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func lwLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func appendLine(t *testing.T, path, line string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString(line + "\n"); err != nil {
		t.Fatal(err)
	}
}

func expectSignal(t *testing.T, ch <-chan struct{}, within time.Duration, msg string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(within):
		t.Fatal(msg)
	}
}

func TestLogWatcherDetectsWin(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "curio.log")
	if err := os.WriteFile(logPath, []byte("startup\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := NewLogWatcher(logPath, lwLogger()).Watch(ctx)
	if err != nil {
		t.Fatal(err)
	}

	appendLine(t, logPath, "2026-06-03 some unrelated log line")            // ignored
	appendLine(t, logPath, "2026-06-03 WinPostTask won election deadline=5") // matched
	expectSignal(t, ch, 3*time.Second, "expected detection of 'WinPostTask won election'")
}

// TestLogWatcherHandlesRotation is the regression for the rotation gap: after the
// log is rotated away and a fresh file appears at the same path, new wins must
// still be detected (the watcher must reopen).
func TestLogWatcherHandlesRotation(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "curio.log")
	os.WriteFile(logPath, []byte("startup\n"), 0644)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, _ := NewLogWatcher(logPath, lwLogger()).Watch(ctx)

	appendLine(t, logPath, "WinPostTask won election #1")
	expectSignal(t, ch, 3*time.Second, "first win not detected")

	// rotate: move aside, create a fresh file at the same path
	if err := os.Rename(logPath, logPath+".1"); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(logPath, []byte(""), 0644)
	appendLine(t, logPath, "WinPostTask won election #2 after rotation")
	expectSignal(t, ch, 3*time.Second, "win after rotation not detected (reopen failed)")
}

// TestLogWatcherHandlesTruncation covers in-place truncation (same inode, size
// shrinks below the read offset).
func TestLogWatcherHandlesTruncation(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "curio.log")
	os.WriteFile(logPath, []byte("a reasonably long initial line for padding padding\n"), 0644)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, _ := NewLogWatcher(logPath, lwLogger()).Watch(ctx)

	appendLine(t, logPath, "WinPostTask won election before truncate")
	expectSignal(t, ch, 3*time.Second, "pre-truncate win not detected")

	if err := os.Truncate(logPath, 0); err != nil {
		t.Fatal(err)
	}
	appendLine(t, logPath, "WinPostTask won election after truncate")
	expectSignal(t, ch, 3*time.Second, "win after truncation not detected (no reopen)")
}

func TestLogWatcherMissingFileErrors(t *testing.T) {
	_, err := NewLogWatcher(filepath.Join(t.TempDir(), "nope.log"), lwLogger()).Watch(context.Background())
	if err == nil {
		t.Fatal("expected an error when the log file does not exist")
	}
}

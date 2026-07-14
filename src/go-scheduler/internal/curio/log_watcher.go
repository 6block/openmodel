package curio

import (
	"bufio"
	"context"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"
)

// LogWatcher tails a Curio log file and detects "WinPostTask won election"
// entries in real-time. This fires ~4 seconds before the DB is updated,
// allowing the scheduler to yield GPU before proof computation starts.
type LogWatcher struct {
	logPath string
	logger  *slog.Logger
}

// NewLogWatcher creates a watcher for the given Curio log file path.
func NewLogWatcher(logPath string, logger *slog.Logger) *LogWatcher {
	return &LogWatcher{
		logPath: logPath,
		logger:  logger,
	}
}

// Watch tails the Curio log file and sends a signal on the returned channel
// each time "WinPostTask won election" is detected. The channel is closed
// when ctx is cancelled or an unrecoverable error occurs.
func (w *LogWatcher) Watch(ctx context.Context) (<-chan struct{}, error) {
	ch := make(chan struct{}, 4)

	f, err := os.Open(w.logPath)
	if err != nil {
		close(ch)
		return nil, err
	}

	// Seek to end — only watch new entries, not historical ones
	if _, err := f.Seek(0, io.SeekEnd); err != nil {
		f.Close()
		close(ch)
		return nil, err
	}

	w.logger.Info("Curio log watcher started",
		"path", w.logPath,
		"detection_target", "WinPostTask won election",
	)

	go func() {
		defer close(ch)
		cur := f
		defer func() { cur.Close() }()

		reader := bufio.NewReader(cur)

		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			line, err := reader.ReadString('\n')
			if err != nil {
				// EOF — before sleeping, check whether the log was rotated (a new
				// inode now sits at logPath) or truncated (shrank below our offset).
				// If so, reopen from the start so new entries aren't missed.
				if nf := w.reopenIfRotated(cur); nf != nil {
					cur.Close()
					cur = nf
					reader = bufio.NewReader(cur)
					w.logger.Info("Curio log rotated/truncated, reopened", "path", w.logPath)
					continue
				}
				time.Sleep(200 * time.Millisecond)
				continue
			}

			if strings.Contains(line, "WinPostTask won election") {
				w.logger.Info("detected WinningPoSt election in Curio log!",
					"log_line_prefix", truncate(line, 120),
				)
				select {
				case ch <- struct{}{}:
				default:
					// Channel full, skip (shouldn't happen with buffer of 4)
				}
			}
		}
	}()

	return ch, nil
}

// reopenIfRotated returns a freshly-opened *os.File (positioned at the start) when
// the file at logPath has been rotated (different inode) or truncated (size shrank
// below the current read offset), else nil. The new file is read from the
// beginning because all of its content is new since the rotation.
func (w *LogWatcher) reopenIfRotated(cur *os.File) *os.File {
	pathInfo, err := os.Stat(w.logPath)
	if err != nil {
		return nil // file briefly absent mid-rotation; keep current fd and retry
	}
	curInfo, err := cur.Stat()
	if err != nil {
		return nil
	}
	rotated := !os.SameFile(pathInfo, curInfo)
	curOffset, _ := cur.Seek(0, io.SeekCurrent)
	truncated := pathInfo.Size() < curOffset
	if !rotated && !truncated {
		return nil
	}
	nf, err := os.Open(w.logPath)
	if err != nil {
		return nil
	}
	return nf
}

func truncate(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

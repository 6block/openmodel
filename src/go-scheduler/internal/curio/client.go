// Package curio provides integration with the Curio storage provider's database
// to detect WindowPoSt proof completion in real-time.
package curio

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"time"

	_ "github.com/lib/pq" // PostgreSQL/YugabyteDB driver
)

// WaitConfig configures proof completion polling.
type WaitConfig struct {
	PollInterval time.Duration // How often to check (default: 5s)
	MaxWait      time.Duration // Safety timeout (default: 10min)
}

// ProofMonitor detects when WindowPoSt proof computation is complete
// and monitors WinningPoSt wins via Curio's mining_tasks table.
type ProofMonitor interface {
	// IsProofComplete checks if the proof for the given deadline has been computed.
	IsProofComplete(ctx context.Context, spID int64, periodStart int64, deadline uint64) (bool, error)

	// WaitForProofComplete blocks until proof is detected or timeout.
	WaitForProofComplete(ctx context.Context, spID int64, periodStart int64, deadline uint64, cfg WaitConfig) bool

	// CheckWinningPost checks if the miner won a block since the given epoch.
	// Returns nil if no win, or the winning details.
	CheckWinningPost(ctx context.Context, spID int64, sinceEpoch int64) (*WinningPostWin, error)

	// Close releases resources.
	Close() error
}

// DBClient connects to Curio's YugabyteDB/PostgreSQL to monitor proof status.
type DBClient struct {
	db     *sql.DB
	logger *slog.Logger
}

// NewDBClient creates a new Curio database client.
// dsn example: "host=localhost port=5433 user=yugabyte dbname=yugabyte sslmode=disable search_path=curio"
func NewDBClient(dsn string, logger *slog.Logger) (*DBClient, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("open curio db: %w", err)
	}

	// Verify connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping curio db: %w", err)
	}

	// Conservative pool settings — we only do light polling
	db.SetMaxOpenConns(2)
	db.SetMaxIdleConns(1)

	logger.Info("connected to curio database")
	return &DBClient{db: db, logger: logger}, nil
}

// IsProofComplete checks if proof computation is done for ALL partitions in
// the given deadline. A deadline may contain multiple partitions (each up to
// 2349 sectors). Curio computes proofs sequentially per partition and writes
// proof_params upon completion of each. We must wait until every partition
// is done before resuming GPU for inference.
func (c *DBClient) IsProofComplete(ctx context.Context, spID int64, periodStart int64, deadline uint64) (bool, error) {
	var total, completed int
	err := c.db.QueryRowContext(ctx,
		`SELECT COUNT(*),
		        COUNT(CASE WHEN proof_params IS NOT NULL THEN 1 END)
		 FROM wdpost_proofs
		 WHERE sp_id = $1
		   AND proving_period_start = $2
		   AND deadline = $3`,
		spID, periodStart, int64(deadline),
	).Scan(&total, &completed)

	if err != nil {
		return false, fmt.Errorf("query wdpost_proofs: %w", err)
	}

	// No rows yet means Curio hasn't created partition tasks — not complete.
	if total == 0 {
		return false, nil
	}

	if completed < total {
		c.logger.Debug("proof in progress",
			"deadline", deadline,
			"completed_partitions", completed,
			"total_partitions", total,
		)
	}

	return completed == total, nil
}

// WaitForProofComplete polls the database until proof completion is detected or timeout.
func (c *DBClient) WaitForProofComplete(ctx context.Context, spID int64, periodStart int64, deadline uint64, cfg WaitConfig) bool {
	c.logger.Info("waiting for proof completion in curio DB",
		"sp_id", spID,
		"period_start", periodStart,
		"deadline", deadline,
		"poll_interval", cfg.PollInterval,
		"max_wait", cfg.MaxWait,
	)

	timeout := time.After(cfg.MaxWait)
	ticker := time.NewTicker(cfg.PollInterval)
	defer ticker.Stop()

	// Check immediately before first tick
	if done, err := c.IsProofComplete(ctx, spID, periodStart, deadline); err == nil && done {
		c.logger.Info("proof already complete in curio DB")
		return true
	}

	for {
		select {
		case <-ctx.Done():
			return false
		case <-timeout:
			c.logger.Warn("proof detection timed out",
				"sp_id", spID,
				"deadline", deadline,
				"max_wait", cfg.MaxWait,
			)
			return false
		case <-ticker.C:
			done, err := c.IsProofComplete(ctx, spID, periodStart, deadline)
			if err != nil {
				c.logger.Warn("proof check query failed", "error", err)
				continue
			}
			if done {
				c.logger.Info("proof computation complete",
					"sp_id", spID,
					"period_start", periodStart,
					"deadline", deadline,
				)
				return true
			}
		}
	}
}

// WinningPostWin represents a detected winning block from mining_tasks.
type WinningPostWin struct {
	Epoch    int64  `json:"epoch"`
	MinedCID string `json:"mined_cid"`
}

// CheckWinningPost checks if the miner has won any blocks since the given epoch.
// Queries mining_tasks WHERE sp_id=$1 AND won='t' AND epoch > $2.
func (c *DBClient) CheckWinningPost(ctx context.Context, spID int64, sinceEpoch int64) (*WinningPostWin, error) {
	var epoch int64
	var minedCID sql.NullString
	err := c.db.QueryRowContext(ctx,
		`SELECT epoch, mined_cid FROM mining_tasks
		 WHERE sp_id = $1 AND won = true AND epoch > $2
		 ORDER BY epoch DESC LIMIT 1`,
		spID, sinceEpoch,
	).Scan(&epoch, &minedCID)

	if err == sql.ErrNoRows {
		return nil, nil // No wins
	}
	if err != nil {
		return nil, fmt.Errorf("query mining_tasks: %w", err)
	}

	cid := ""
	if minedCID.Valid {
		cid = minedCID.String
	}
	return &WinningPostWin{Epoch: epoch, MinedCID: cid}, nil
}

// Close closes the database connection.
func (c *DBClient) Close() error {
	return c.db.Close()
}

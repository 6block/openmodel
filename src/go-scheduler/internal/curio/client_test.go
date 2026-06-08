package curio

import (
	"context"
	"log/slog"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
)

func newTestClient(t *testing.T) (*DBClient, sqlmock.Sqlmock) {
	t.Helper()
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("create sqlmock: %v", err)
	}
	client := &DBClient{
		db:     db,
		logger: slog.Default(),
	}
	t.Cleanup(func() { db.Close() })
	return client, mock
}

// Single partition, proof complete → should return true.
func TestIsProofComplete_SinglePartition_Done(t *testing.T) {
	client, mock := newTestClient(t)

	mock.ExpectQuery("SELECT COUNT").
		WithArgs(int64(1000), int64(500000), int64(5)).
		WillReturnRows(sqlmock.NewRows([]string{"total", "completed"}).AddRow(1, 1))

	done, err := client.IsProofComplete(context.Background(), 1000, 500000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !done {
		t.Error("expected proof complete for single partition with proof_params")
	}
}

// Single partition, proof not yet computed → should return false.
func TestIsProofComplete_SinglePartition_Pending(t *testing.T) {
	client, mock := newTestClient(t)

	mock.ExpectQuery("SELECT COUNT").
		WithArgs(int64(1000), int64(500000), int64(5)).
		WillReturnRows(sqlmock.NewRows([]string{"total", "completed"}).AddRow(1, 0))

	done, err := client.IsProofComplete(context.Background(), 1000, 500000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if done {
		t.Error("expected proof NOT complete when partition has no proof_params")
	}
}

// No rows yet (Curio hasn't created tasks) → should return false.
func TestIsProofComplete_NoRows(t *testing.T) {
	client, mock := newTestClient(t)

	mock.ExpectQuery("SELECT COUNT").
		WithArgs(int64(1000), int64(500000), int64(5)).
		WillReturnRows(sqlmock.NewRows([]string{"total", "completed"}).AddRow(0, 0))

	done, err := client.IsProofComplete(context.Background(), 1000, 500000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if done {
		t.Error("expected proof NOT complete when no rows exist")
	}
}

// Multi-partition: 3 partitions, only 1 complete → should return false.
// This is the critical test: the old code (count > 0) would incorrectly
// return true here, causing premature GPU resume.
func TestIsProofComplete_MultiPartition_Partial(t *testing.T) {
	client, mock := newTestClient(t)

	// 3 total rows, 1 has proof_params
	mock.ExpectQuery("SELECT COUNT").
		WithArgs(int64(1000), int64(500000), int64(5)).
		WillReturnRows(sqlmock.NewRows([]string{"total", "completed"}).AddRow(3, 1))

	done, err := client.IsProofComplete(context.Background(), 1000, 500000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if done {
		t.Error("expected proof NOT complete when only 1 of 3 partitions done — old bug would return true here")
	}
}

// Multi-partition: 3 partitions, 2 complete → should return false.
func TestIsProofComplete_MultiPartition_TwoOfThree(t *testing.T) {
	client, mock := newTestClient(t)

	mock.ExpectQuery("SELECT COUNT").
		WithArgs(int64(1000), int64(500000), int64(5)).
		WillReturnRows(sqlmock.NewRows([]string{"total", "completed"}).AddRow(3, 2))

	done, err := client.IsProofComplete(context.Background(), 1000, 500000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if done {
		t.Error("expected proof NOT complete when only 2 of 3 partitions done")
	}
}

// Multi-partition: all 3 complete → should return true.
func TestIsProofComplete_MultiPartition_AllDone(t *testing.T) {
	client, mock := newTestClient(t)

	mock.ExpectQuery("SELECT COUNT").
		WithArgs(int64(1000), int64(500000), int64(5)).
		WillReturnRows(sqlmock.NewRows([]string{"total", "completed"}).AddRow(3, 3))

	done, err := client.IsProofComplete(context.Background(), 1000, 500000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !done {
		t.Error("expected proof complete when all 3 partitions have proof_params")
	}
}

package curio

import (
	"context"
	"database/sql"
	"errors"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

// IsProofComplete query-error branch (the one block missed by client_test.go).
func TestIsProofComplete_QueryError(t *testing.T) {
	client, mock := newTestClient(t)
	mock.ExpectQuery("SELECT COUNT").
		WithArgs(int64(1), int64(2), int64(3)).
		WillReturnError(errors.New("db down"))
	if _, err := client.IsProofComplete(context.Background(), 1, 2, 3); err == nil {
		t.Fatal("expected an error when the query fails")
	}
}

func TestCheckWinningPost_Win(t *testing.T) {
	client, mock := newTestClient(t)
	mock.ExpectQuery("SELECT epoch, mined_cid FROM mining_tasks").
		WithArgs(int64(1000), int64(500)).
		WillReturnRows(sqlmock.NewRows([]string{"epoch", "mined_cid"}).AddRow(int64(3601228), "bafyMined"))
	win, err := client.CheckWinningPost(context.Background(), 1000, 500)
	if err != nil {
		t.Fatal(err)
	}
	if win == nil || win.Epoch != 3601228 || win.MinedCID != "bafyMined" {
		t.Errorf("got %+v", win)
	}
}

func TestCheckWinningPost_NoRows(t *testing.T) {
	client, mock := newTestClient(t)
	mock.ExpectQuery("SELECT epoch").
		WithArgs(int64(1000), int64(500)).
		WillReturnError(sql.ErrNoRows)
	win, err := client.CheckWinningPost(context.Background(), 1000, 500)
	if err != nil || win != nil {
		t.Errorf("no-win should be (nil,nil), got (%v,%v)", win, err)
	}
}

func TestCheckWinningPost_NullCID(t *testing.T) {
	client, mock := newTestClient(t)
	mock.ExpectQuery("SELECT epoch").
		WithArgs(int64(1000), int64(0)).
		WillReturnRows(sqlmock.NewRows([]string{"epoch", "mined_cid"}).AddRow(int64(42), nil)) // NULL mined_cid
	win, err := client.CheckWinningPost(context.Background(), 1000, 0)
	if err != nil {
		t.Fatal(err)
	}
	if win == nil || win.Epoch != 42 || win.MinedCID != "" {
		t.Errorf("null mined_cid should map to empty string, got %+v", win)
	}
}

func TestCheckWinningPost_QueryError(t *testing.T) {
	client, mock := newTestClient(t)
	mock.ExpectQuery("SELECT epoch").
		WithArgs(int64(1000), int64(0)).
		WillReturnError(errors.New("conn lost"))
	if _, err := client.CheckWinningPost(context.Background(), 1000, 0); err == nil {
		t.Fatal("expected an error on query failure")
	}
}

func proofRows(total, completed int) *sqlmock.Rows {
	return sqlmock.NewRows([]string{"total", "completed"}).AddRow(total, completed)
}

func TestWaitForProofComplete_Immediate(t *testing.T) {
	client, mock := newTestClient(t)
	mock.ExpectQuery("SELECT COUNT").WithArgs(int64(1), int64(2), int64(3)).WillReturnRows(proofRows(1, 1))
	ok := client.WaitForProofComplete(context.Background(), 1, 2, 3,
		WaitConfig{PollInterval: time.Hour, MaxWait: time.Hour})
	if !ok {
		t.Error("expected immediate completion to return true")
	}
}

func TestWaitForProofComplete_Timeout(t *testing.T) {
	client, mock := newTestClient(t)
	// immediate check is pending; no ticker fires (interval > MaxWait) → timeout
	mock.ExpectQuery("SELECT COUNT").WithArgs(int64(1), int64(2), int64(3)).WillReturnRows(proofRows(1, 0))
	ok := client.WaitForProofComplete(context.Background(), 1, 2, 3,
		WaitConfig{PollInterval: time.Hour, MaxWait: 20 * time.Millisecond})
	if ok {
		t.Error("expected timeout to return false")
	}
}

// Covers the per-poll error `continue` plus eventual success.
func TestWaitForProofComplete_PollsThroughErrorUntilComplete(t *testing.T) {
	client, mock := newTestClient(t)
	mock.ExpectQuery("SELECT COUNT").WithArgs(int64(1), int64(2), int64(3)).WillReturnRows(proofRows(1, 0)) // immediate: pending
	mock.ExpectQuery("SELECT COUNT").WithArgs(int64(1), int64(2), int64(3)).WillReturnError(errors.New("transient")) // poll 1: error → continue
	mock.ExpectQuery("SELECT COUNT").WithArgs(int64(1), int64(2), int64(3)).WillReturnRows(proofRows(1, 1)) // poll 2: done
	ok := client.WaitForProofComplete(context.Background(), 1, 2, 3,
		WaitConfig{PollInterval: 5 * time.Millisecond, MaxWait: time.Hour})
	if !ok {
		t.Error("expected eventual completion (through a transient error) to return true")
	}
}

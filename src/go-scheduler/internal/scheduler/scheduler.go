package scheduler

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"openmodel/go-scheduler/internal/curio"
	"openmodel/go-scheduler/internal/lotus"

	pb "openmodel/go-scheduler/proto/sidecar"
)

// Scheduler manages GPU allocation between mining and AI inference.
type Scheduler struct {
	lotus  lotus.Client
	policy YieldPolicy
	sm     *StateMachine
	logger *slog.Logger

	// Event subscribers
	subMu       sync.RWMutex
	subscribers map[int]chan *pb.ScheduleEvent
	nextSubID   int

	// Latest decision cache
	decMu          sync.RWMutex
	latestDecision *YieldDecision

	// Sector cache: tracks which deadlines have sectors to avoid unnecessary yields.
	sectorCacheMu    sync.RWMutex
	sectorCache      map[uint64]int
	sectorCacheEpoch int64

	// Proof completion monitoring via Curio DB.
	proofMonitor       curio.ProofMonitor
	proofMonitorActive atomic.Bool // true while waiting for proof completion
	// Monotonic generation for WinningPoSt resume timers. Each new win bumps it;
	// a timer only resumes if its generation is still current, so an earlier
	// timer cannot resume during a later, still-active win (fixes double-timer race).
	winningGen atomic.Int64
	proofWaitCfg       curio.WaitConfig
	minerID            int64 // sp_id numeric (e.g., 182063 for t0182063)

	// Track completed proofs to prevent re-yielding for the same deadline.
	// A set keyed by (periodStart, deadline) — a single proving period can have
	// several deadlines with sectors, so a single-slot tracker would forget the
	// earlier ones. Bounded in size; an occasional re-yield after a reset is safe.
	completedProofMu sync.RWMutex
	completedProofs  map[proofKey]bool

	// Curio log watcher for fast WinningPoSt detection.
	curioLogWatcher *curio.LogWatcher

	// Optional callback fired on every state transition.
	// Used by health package to update Prometheus metrics without import cycles.
	onStateChange func(state pb.GpuState, reason pb.YieldReason)

	// Retry cadence for the WinningPoSt monitor's initial epoch fetch
	// (overridable in tests). Defaults to 5s.
	initRetryInterval time.Duration

	// When > 0, overrides the WinningPoSt resume delay (tests use a small value).
	winningResumeDelay time.Duration
}

// SetOnStateChange registers a callback that fires whenever the GPU state
// transitions to a new value. Used to wire up Prometheus counters.
func (s *Scheduler) SetOnStateChange(cb func(state pb.GpuState, reason pb.YieldReason)) {
	s.onStateChange = cb
}

// New creates a new Scheduler.
func New(lotusClient lotus.Client, policy YieldPolicy, logger *slog.Logger) *Scheduler {
	return &Scheduler{
		lotus:             lotusClient,
		policy:            policy,
		sm:                NewStateMachine(),
		logger:            logger,
		subscribers:       make(map[int]chan *pb.ScheduleEvent),
		initRetryInterval: 5 * time.Second,
	}
}

// SetCurioLogWatcher configures fast WinningPoSt detection via Curio log tailing.
func (s *Scheduler) SetCurioLogWatcher(watcher *curio.LogWatcher) {
	s.curioLogWatcher = watcher
}

// SetProofMonitor configures real-time proof completion detection.
// When set, the scheduler resumes inference as soon as proof computation
// finishes instead of waiting for the entire deadline window.
func (s *Scheduler) SetProofMonitor(monitor curio.ProofMonitor, minerID int64, cfg curio.WaitConfig) {
	s.proofMonitor = monitor
	s.minerID = minerID
	s.proofWaitCfg = cfg
	s.logger.Info("proof monitor configured",
		"miner_id", minerID,
		"poll_interval", cfg.PollInterval,
		"max_wait", cfg.MaxWait,
	)
}

// Run starts the scheduler's main loop.
func (s *Scheduler) Run(ctx context.Context, pollInterval time.Duration) {
	go s.pollWindowPost(ctx, pollInterval)

	if s.policy.WinningPost.Enabled {
		go s.monitorWinningPost(ctx)
	}

	<-ctx.Done()
	s.logger.Info("scheduler stopped")
}

func (s *Scheduler) pollWindowPost(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	s.checkWindowPost(ctx)
	pollCount := int64(0)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pollCount++
			t0 := time.Now()
			s.checkWindowPost(ctx)
			elapsed := time.Since(t0)
			// Log every 100th poll or if poll took > 5s
			if pollCount%100 == 0 || elapsed > 5*time.Second {
				s.logger.Info("poll heartbeat",
					"poll_count", pollCount,
					"elapsed_ms", elapsed.Milliseconds(),
					"state", s.sm.Current().String(),
				)
			}
		}
	}
}

// sectorCacheRefreshInterval returns how many epochs the per-deadline sector
// cache stays valid before a full refresh. Configurable via policy; defaults to
// 2880 (~24h) only if unset (e.g. zero-value policy in tests).
func (s *Scheduler) sectorCacheRefreshInterval() int64 {
	if s.policy.WindowPost.SectorCacheRefreshEpochs > 0 {
		return int64(s.policy.WindowPost.SectorCacheRefreshEpochs)
	}
	return 2880
}

func (s *Scheduler) checkWindowPost(ctx context.Context) {
	// If proof monitor is actively waiting, skip WindowPoSt evaluation
	// to prevent re-sending WINDOW_POST state that would override resume.
	if s.proofMonitorActive.Load() {
		return
	}

	// If currently in WinningPoSt state, don't override — let the resume timer handle it.
	if s.sm.Current() == StateWinningPost {
		return
	}

	t1 := time.Now()
	info, err := s.lotus.GetProvingDeadline(ctx)
	lotusElapsed := time.Since(t1)
	if lotusElapsed > 2*time.Second {
		s.logger.Warn("slow Lotus API call", "elapsed_ms", lotusElapsed.Milliseconds())
	}
	if err != nil {
		s.logger.Error("failed to get proving deadline", "error", err)
		if s.policy.FailSafeOnDisconnect {
			decision := YieldDecision{
				State:   StateWindowPost,
				Urgency: pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE,
				Reason:  pb.YieldReason_YIELD_REASON_LOTUS_DISCONNECTED,
				Message: "lotus disconnected, fail-safe activated",
			}
			s.applyDecision(&decision)
		}
		return
	}

	// Current deadline has sectors — evaluate it directly.
	if !s.deadlineHasNoSectors(ctx, info) {
		s.evaluateAndApply(ctx, info)
		return
	}

	// Current deadline has NO sectors — look ahead to the next deadline that does.
	s.refreshFullSectorCache(ctx, info)

	nextIdx, found := s.findNextSectorDeadline(info)
	if !found {
		// No deadlines have sectors at all — stay available
		s.logger.Debug("no deadlines with sectors found, staying available")
		decision := YieldDecision{
			State:   StateAvailable,
			Urgency: pb.YieldUrgency_YIELD_URGENCY_NORMAL,
			Reason:  pb.YieldReason_YIELD_REASON_RESUME,
			Message: "no deadlines with sectors",
		}
		s.applyDecisionUnlessWinning(&decision)
		return
	}

	// Build a virtual DeadlineInfo for the upcoming deadline with sectors,
	// so the policy can evaluate graceful/hard-stop thresholds against it.
	nextOpen := s.nextDeadlineOpenEpoch(info, nextIdx)

	// Use the correct PeriodStart: if the target deadline wraps into the
	// next proving period, advance PeriodStart so proof queries match
	// the right period (not the old one).
	periodStart := info.PeriodStart
	if nextIdx <= info.Index {
		// Wrapped around to next period
		periodStart = info.PeriodStart + info.WPoStProvingPeriod
	}

	virtualInfo := &lotus.DeadlineInfo{
		CurrentEpoch:         info.CurrentEpoch,
		PeriodStart:          periodStart,
		Index:                nextIdx,
		Open:                 nextOpen,
		Close:                nextOpen + info.WPoStChallengeWindow,
		WPoStPeriodDeadlines: info.WPoStPeriodDeadlines,
		WPoStProvingPeriod:   info.WPoStProvingPeriod,
		WPoStChallengeWindow: info.WPoStChallengeWindow,
	}

	secondsUntil := virtualInfo.SecondsUntilOpen()
	// Log at INFO when close to threshold (within 10 minutes)
	if secondsUntil <= 600 {
		s.logger.Info("look-ahead: deadline approaching",
			"current_deadline", info.Index,
			"target_deadline", nextIdx,
			"current_epoch", info.CurrentEpoch,
			"target_open_epoch", nextOpen,
			"seconds_until_open", secondsUntil,
			"graceful_threshold", s.policy.WindowPost.GracefulYieldThresholdSec,
		)
	} else {
		s.logger.Debug("look-ahead: next deadline with sectors",
			"current_deadline", info.Index,
			"target_deadline", nextIdx,
			"seconds_until_open", secondsUntil,
		)
	}

	s.evaluateAndApply(ctx, virtualInfo)
}

// evaluateAndApply runs the policy evaluation and applies the decision.
// If entering WINDOW_POST, launches proof monitor if configured.
func (s *Scheduler) evaluateAndApply(ctx context.Context, info *lotus.DeadlineInfo) {
	decision := s.policy.EvaluateWindowPost(info)

	// If policy says WINDOW_POST but we already completed proof for this deadline,
	// skip re-yielding — stay available until the deadline window closes.
	if (decision.State == StateWindowPost || decision.State == StateYielding) &&
		s.isProofAlreadyCompleted(info.PeriodStart, info.Index) {
		s.logger.Debug("skipping yield — proof already completed for this deadline",
			"deadline", info.Index,
			"period_start", info.PeriodStart,
		)
		// If deadline is still open, keep AVAILABLE. When deadline closes and
		// next poll sees a different deadline, this naturally resets.
		decision = YieldDecision{
			State:   StateAvailable,
			Urgency: pb.YieldUrgency_YIELD_URGENCY_NORMAL,
			Reason:  pb.YieldReason_YIELD_REASON_RESUME,
			Message: "proof already completed for this deadline, staying available",
		}
		s.applyDecisionUnlessWinning(&decision)
		return
	}

	prevState := s.sm.Current()
	s.applyDecisionUnlessWinning(&decision)

	// If we just transitioned INTO WindowPost and proof monitor is configured,
	// launch background proof completion detection.
	if decision.State == StateWindowPost && prevState != StateWindowPost &&
		s.proofMonitor != nil && s.policy.WindowPost.ProofDetectionEnabled {
		s.proofMonitorActive.Store(true)
		go s.waitForProofAndResume(ctx, info)
	}

	s.logger.Debug("window_post check",
		"epoch", info.CurrentEpoch,
		"deadline_index", info.Index,
		"deadline_open", info.Open,
		"seconds_until_open", info.SecondsUntilOpen(),
		"decision_state", decision.State.String(),
		"current_state", prevState.String(),
	)
}

// findNextSectorDeadline scans the sector cache for the next deadline (after current)
// that has sectors. Returns the deadline index and true if found.
func (s *Scheduler) findNextSectorDeadline(info *lotus.DeadlineInfo) (uint64, bool) {
	s.sectorCacheMu.RLock()
	defer s.sectorCacheMu.RUnlock()

	numDeadlines := info.WPoStPeriodDeadlines
	if numDeadlines == 0 {
		numDeadlines = 48
	}

	// Search from current+1 through all deadlines (wrapping around)
	for i := uint64(1); i < numDeadlines; i++ {
		idx := (info.Index + i) % numDeadlines
		if sectors, ok := s.sectorCache[idx]; ok && sectors > 0 {
			return idx, true
		}
	}
	return 0, false
}

// nextDeadlineOpenEpoch calculates the open epoch for a target deadline index.
func (s *Scheduler) nextDeadlineOpenEpoch(info *lotus.DeadlineInfo, targetIdx uint64) int64 {
	if targetIdx > info.Index {
		// Same proving period
		return info.PeriodStart + int64(targetIdx)*info.WPoStChallengeWindow
	}
	// Next proving period (wrapped around)
	return info.PeriodStart + info.WPoStProvingPeriod + int64(targetIdx)*info.WPoStChallengeWindow
}

// refreshFullSectorCache populates the sector cache for all 48 deadlines.
// Only queries Lotus if the cache is stale or incomplete.
func (s *Scheduler) refreshFullSectorCache(ctx context.Context, info *lotus.DeadlineInfo) {
	s.sectorCacheMu.RLock()
	cacheEpoch := s.sectorCacheEpoch
	cacheLen := len(s.sectorCache)
	s.sectorCacheMu.RUnlock()

	numDeadlines := info.WPoStPeriodDeadlines
	if numDeadlines == 0 {
		numDeadlines = 48
	}

	// Only refresh if cache is stale or incomplete
	stale := (info.CurrentEpoch - cacheEpoch) > s.sectorCacheRefreshInterval()
	incomplete := uint64(cacheLen) < numDeadlines
	if !stale && !incomplete {
		return
	}

	s.logger.Info("refreshing full sector cache for all deadlines")

	s.sectorCacheMu.Lock()
	if s.sectorCache == nil {
		s.sectorCache = make(map[uint64]int)
	}
	s.sectorCacheMu.Unlock()

	for i := uint64(0); i < numDeadlines; i++ {
		// Skip if already cached and not stale
		if !stale {
			s.sectorCacheMu.RLock()
			_, exists := s.sectorCache[i]
			s.sectorCacheMu.RUnlock()
			if exists {
				continue
			}
		}

		ds, err := s.lotus.GetDeadlineSectors(ctx, i)
		if err != nil {
			s.logger.Warn("failed to get deadline sectors",
				"deadline", i, "error", err)
			continue
		}

		s.sectorCacheMu.Lock()
		s.sectorCache[i] = ds.Sectors
		s.sectorCacheMu.Unlock()

		if ds.Sectors > 0 {
			s.logger.Info("sector cache: deadline has sectors",
				"deadline", i,
				"sectors", ds.Sectors,
				"partitions", ds.Partitions,
			)
		}
	}

	s.sectorCacheMu.Lock()
	s.sectorCacheEpoch = info.CurrentEpoch
	s.sectorCacheMu.Unlock()
}

// waitForProofAndResume blocks until proof completion is detected, then resumes inference.
func (s *Scheduler) waitForProofAndResume(ctx context.Context, info *lotus.DeadlineInfo) {
	defer s.proofMonitorActive.Store(false)

	s.logger.Info("proof monitor started — waiting for proof completion",
		"deadline", info.Index,
		"period_start", info.PeriodStart,
		"miner_id", s.minerID,
	)

	detected := s.proofMonitor.WaitForProofComplete(
		ctx, s.minerID, info.PeriodStart, info.Index, s.proofWaitCfg,
	)

	if detected {
		s.logger.Info("proof computation complete, resuming inference",
			"deadline", info.Index,
			"period_start", info.PeriodStart,
		)
	} else {
		s.logger.Warn("proof detection timed out or cancelled, resuming inference anyway",
			"deadline", info.Index,
		)
	}

	// Record that this deadline's proof is done, so checkWindowPost
	// won't re-yield for it while the deadline window is still open.
	s.markProofCompleted(info.PeriodStart, info.Index)

	resume := YieldDecision{
		State:   StateAvailable,
		Urgency: pb.YieldUrgency_YIELD_URGENCY_NORMAL,
		Reason:  pb.YieldReason_YIELD_REASON_RESUME,
		Message: "proof complete, GPU available for inference",
	}
	// Do NOT clobber an active WINNING_POST: if a block-production window is in
	// progress, the WinningPoSt resume timer (which re-arms while a proof is active)
	// owns the final resume. We've recorded the proof as complete above, and our
	// deferred proofMonitorActive=false lets that timer proceed.
	s.applyDecisionUnlessWinning(&resume)
}

// proofKey identifies a completed WindowPoSt proof by proving period + deadline.
type proofKey struct {
	period   int64
	deadline uint64
}

func (s *Scheduler) markProofCompleted(periodStart int64, deadline uint64) {
	s.completedProofMu.Lock()
	defer s.completedProofMu.Unlock()
	if s.completedProofs == nil {
		s.completedProofs = make(map[proofKey]bool)
	}
	// Bound memory: an occasional re-yield after a reset is harmless (yielding is
	// always safe; the set is only an optimization to avoid redundant yields).
	if len(s.completedProofs) > 128 {
		s.completedProofs = make(map[proofKey]bool)
	}
	s.completedProofs[proofKey{period: periodStart, deadline: deadline}] = true
}

func (s *Scheduler) isProofAlreadyCompleted(periodStart int64, deadline uint64) bool {
	s.completedProofMu.RLock()
	defer s.completedProofMu.RUnlock()
	return s.completedProofs[proofKey{period: periodStart, deadline: deadline}]
}

func (s *Scheduler) deadlineHasNoSectors(ctx context.Context, info *lotus.DeadlineInfo) bool {
	s.sectorCacheMu.RLock()
	cachedSectors, cached := s.sectorCache[info.Index]
	cacheEpoch := s.sectorCacheEpoch
	s.sectorCacheMu.RUnlock()

	needsRefresh := !cached || (info.CurrentEpoch-cacheEpoch) > s.sectorCacheRefreshInterval()
	if needsRefresh {
		ds, err := s.lotus.GetDeadlineSectors(ctx, info.Index)
		if err != nil {
			s.logger.Warn("failed to get deadline sectors, assuming has sectors (safe default)",
				"deadline", info.Index, "error", err)
			return false
		}

		s.sectorCacheMu.Lock()
		if s.sectorCache == nil {
			s.sectorCache = make(map[uint64]int)
		}
		s.sectorCache[info.Index] = ds.Sectors
		s.sectorCacheEpoch = info.CurrentEpoch
		s.sectorCacheMu.Unlock()

		cachedSectors = ds.Sectors
		s.logger.Info("refreshed sector cache",
			"deadline", info.Index,
			"sectors", ds.Sectors,
			"partitions", ds.Partitions,
		)
	}

	return cachedSectors == 0
}

// monitorWinningPost detects actual block wins using two methods:
// 1. Primary: tail Curio log for "WinPostTask won election" (~instant, before proof computation)
// 2. Fallback: poll mining_tasks.won=true every 5s (after proof computation, ~4s delay)
func (s *Scheduler) monitorWinningPost(ctx context.Context) {
	if s.proofMonitor == nil {
		s.logger.Warn("WinningPoSt monitoring requires proof monitor (Curio DB), disabled")
		return
	}

	// Establish the starting epoch. A transient Lotus failure at startup must
	// NOT permanently disable WinningPoSt detection (which would silently lose
	// block rewards) — retry until Lotus answers or the context is cancelled.
	var lastCheckedEpoch int64
	for {
		info, err := s.lotus.GetProvingDeadline(ctx)
		if err == nil {
			lastCheckedEpoch = info.CurrentEpoch
			break
		}
		s.logger.Warn("WinningPoSt monitor: initial epoch fetch failed, retrying", "error", err)
		select {
		case <-ctx.Done():
			return
		case <-time.After(s.initRetryInterval):
		}
	}

	// Start log watcher (primary, fastest detection)
	var logWinCh <-chan struct{}
	if s.curioLogWatcher != nil {
		ch, err := s.curioLogWatcher.Watch(ctx)
		if err != nil {
			s.logger.Warn("failed to start Curio log watcher, using DB polling only", "error", err)
		} else {
			logWinCh = ch
		}
	}

	s.logger.Info("WinningPoSt monitor started",
		"since_epoch", lastCheckedEpoch,
		"log_watcher", logWinCh != nil,
		"db_poll_interval", "5s",
	)

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return

		case _, ok := <-logWinCh:
			if !ok {
				s.logger.Warn("Curio log watcher closed, falling back to DB polling only")
				logWinCh = nil
				continue
			}
			s.logger.Info("WinningPoSt detected via Curio log (pre-computation)!")
			s.triggerWinningYield(ctx, "detected via Curio log before proof computation")

		case <-ticker.C:
			win, err := s.proofMonitor.CheckWinningPost(ctx, s.minerID, lastCheckedEpoch)
			if err != nil {
				s.logger.Warn("WinningPoSt DB check failed", "error", err)
				continue
			}
			if win == nil {
				continue
			}

			lastCheckedEpoch = win.Epoch

			// Only trigger if not already in WinningPost state (log watcher may have caught it first)
			if s.sm.Current() == StateWinningPost {
				s.logger.Info("WinningPoSt DB confirmation (already yielding)",
					"epoch", win.Epoch, "mined_cid", win.MinedCID)
				continue
			}

			s.logger.Info("WinningPoSt detected via Curio DB (post-computation)",
				"epoch", win.Epoch, "mined_cid", win.MinedCID)
			s.triggerWinningYield(ctx, "detected via Curio DB after proof computation")
		}
	}
}

// triggerWinningYield applies a WinningPoSt yield decision and schedules resume.
func (s *Scheduler) triggerWinningYield(ctx context.Context, message string) {
	decision := YieldDecision{
		State:                  StateWinningPost,
		Urgency:                pb.YieldUrgency_YIELD_URGENCY_IMMEDIATE,
		Reason:                 pb.YieldReason_YIELD_REASON_WINNING_POST,
		Message:                "WinningPoSt: " + message,
		SecondsUntilNextChange: int64(s.policy.WinningPost.ResumeDelaySec),
	}
	s.applyDecision(&decision)

	// Claim a generation for this win. A later win bumps the generation, so this
	// timer becomes stale and must NOT resume (the later win's timer owns resume).
	gen := s.winningGen.Add(1)

	resumeDelay := time.Duration(s.policy.WinningPost.ResumeDelaySec) * time.Second
	if s.winningResumeDelay > 0 {
		resumeDelay = s.winningResumeDelay
	}

	go func() {
		timer := time.NewTimer(resumeDelay)
		defer timer.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-timer.C:
				// A newer WinningPoSt occurred — its window is still active and its
				// own timer will handle resume. Resuming now could put inference on
				// the GPU during block production.
				if s.winningGen.Load() != gen {
					s.logger.Info("WinningPoSt resume timer superseded by a newer win, skipping resume",
						"timer_gen", gen, "current_gen", s.winningGen.Load())
					return
				}
				// A WindowPoSt proof is still being computed — the GPU must stay
				// yielded. Re-arm and check again rather than abandoning resume:
				// the proof monitor no longer resumes on its own while WINNING_POST
				// is active (it would clobber it), so THIS timer owns the final
				// resume once the proof finishes. Bounded by the proof monitor's
				// own MaxWait, after which proofMonitorActive clears.
				if s.proofMonitorActive.Load() {
					s.logger.Info("WinningPoSt window elapsed but WindowPoSt proof still active; re-arming resume check")
					timer.Reset(resumeDelay)
					continue
				}
				resume := YieldDecision{
					State:   StateAvailable,
					Urgency: pb.YieldUrgency_YIELD_URGENCY_NORMAL,
					Reason:  pb.YieldReason_YIELD_REASON_RESUME,
					Message: "WinningPoSt complete, GPU available for inference",
				}
				s.applyDecision(&resume)
				return
			}
		}
	}()
}

func (s *Scheduler) applyDecision(decision *YieldDecision) {
	changed := s.sm.Transition(decision.State)

	s.decMu.Lock()
	s.latestDecision = decision
	s.decMu.Unlock()

	if changed {
		s.logger.Info("GPU state changed",
			"state", decision.State.String(),
			"reason", decision.Reason.String(),
			"message", decision.Message,
		)
		s.broadcastEvent(decision)
		if s.onStateChange != nil {
			s.onStateChange(decision.State, decision.Reason)
		}
	}
}

// applyDecisionUnlessWinning applies a decision only if the GPU is NOT currently in
// WINNING_POST. Used by the WindowPoSt evaluation and the proof-monitor resume so
// they can never clobber an in-progress block-production yield — the check-and-set
// is atomic in the state machine, closing the TOCTOU window between checkWindowPost's
// slow Lotus call and a concurrent WinningPoSt win (audit HIGH fix).
//
// A SAME-STATE no-op (e.g. AVAILABLE→AVAILABLE, which is what every 15s poll produces
// for a worker in steady service) still REFRESHES latestDecision — without the
// broadcast/onStateChange side effects. B1's /ready seconds_until_change is served
// from latestDecision; before this refresh existed the countdown froze at whatever
// value accompanied the last real state change (a resume decision carries none → the
// field read 0 for hours) and the gateway's predictive de-prioritization never saw a
// live countdown while the worker was servable — exactly when it matters.
func (s *Scheduler) applyDecisionUnlessWinning(decision *YieldDecision) {
	if !s.sm.TransitionUnless(decision.State, StateWinningPost) {
		if s.sm.Current() == StateWinningPost {
			// Blocked to protect block production: keep the WINNING_POST decision intact.
			if decision.State != StateWinningPost {
				s.logger.Info("skipped GPU state change to protect active WINNING_POST",
					"attempted_state", decision.State.String(), "reason", decision.Reason.String())
			}
			return
		}
		// Same-state no-op: refresh the decision (keeps the B1 countdown live) but do
		// not log/broadcast — nothing changed for subscribers. A win landing between
		// the two checks above at worst leaves a stale decision for one WinningPoSt
		// window (≤30s); the resume path writes unconditionally and self-heals it.
		s.decMu.Lock()
		s.latestDecision = decision
		s.decMu.Unlock()
		return
	}
	s.decMu.Lock()
	s.latestDecision = decision
	s.decMu.Unlock()
	s.logger.Info("GPU state changed",
		"state", decision.State.String(),
		"reason", decision.Reason.String(),
		"message", decision.Message,
	)
	s.broadcastEvent(decision)
	if s.onStateChange != nil {
		s.onStateChange(decision.State, decision.Reason)
	}
}

func (s *Scheduler) broadcastEvent(decision *YieldDecision) {
	event := &pb.ScheduleEvent{
		State:                decision.State,
		Urgency:              decision.Urgency,
		SecondsUntilDeadline: decision.SecondsUntilNextChange,
		Reason:               decision.Reason,
		Message:              decision.Message,
	}

	s.subMu.RLock()
	defer s.subMu.RUnlock()

	for id, ch := range s.subscribers {
		select {
		case ch <- event:
		default:
			s.logger.Warn("subscriber buffer full, dropping event", "subscriber_id", id)
		}
	}
}

// Subscribe registers a new event subscriber and returns its channel and ID.
func (s *Scheduler) Subscribe() (int, <-chan *pb.ScheduleEvent) {
	s.subMu.Lock()
	defer s.subMu.Unlock()

	id := s.nextSubID
	s.nextSubID++
	ch := make(chan *pb.ScheduleEvent, 32)
	s.subscribers[id] = ch
	return id, ch
}

// Unsubscribe removes an event subscriber.
func (s *Scheduler) Unsubscribe(id int) {
	s.subMu.Lock()
	defer s.subMu.Unlock()

	if ch, ok := s.subscribers[id]; ok {
		close(ch)
		delete(s.subscribers, id)
	}
}

// CurrentState returns the current GPU state.
func (s *Scheduler) CurrentState() GpuState {
	return s.sm.Current()
}

// LatestDecision returns the most recent yield decision.
func (s *Scheduler) LatestDecision() *YieldDecision {
	s.decMu.RLock()
	defer s.decMu.RUnlock()
	return s.latestDecision
}

// SetLatestDecisionForTest injects a decision so tests can exercise the gRPC handler
// and event-broadcast paths (e.g. forward-compat enum passthrough) without driving
// the full polling loop. Test-only seam; not used in production.
func (s *Scheduler) SetLatestDecisionForTest(d *YieldDecision) {
	s.decMu.Lock()
	defer s.decMu.Unlock()
	s.latestDecision = d
}

// SectorCacheSnapshot returns a copy of the current sector cache for debugging.
func (s *Scheduler) SectorCacheSnapshot() map[uint64]int {
	s.sectorCacheMu.RLock()
	defer s.sectorCacheMu.RUnlock()

	snapshot := make(map[uint64]int, len(s.sectorCache))
	for k, v := range s.sectorCache {
		snapshot[k] = v
	}
	return snapshot
}

// WinningPostStatus is the result of a live WinningPoSt check via Curio DB.
type WinningPostStatus struct {
	CurrentEpoch int64  `json:"current_epoch"`
	LastWinEpoch int64  `json:"last_win_epoch,omitempty"`
	LastWinCID   string `json:"last_win_cid,omitempty"`
	Error        string `json:"error,omitempty"`
	CurioDBOK    bool   `json:"curio_db_ok"`
}

// CheckWinningPostNow queries Curio DB for recent block wins.
func (s *Scheduler) CheckWinningPostNow(ctx context.Context) WinningPostStatus {
	status := WinningPostStatus{}

	if s.proofMonitor == nil {
		status.Error = "proof monitor not configured"
		return status
	}

	info, err := s.lotus.GetProvingDeadline(ctx)
	if err != nil {
		status.Error = "get epoch failed: " + err.Error()
		return status
	}
	status.CurrentEpoch = info.CurrentEpoch

	// Check for the most recent win (look back ~24h = 2880 epochs)
	win, err := s.proofMonitor.CheckWinningPost(ctx, s.minerID, info.CurrentEpoch-2880)
	if err != nil {
		status.Error = "curio DB query failed: " + err.Error()
		return status
	}
	status.CurioDBOK = true

	if win != nil {
		status.LastWinEpoch = win.Epoch
		status.LastWinCID = win.MinedCID
	}

	return status
}

// TriggerWinningPost manually triggers a WinningPoSt yield event.
// Works in any mode (dev or prod), used for testing via debug endpoint.
func (s *Scheduler) TriggerWinningPost(ctx context.Context) {
	s.logger.Info("WinningPoSt manually triggered via debug endpoint")
	s.triggerWinningYield(ctx, "manually triggered for testing")
}

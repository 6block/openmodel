"""EnginePool: Multi-GPU engine abstraction supporting tensor-parallel and multi-instance modes.

EnginePool implements InferenceEngine, so it is a drop-in replacement for any
single engine. The rest of the system (API server, scheduler listener) can use
it without modification.
"""

import asyncio
import logging
import time
from typing import Any

from .engine import (
    InferenceEngine, EngineState, EngineStatus, YieldUrgency,
    create_engine,
)

logger = logging.getLogger(__name__)


class EnginePool(InferenceEngine):
    """Manages one or more InferenceEngine instances across GPUs.

    Mode A (tensor_parallel):
        Single engine spanning all GPUs via vLLM NCCL.
        pause/resume/generate all operate on the single instance.

    Mode B (multi_instance):
        N independent engines, one per GPU (or device group).
        pause/resume fan out to all engines via asyncio.gather.
        generate routes via load balancing (round-robin or least-busy).
    """

    def __init__(self, mode: str, engines: list[InferenceEngine],
                 lb_strategy: str = "round_robin"):
        self._mode = mode
        self._engines = engines
        self._lb_strategy = lb_strategy
        self._rr_index = 0  # round-robin counter
        # Set when a pause is requested; the sequential multi_instance resume
        # loop checks this between engines and aborts so it never loads
        # not-yet-resumed engines onto a GPU that is now needed for mining.
        self._pause_requested = False
        # True for the whole duration of a pool-wide model switch. In
        # multi_instance mode engines switch sequentially, so the not-yet-switched
        # engines keep reporting RUNNING and loaded_model flips to the new model as
        # soon as engine 0 finishes — which would make status() look "done" while
        # engines 1..N are still reloading. While this flag is set, status() reports
        # LOADING so the gateway keeps the worker out of routing until EVERY engine
        # has switched (otherwise requests routed mid-switch hang on a reloading engine).
        self._switching = False
        # Engine-level session affinity: pin a session to one engine so vLLM's
        # per-engine prefix cache is actually reused (worker-level affinity alone
        # isn't enough on a multi-instance worker — requests scatter across the N
        # engines and each has its own cache). Maps session key → (engine, expiry).
        # Falls back to load balancing when the pinned engine is gone or much busier
        # than the least-loaded one, so a hot session can't starve an engine.
        self._engine_sessions: dict[str, tuple[InferenceEngine, float]] = {}
        self._engine_session_ttl = 600.0  # seconds
        self._engine_affinity_slack = 2   # stick unless pinned engine is this many reqs busier than the least-loaded

    @property
    def engine_count(self) -> int:
        return len(self._engines)

    @property
    def mode(self) -> str:
        return self._mode

    # --- InferenceEngine interface ---

    async def start(self, model_path: str) -> None:
        """Start all engines. For multi_instance, start concurrently."""
        logger.info("starting engine pool: mode=%s, engines=%d",
                     self._mode, len(self._engines))
        if self._mode == "tensor_parallel":
            await self._engines[0].start(model_path)
        else:
            await asyncio.gather(*[e.start(model_path) for e in self._engines])
        logger.info("engine pool started")

    async def start_paused(self, model_path: str) -> None:
        """Record model path and set engines to PAUSED without loading.

        Used when the scheduler indicates GPU is not available at startup
        (e.g., mining is active). Engines will load when resume() is called.
        """
        logger.info("starting engine pool in PAUSED state (model=%s)", model_path)
        for e in self._engines:
            e._model = model_path
            e._state = EngineState.PAUSED
        logger.info("engine pool paused — will load model on resume")

    def request_pause(self) -> None:
        """Signal an in-flight resume loop to abort ASAP (non-blocking, sync).

        Called by the scheduler listener the moment a yield is detected, even
        while a long multi_instance resume is still running, so the resume loop
        stops loading further engines before pause() runs.
        """
        self._pause_requested = True

    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        """Pause ALL engines (all-or-nothing yield)."""
        self._pause_requested = True
        logger.info("pausing engine pool (%d engines, urgency=%s)",
                     len(self._engines), urgency.name)
        await asyncio.gather(*[e.pause(urgency) for e in self._engines])

    async def resume(self) -> None:
        """Resume engines sequentially.

        In multi_instance mode each engine sets CUDA_VISIBLE_DEVICES before
        spawning its vLLM subprocess. Since all engines share the same parent
        process, concurrent resume would race on the env var. Sequential
        resume ensures each subprocess inherits the correct device.

        The loop checks _pause_requested between engines: if a yield arrived
        mid-resume, it stops immediately so already-paused engines are NOT
        reloaded onto a GPU that mining now needs.
        """
        self._pause_requested = False
        logger.info("resuming engine pool (%d engines)", len(self._engines))
        if self._mode == "multi_instance":
            for i, e in enumerate(self._engines):
                if self._pause_requested:
                    logger.warning(
                        "pause requested mid-resume — aborting resume at engine %d/%d "
                        "(remaining engines stay paused)", i + 1, len(self._engines))
                    return
                logger.info("resuming engine %d/%d", i + 1, len(self._engines))
                await e.resume()
        else:
            # tensor_parallel: single engine, no race
            await asyncio.gather(*[e.resume() for e in self._engines])

    async def stop(self) -> None:
        """Stop ALL engines."""
        await asyncio.gather(*[e.stop() for e in self._engines])

    async def switch_model(self, new_model_path: str) -> None:
        """Switch all engines to a different model.

        In multi_instance mode, switch SEQUENTIALLY for the same reason resume() is
        sequential: each engine sets the process-wide CUDA_VISIBLE_DEVICES before
        spawning its vLLM subprocess, so a concurrent switch would race on the env var
        and land multiple subprocesses on the same GPU (audit MEDIUM fix).
        """
        logger.info("switching engine pool to model: %s", new_model_path)
        # Report LOADING for the whole switch (see _switching). Cleared in finally
        # so an aborted/failed switch doesn't leave the worker stuck excluded.
        self._switching = True
        try:
            if self._mode == "multi_instance":
                for i, e in enumerate(self._engines):
                    if self._pause_requested:
                        # A mining yield arrived mid-switch: stop loading further
                        # engines — the GPU is needed NOW. Engines already switched
                        # keep the new model; the rest switch on the next request
                        # after resume (switch_model per-engine skips when current).
                        logger.warning(
                            "pause requested mid-switch — aborting model switch at "
                            "engine %d/%d", i + 1, len(self._engines))
                        return
                    logger.info("switching engine %d/%d", i + 1, len(self._engines))
                    await e.switch_model(new_model_path)
            else:
                await asyncio.gather(*[e.switch_model(new_model_path) for e in self._engines])
            logger.info("engine pool model switch complete")
        finally:
            self._switching = False

    @property
    def current_model(self) -> str:
        """Return the model currently loaded in the pool."""
        if self._engines:
            return self._engines[0].current_model
        return ""

    def is_available(self) -> bool:
        if self._mode == "tensor_parallel":
            return self._engines[0].is_available()
        return any(e.is_available() for e in self._engines)

    def status(self) -> EngineStatus:
        """Aggregate status across all engines."""
        if self._mode == "tensor_parallel":
            s = self._engines[0].status()
            if self._switching and s.state == EngineState.RUNNING:
                return EngineStatus(state=EngineState.LOADING,
                                    active_requests=s.active_requests,
                                    loaded_model=s.loaded_model, gpu_id=s.gpu_id)
            return s

        statuses = [e.status() for e in self._engines]
        states = [s.state for s in statuses]
        if any(s == EngineState.RUNNING for s in states):
            # At least one engine can serve — pool is (degraded but) available.
            agg_state = EngineState.RUNNING
        elif all(s == EngineState.DEAD for s in states):
            # Every core crashed: report DEAD so /health stops claiming the
            # node is serviceable and the gateway routes elsewhere (soak fix —
            # dead cores used to keep reporting "running").
            agg_state = EngineState.DEAD
        elif all(s == EngineState.PAUSED for s in states):
            agg_state = EngineState.PAUSED
        elif all(s == EngineState.STOPPED for s in states):
            agg_state = EngineState.STOPPED
        else:
            # Mixed non-serving states (paused/loading/dead) — report paused:
            # not routable right now, recovery/resume is in progress.
            agg_state = EngineState.PAUSED

        # A pool-wide switch is in progress: not-yet-switched engines still report
        # RUNNING (so agg_state would be RUNNING) and loaded_model already shows the
        # new model after engine 0 — but engines 1..N are still reloading. Force
        # LOADING so the gateway keeps the worker excluded until the whole switch ends.
        if self._switching:
            agg_state = EngineState.LOADING

        return EngineStatus(
            state=agg_state,
            active_requests=sum(s.active_requests for s in statuses),
            loaded_model=statuses[0].loaded_model if statuses else "",
        )

    async def generate(self, prompt: str, *, session_key: str | None = None, **kwargs):
        """Route to appropriate engine based on mode and LB strategy.

        session_key (optional) pins the request to this session's engine for
        prefix-cache reuse; it is consumed here and NOT forwarded to the engine.
        """
        if self._mode == "tensor_parallel":
            return await self._engines[0].generate(prompt, **kwargs)

        engine = self._pick_engine(session_key)
        return await engine.generate(prompt, **kwargs)

    async def generate_stream(self, prompt: str, *, session_key: str | None = None, **kwargs):
        """Streaming version of generate — yields StreamChunk objects."""
        if self._mode == "tensor_parallel":
            async for chunk in self._engines[0].generate_stream(prompt, **kwargs):
                yield chunk
            return

        engine = self._pick_engine(session_key)
        async for chunk in engine.generate_stream(prompt, **kwargs):
            yield chunk

    # --- Multi-GPU specific ---

    def detailed_status(self) -> dict[str, Any]:
        """Return per-engine status for observability."""
        # Take ONE status() snapshot per engine so the reported fields are
        # mutually consistent — the old code called status() four times per
        # engine, so a mid-call state transition could yield an incoherent row
        # (e.g. state=RUNNING but active_requests from the PAUSED instant).
        engines = []
        for i, e in enumerate(self._engines):
            s = e.status()
            engines.append({
                "index": i,
                "gpu_id": s.gpu_id,
                "state": s.state.value,
                "active_requests": s.active_requests,
                "loaded_model": s.loaded_model,
            })
        return {
            "mode": self._mode,
            "engine_count": len(self._engines),
            "lb_strategy": self._lb_strategy,
            "engines": engines,
        }

    def _pick_engine(self, session_key: str | None = None) -> InferenceEngine:
        """Select an engine for the next request.

        With a session_key, prefer the engine this session last used (so vLLM's
        prefix cache is reused) — unless that engine is gone or much busier than the
        least-loaded available engine, in which case fall back to normal balancing.
        """
        available = [e for e in self._engines if e.is_available()]
        if not available:
            raise RuntimeError("No engines available for inference")

        now = time.monotonic()
        if session_key:
            ent = self._engine_sessions.get(session_key)
            if ent is not None and ent[1] > now and ent[0] in available:
                pinned = ent[0]
                minload = min(e.status().active_requests for e in available)
                if pinned.status().active_requests <= minload + self._engine_affinity_slack:
                    self._engine_sessions[session_key] = (pinned, now + self._engine_session_ttl)
                    return pinned

        if self._lb_strategy == "least_busy":
            chosen = min(available, key=lambda e: e.status().active_requests)
        else:
            # round_robin (default)
            idx = self._rr_index % len(available)
            self._rr_index += 1
            chosen = available[idx]

        if session_key:
            self._engine_sessions[session_key] = (chosen, now + self._engine_session_ttl)
            if len(self._engine_sessions) > 4096:  # bound memory: drop a batch of expired keys
                for k in [k for k, v in self._engine_sessions.items() if v[1] <= now][:512]:
                    del self._engine_sessions[k]
        return chosen


def create_engine_pool(cfg) -> EnginePool:
    """Build an EnginePool from AppConfig.

    Handles both tensor_parallel and multi_instance modes.
    Single-GPU configurations are wrapped in a 1-engine pool for uniformity.
    """
    from ..config import AppConfig
    assert isinstance(cfg, AppConfig)

    multi_gpu = cfg.inference.multi_gpu
    mode = multi_gpu.mode
    device_ids = multi_gpu.device_ids
    engine_type = cfg.inference.engine

    logger.info("creating engine pool: type=%s, mode=%s, gpu_count=%d, devices=%s",
                engine_type, mode, multi_gpu.gpu_count, device_ids)

    if mode == "tensor_parallel":
        # Single engine spanning all GPUs
        engine = create_engine(
            engine_type,
            gpu_memory_utilization=cfg.inference.gpu_memory_utilization,
            max_model_len=cfg.inference.max_model_len,
            tensor_parallel_size=multi_gpu.gpu_count,
            device_ids=device_ids,
            gpu_id=0,  # mock: show as GPU 0
            enforce_eager=cfg.inference.enforce_eager,
        )
        return EnginePool(mode="tensor_parallel", engines=[engine])

    elif mode == "multi_instance":
        # One engine per device
        engines = []
        for dev_id in device_ids:
            engine = create_engine(
                engine_type,
                gpu_memory_utilization=cfg.inference.gpu_memory_utilization,
                max_model_len=cfg.inference.max_model_len,
                tensor_parallel_size=1,
                device_ids=[dev_id],
                gpu_id=dev_id,  # mock: show GPU ID
                enforce_eager=cfg.inference.enforce_eager,
            )
            engines.append(engine)
        return EnginePool(
            mode="multi_instance",
            engines=engines,
            lb_strategy=cfg.inference.multi_instance.load_balancer,
        )

    else:
        raise ValueError(f"Unknown multi_gpu mode: {mode}")

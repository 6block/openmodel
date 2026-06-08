"""EnginePool: Multi-GPU engine abstraction supporting tensor-parallel and multi-instance modes.

EnginePool implements InferenceEngine, so it is a drop-in replacement for any
single engine. The rest of the system (API server, scheduler listener) can use
it without modification.
"""

import asyncio
import logging
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

    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        """Pause ALL engines (all-or-nothing yield)."""
        logger.info("pausing engine pool (%d engines, urgency=%s)",
                     len(self._engines), urgency.name)
        await asyncio.gather(*[e.pause(urgency) for e in self._engines])

    async def resume(self) -> None:
        """Resume engines sequentially.

        In multi_instance mode each engine sets CUDA_VISIBLE_DEVICES before
        spawning its vLLM subprocess. Since all engines share the same parent
        process, concurrent resume would race on the env var. Sequential
        resume ensures each subprocess inherits the correct device.
        """
        logger.info("resuming engine pool (%d engines)", len(self._engines))
        if self._mode == "multi_instance":
            for i, e in enumerate(self._engines):
                logger.info("resuming engine %d/%d", i + 1, len(self._engines))
                await e.resume()
        else:
            # tensor_parallel: single engine, no race
            await asyncio.gather(*[e.resume() for e in self._engines])

    async def stop(self) -> None:
        """Stop ALL engines."""
        await asyncio.gather(*[e.stop() for e in self._engines])

    async def switch_model(self, new_model_path: str) -> None:
        """Switch all engines to a different model."""
        logger.info("switching engine pool to model: %s", new_model_path)
        await asyncio.gather(*[e.switch_model(new_model_path) for e in self._engines])
        logger.info("engine pool model switch complete")

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
            return self._engines[0].status()

        statuses = [e.status() for e in self._engines]
        # Use the "worst" state: if any is paused, report paused
        states = [s.state for s in statuses]
        if all(s == EngineState.RUNNING for s in states):
            agg_state = EngineState.RUNNING
        elif all(s == EngineState.PAUSED for s in states):
            agg_state = EngineState.PAUSED
        elif all(s == EngineState.STOPPED for s in states):
            agg_state = EngineState.STOPPED
        else:
            # Mixed state (some running, some paused) — report running
            # since we can still serve requests
            agg_state = EngineState.RUNNING

        return EngineStatus(
            state=agg_state,
            active_requests=sum(s.active_requests for s in statuses),
            loaded_model=statuses[0].loaded_model if statuses else "",
        )

    async def generate(self, prompt: str, **kwargs):
        """Route to appropriate engine based on mode and LB strategy."""
        if self._mode == "tensor_parallel":
            return await self._engines[0].generate(prompt, **kwargs)

        engine = self._pick_engine()
        return await engine.generate(prompt, **kwargs)

    async def generate_stream(self, prompt: str, **kwargs):
        """Streaming version of generate — yields StreamChunk objects."""
        if self._mode == "tensor_parallel":
            async for chunk in self._engines[0].generate_stream(prompt, **kwargs):
                yield chunk
            return

        engine = self._pick_engine()
        async for chunk in engine.generate_stream(prompt, **kwargs):
            yield chunk

    # --- Multi-GPU specific ---

    def detailed_status(self) -> dict[str, Any]:
        """Return per-engine status for observability."""
        return {
            "mode": self._mode,
            "engine_count": len(self._engines),
            "lb_strategy": self._lb_strategy,
            "engines": [
                {
                    "index": i,
                    "gpu_id": e.status().gpu_id,
                    "state": e.status().state.value,
                    "active_requests": e.status().active_requests,
                    "loaded_model": e.status().loaded_model,
                }
                for i, e in enumerate(self._engines)
            ],
        }

    def _pick_engine(self) -> InferenceEngine:
        """Select an engine for the next request using the configured LB strategy."""
        available = [e for e in self._engines if e.is_available()]
        if not available:
            raise RuntimeError("No engines available for inference")

        if self._lb_strategy == "least_busy":
            return min(available, key=lambda e: e.status().active_requests)
        else:
            # round_robin (default)
            idx = self._rr_index % len(available)
            self._rr_index += 1
            return available[idx]


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

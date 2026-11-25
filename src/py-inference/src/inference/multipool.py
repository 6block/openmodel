"""Multi-pool aggregator: several engine pools, one per modality, on disjoint GPUs.

WHY ONE PROCESS, MANY POOLS. A provider's 8 cards should serve text, image generation and
speech at the same time, one modality per card set (design doc §5). Running each modality
as its own registered worker would multiply registrations, mTLS endpoints and mining
yields; one worker that owns several pools keeps the provider's setup unchanged and makes
the mining yield a single, all-or-nothing operation.

WHAT THIS CLASS IS, AND IS NOT. It is the top-level `engine` the scheduler listener and
the API server talk to. It does not run models itself: every request is handed to the
pool that OWNS the requested model, and every lifecycle call (pause / resume /
request_pause / stop) fans out to all pools. It does NOT pick a pool for a model it does
not know — an unknown model is a clear error, never a silent fall-through to "pool 0",
which would hand a speech request to a text model.

THE YIELD CONTRACT IS THE POINT. The scheduler gives the GPUs to mining only when this
object reports PAUSED, and mining needs EVERY card. So PAUSED is reported only when every
pool reports PAUSED; a mixed state reports UNLOADING / LOADING, never PAUSED. A pool that
says it is paused while still holding a card is exactly the failure P2 found on real
hardware (an IMMEDIATE yield reported PAUSED at 9480 MiB), and the per-engine fixes for it
only help if the aggregate never claims more than its slowest pool.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from .engine import EngineState, EngineStatus, InferenceEngine, YieldUrgency

logger = logging.getLogger(__name__)


def normalize_model_id(model: str) -> str:
    """Spelling-insensitive model key, matching the gateway's own normalisation.

    "/models/Qwen--Qwen3-32B-AWQ" and "Qwen/Qwen3-32B-AWQ" name the same model; the
    gateway accepts either on a request, so the worker must too.
    """
    s = (model or "").strip()
    if s.startswith("/models/"):
        s = s[len("/models/"):]
    s = s.lstrip("/").replace("--", "/")
    return s.lower()


@dataclass
class Pool:
    """One modality's engine pool and the models it may serve."""
    name: str
    engine: InferenceEngine
    devices: list[int]
    # Models this pool may switch to. The resident model is always included.
    supported_models: list[str] = field(default_factory=list)
    # Opt-in card sharing for small engines (speech): a card may hold several pools
    # only if EVERY pool on it sets this. Each engine still confirms its own release.
    share_devices: bool = False

    def serves(self, model: str) -> bool:
        key = normalize_model_id(model)
        if not key:
            return False
        if normalize_model_id(self.engine.current_model) == key:
            return True
        return any(normalize_model_id(m) == key for m in self.supported_models)


class MultiPoolEngine(InferenceEngine):
    """Aggregates several pools behind the single-engine interface."""

    def __init__(self, pools: list[Pool]):
        if not pools:
            raise ValueError("MultiPoolEngine needs at least one pool")
        seen: dict[int, Pool] = {}
        for p in pools:
            for d in p.devices:
                prev = seen.get(d)
                if prev is not None and not (prev.share_devices and p.share_devices):
                    # Two engines on one card would fight for VRAM and OOM at the
                    # worst moment. Refuse at construction, where the message is clear —
                    # unless both pools explicitly opted in to sharing.
                    raise ValueError(f"GPU {d} is assigned to both pool {prev.name!r} and pool {p.name!r}")
                seen[d] = p
        self._pools = pools
        self._pause_requested = False

    # --- composition --------------------------------------------------------------

    @property
    def pools(self) -> list[Pool]:
        return list(self._pools)

    def owner(self, model: str | None) -> Pool | None:
        """The pool that serves `model`, or None.

        "default" / empty means "whatever the primary pool has", preserving the
        single-engine behaviour for clients that never name a model.
        """
        if not model or model == "default":
            return self._pools[0]
        for p in self._pools:
            if p.serves(model):
                return p
        return None

    def engine_for(self, model: str | None) -> InferenceEngine | None:
        p = self.owner(model)
        return p.engine if p else None

    # --- capabilities (union over pools) -----------------------------------------------

    def _any(self, attr: str) -> bool:
        return any(bool(getattr(p.engine, attr, False)) for p in self._pools)

    @property
    def is_vlm(self) -> bool:
        return self._any("is_vlm")

    @property
    def is_image_gen(self) -> bool:
        return self._any("is_image_gen")

    @property
    def is_asr(self) -> bool:
        return self._any("is_asr")

    @property
    def is_tts(self) -> bool:
        return self._any("is_tts")

    @property
    def current_model(self) -> str:
        """The primary pool's model — what older gateways read as `loaded_model`."""
        return self._pools[0].engine.current_model

    @property
    def loaded_models(self) -> list[str]:
        """Every model resident right now (paused pools still report their model)."""
        out: list[str] = []
        for p in self._pools:
            for m in getattr(p.engine, "loaded_models", None) or [p.engine.current_model]:
                if m and m not in out:
                    out.append(m)
        return out

    # --- lifecycle (fan out to every pool) -------------------------------------------

    async def start(self, model_path: str) -> None:
        """Start every pool on its OWN configured model.

        The `model_path` argument exists for interface compatibility with the single
        engine; a multi-pool worker's models come from each pool's configuration, so it
        is used only for the primary pool when that pool was built without a model.
        """
        for i, p in enumerate(self._pools):
            m = p.engine.current_model or (model_path if i == 0 else "")
            await p.engine.start(m)

    async def start_paused(self, model_path: str) -> None:
        for i, p in enumerate(self._pools):
            m = p.engine.current_model or (model_path if i == 0 else "")
            await p.engine.start_paused(m)

    def request_pause(self) -> None:
        self._pause_requested = True
        for p in self._pools:
            fn = getattr(p.engine, "request_pause", None)
            if fn:
                fn()

    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        """Yield every pool. Concurrent: a slow pool must not delay the others' release.

        Exceptions are collected, not short-circuited — one pool failing to pause must
        not leave the others running on cards mining is about to take.
        """
        self._pause_requested = True
        logger.info("pausing all pools (%d, urgency=%s)", len(self._pools), urgency.name)
        results = await asyncio.gather(*[p.engine.pause(urgency) for p in self._pools],
                                       return_exceptions=True)
        errors = [(p.name, r) for p, r in zip(self._pools, results) if isinstance(r, BaseException)]
        for name, err in errors:
            logger.error("pool %s failed to pause: %r", name, err)
        if errors:
            raise RuntimeError("pause failed for pool(s): " + ", ".join(n for n, _ in errors))

    async def resume(self) -> None:
        """Resume the pools, fastest first, stopping if a yield arrives midway.

        The in-process pools (diffusion, speech) reload in seconds, so they come back
        first, one after another. The vLLM pools follow
        one at a time: each sets CUDA_VISIBLE_DEVICES while spawning its engine core,
        which must not interleave. Measured on a test worker before this ordering: image and
        speech requests waited ~45 s behind two vLLM reloads after every yield, long
        enough for a queued image request to time out.

        A yield between steps stops the resume, so no further pool is loaded onto a
        card mining now needs.
        """
        self._pause_requested = False
        fast = [p for p in self._pools if not _spawns_subprocess(p.engine)]
        slow = [p for p in self._pools if _spawns_subprocess(p.engine)]
        errors: list = []
        # One at a time, not gathered: model loads in threads of one process race on
        # torch's global state — on a test worker a concurrent Kokoro load died with "Cannot copy
        # out of meta tensor" while SDXL and whisper loaded beside it. Each is seconds.
        for p in fast:
            if self._pause_requested:
                logger.warning("pause requested mid-resume — stopping before pool %s", p.name)
                return
            try:
                await p.engine.resume()
            except Exception as e:  # noqa: BLE001 — one pool must not block the others
                logger.error("pool %s failed to resume: %r", p.name, e)
                errors.append((p.name, e))
        for i, p in enumerate(slow):
            if self._pause_requested:
                logger.warning("pause requested mid-resume — stopping before pool %s (%d/%d)",
                               p.name, i + 1, len(slow))
                return
            await p.engine.resume()
        if errors:
            raise RuntimeError("resume failed for pool(s): " + ", ".join(n for n, _ in errors))

    async def stop(self) -> None:
        await asyncio.gather(*[p.engine.stop() for p in self._pools], return_exceptions=True)

    async def switch_model(self, new_model_path: str) -> None:
        """Switch the OWNING pool. A model no pool serves is an error, not a guess."""
        p = self.owner(new_model_path)
        if p is None:
            raise ValueError(f"no pool on this worker serves model {new_model_path!r}")
        await p.engine.switch_model(new_model_path)

    # --- state ----------------------------------------------------------------------

    def is_available(self) -> bool:
        return any(p.engine.is_available() for p in self._pools)

    def status(self) -> EngineStatus:
        """Aggregate state. PAUSED only when EVERY pool is paused (see module docstring)."""
        statuses = [p.engine.status() for p in self._pools]
        states = [s.state for s in statuses]
        if all(s == EngineState.PAUSED for s in states):
            agg = EngineState.PAUSED
        elif all(s == EngineState.STOPPED for s in states):
            agg = EngineState.STOPPED
        elif all(s == EngineState.DEAD for s in states):
            agg = EngineState.DEAD
        elif any(s == EngineState.UNLOADING for s in states):
            # Some pool is still releasing its cards: not yet safe to mine.
            agg = EngineState.UNLOADING
        elif any(s == EngineState.RUNNING for s in states):
            agg = EngineState.RUNNING
        elif any(s in (EngineState.LOADING, EngineState.STARTING) for s in states):
            agg = EngineState.LOADING
        else:
            agg = EngineState.UNLOADING
        return EngineStatus(
            state=agg,
            active_requests=sum(s.active_requests for s in statuses),
            loaded_model=self.current_model,
        )

    def detailed_status(self) -> dict[str, Any]:
        pools = []
        engine_count = 0
        engines: list[dict] = []
        for p in self._pools:
            s = p.engine.status()
            det = p.engine.detailed_status() if hasattr(p.engine, "detailed_status") else {}
            engine_count += int(det.get("engine_count", 1) or 1)
            engines.extend(det.get("engines", []))
            pools.append({
                "name": p.name,
                "engine_state": s.state.value,
                "active_requests": s.active_requests,
                "models": getattr(p.engine, "loaded_models", None) or [p.engine.current_model],
                "devices": list(p.devices),
                "features": _engine_features(p.engine),
                # Per-model request limits, so a gateway with several workers for one
                # model sends a request only where it fits (see api_server /health).
                "limits": _engine_limits(p.engine),
                # Models this pool may switch to: while it switches, they are the ones
                # unavailable here — the worker's other pools keep serving.
                "supported_models": list(p.supported_models),
            })
        return {"mode": "multi_pool", "engine_count": engine_count, "engines": engines, "pools": pools}

    # --- generation: route to the owner ------------------------------------------------

    async def generate(self, prompt: str, **kwargs):
        e = self._route(kwargs.get("model"))
        kwargs.pop("model", None)
        return await e.generate(prompt, **kwargs)

    async def generate_stream(self, prompt: str, **kwargs):
        e = self._route(kwargs.get("model"))
        kwargs.pop("model", None)
        async for chunk in e.generate_stream(prompt, **kwargs):
            yield chunk

    def _route(self, model: str | None) -> InferenceEngine:
        e = self.engine_for(model)
        if e is None:
            raise ValueError(f"no pool on this worker serves model {model!r}")
        return e


def _spawns_subprocess(engine) -> bool:
    """Whether resuming this pool spawns a vLLM engine core (process-wide CUDA env)."""
    from .engine import VLLMEngine
    return any(isinstance(e, VLLMEngine) for e in getattr(engine, "_engines", [engine]))


def _engine_limits(eng) -> dict:
    out = {}
    if getattr(eng, "is_vlm", False):
        out["max_images_per_prompt"] = int(getattr(eng, "max_images_per_prompt", 0) or 0)
    ctx = int(getattr(eng, "context_limit", 0) or 0)
    if ctx:
        out["context_limit"] = ctx
    return out


def _engine_features(engine) -> list[str]:
    """Capability flags one engine advertises (the /health `features` vocabulary)."""
    out = []
    if getattr(engine, "is_vlm", False):
        out.append("vision")
    if getattr(engine, "is_image_gen", False):
        out.append("image_gen")
    if getattr(engine, "is_asr", False):
        out.append("asr")
    if getattr(engine, "is_tts", False):
        out.append("tts")
    return out

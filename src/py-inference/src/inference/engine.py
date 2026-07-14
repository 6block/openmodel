"""Inference engine abstraction with vLLM and Mock implementations."""

import asyncio
import enum
import gc
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Max seconds to wait for the NEXT token from the engine core before treating a request as
# stalled (dead/wedged core) and failing it fast with a retryable error — instead of
# hanging until the client's own timeout. A live engine streams tokens continuously, so a
# gap this long means the core is gone. Soak finding: a SIGKILL'd EngineCore left in-flight
# requests hanging ~120s until the client timed out, giving the gateway no chance to retry
# (0.009% of requests, only under a hard crash). Tunable via env for slow-first-token setups.
_GEN_STALL_TIMEOUT = float(os.environ.get("GEN_STALL_TIMEOUT_SEC", "60"))


class YieldUrgency(enum.IntEnum):
    NORMAL = 0      # Drain all in-flight requests (up to 30s), then pause
    IMMEDIATE = 1   # Abort in-flight requests and stop now


class EngineState(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNLOADING = "unloading"   # Draining requests + destroying engine
    PAUSED = "paused"         # Engine destroyed, VRAM released
    LOADING = "loading"       # Recreating engine + loading model
    STOPPING = "stopping"
    DEAD = "dead"             # Engine core crashed (EngineDeadError) — excluded
                              # from routing until automatic recovery reloads it


@dataclass
class GenerateResult:
    """Result from engine.generate() including token usage and finish reason."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"  # "stop" or "length"
    cached_tokens: int = 0       # Prompt tokens served from vLLM's prefix cache


@dataclass
class StreamChunk:
    """A single streaming chunk yielded during generation."""
    text_delta: str              # New text since last chunk
    finish_reason: str | None = None  # Set only on the last chunk
    prompt_tokens: int = 0       # Set only on the last chunk
    completion_tokens: int = 0   # Set only on the last chunk
    cached_tokens: int = 0       # Set only on the last chunk (prefix-cache hits)


def _extract_cached_tokens(request_output, prompt_tokens: int) -> int:
    """Prompt tokens served from vLLM's automatic prefix cache for this request.

    vLLM exposes this as RequestOutput.num_cached_tokens (0.17.x). Returns 0 if
    the attribute is absent (older vLLM) or invalid, and never reports more than
    the prompt length. Used downstream to bill cache hits at the cache-read rate.
    """
    n = getattr(request_output, "num_cached_tokens", None)
    if n is None:
        return 0
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 0
    if n < 0:
        return 0
    if prompt_tokens and n > prompt_tokens:
        return prompt_tokens
    return n


@dataclass
class EngineStatus:
    state: EngineState
    active_requests: int = 0
    gpu_utilization_pct: float = 0.0
    loaded_model: str = ""
    gpu_id: int = -1  # -1 = all GPUs (tensor parallel)


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    @abstractmethod
    async def start(self, model_path: str) -> None:
        """Load model and start serving."""

    @abstractmethod
    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        """Pause inference. Behavior depends on urgency level."""

    @abstractmethod
    async def resume(self) -> None:
        """Resume inference after pause."""

    @abstractmethod
    async def stop(self) -> None:
        """Full shutdown, unload model."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the engine can accept new requests."""

    @abstractmethod
    def status(self) -> EngineStatus:
        """Return current engine status."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt."""

    async def switch_model(self, new_model_path: str) -> None:
        """Switch to a different model. Default: pause, update path, resume."""
        raise NotImplementedError("subclass must implement switch_model")

    @property
    def current_model(self) -> str:
        """Return the currently loaded model path."""
        return ""


class VLLMEngine(InferenceEngine):
    """Production engine wrapping vLLM's AsyncLLMEngine.

    On pause: destroys the vLLM engine and releases all GPU VRAM.
    On resume: recreates the engine and reloads the model.
    This ensures mining (WindowPoSt) can use the full GPU memory.
    """

    def __init__(self, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 4096,
                 tensor_parallel_size: int = 1,
                 device_ids: list[int] | None = None,
                 enforce_eager: bool = False):
        self._state = EngineState.STOPPED
        self._engine = None
        self._model = ""
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._tensor_parallel_size = tensor_parallel_size
        self._device_ids = device_ids
        self._enforce_eager = enforce_eager
        self._active_requests = 0
        self._inflight_request_ids: set[str] = set()

        # Transition lock prevents concurrent pause/resume/stop races.
        self._transition_lock = asyncio.Lock()
        # Pending flags handle pause-during-load and resume-during-unload.
        self._pending_pause: YieldUrgency | None = None
        self._pending_resume: bool = False

        # PIDs of THIS engine's vLLM core subprocesses, recorded as the
        # active_children() delta around _create_engine. _destroy_engine must
        # only ever kill these — in multi_instance mode all engines share one
        # parent process, so scanning active_children() at destroy time (the
        # old behavior) collected EVERY engine's core and SIGKILLed siblings.
        # That is exactly what crashed healthy engines during a per-engine
        # model switch (soak finding: EngineDeadError storm after switching).
        self._child_pids: set[int] = set()
        # False while a model switch drains this engine: routing skips it but
        # in-flight requests run to completion instead of being aborted.
        self._accepting: bool = True
        # Guards against scheduling more than one dead-engine recovery task.
        self._recovering: bool = False

    # Serializes CUDA_VISIBLE_DEVICES set + subprocess spawn + child-pid delta
    # across ALL engines in this process (class-level), so concurrent creates
    # can neither race on the env var nor mis-attribute each other's children.
    _create_lock: asyncio.Lock | None = None

    @classmethod
    def _creation_lock(cls) -> asyncio.Lock:
        if cls._create_lock is None:
            cls._create_lock = asyncio.Lock()
        return cls._create_lock

    async def _create_engine(self) -> None:
        """Create and initialize the vLLM AsyncLLMEngine.

        Runs under the class-wide creation lock: setting the process-level
        CUDA_VISIBLE_DEVICES, spawning the core subprocess, and attributing the
        new child PIDs to THIS engine must be atomic across all engines.
        """
        import multiprocessing

        async with VLLMEngine._creation_lock():
            # Snapshot pre-existing children so the post-create delta contains
            # only the subprocesses spawned by THIS engine.
            try:
                pre_existing = {c.pid for c in multiprocessing.active_children()}
            except Exception:
                pre_existing = set()

            # Set CUDA_VISIBLE_DEVICES every time (not just on start),
            # so that resume also targets the correct GPU.
            if self._device_ids is not None:
                cuda_devices = ",".join(str(d) for d in self._device_ids)
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
                logger.info("set CUDA_VISIBLE_DEVICES=%s", cuda_devices)
            await self._create_engine_inner()

            try:
                self._child_pids = {c.pid for c in multiprocessing.active_children()} - pre_existing
                logger.info("engine subprocesses for devices %s: %s",
                            self._device_ids, self._child_pids)
            except Exception:
                self._child_pids = set()

    async def _create_engine_inner(self) -> None:
        """The actual vLLM engine construction (called under the creation lock)."""
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        # Auto-adapt max_model_len: if model's max_position_embeddings is smaller
        # than configured max_model_len, use the model's limit instead.
        effective_max_model_len = self._max_model_len
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(self._model, trust_remote_code=True)
            model_max = getattr(model_config, 'max_position_embeddings', None)
            if model_max and model_max < effective_max_model_len:
                logger.info("adapting max_model_len: %d -> %d (model limit)",
                            effective_max_model_len, model_max)
                effective_max_model_len = model_max
        except Exception as e:
            logger.debug("could not read model config for max_model_len: %s", e)

        args = AsyncEngineArgs(
            model=self._model,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=effective_max_model_len,
            tensor_parallel_size=self._tensor_parallel_size,
            trust_remote_code=True,
            enforce_eager=self._enforce_eager,
            enable_prefix_caching=True,  # report num_cached_tokens for cache-read billing (V1 default; set explicitly to be deterministic)
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)

    async def _destroy_engine(self) -> None:
        """Destroy the vLLM engine and release all GPU memory.

        vLLM v1 runs the engine core in a subprocess (EngineCore_DP0).
        shutdown() sends a signal but may not kill it. We find and
        forcefully terminate the subprocess if it lingers.
        """
        if self._engine is not None:
            # Abort all in-flight requests
            for req_id in list(self._inflight_request_ids):
                try:
                    await self._engine.abort(req_id)
                except Exception:
                    pass  # Best-effort abort
            self._inflight_request_ids.clear()

            # Kill ONLY this engine's own core subprocesses (recorded at create
            # time). Never scan multiprocessing.active_children() here: in
            # multi_instance mode all engines share one parent process, so the
            # old scan collected every sibling engine's core and SIGKILLed
            # healthy engines during a per-engine model switch (soak finding:
            # EngineDeadError storm, ~1/8 requests surviving).
            child_pids = set(self._child_pids)
            logger.info("this engine's vLLM subprocesses before shutdown: %s", child_pids)

            # Shutdown the engine
            if hasattr(self._engine, 'shutdown'):
                try:
                    self._engine.shutdown()
                except Exception as e:
                    logger.warning("engine shutdown error (non-fatal): %s", e)
            self._engine = None

            # Wait for subprocess to exit gracefully, then kill stragglers.
            # 5s gives the CUDA driver more time to release resources
            # (2s was too short, especially with older NVIDIA drivers like 560.x).
            if child_pids:
                await asyncio.sleep(5)
                import signal
                for pid in child_pids:
                    try:
                        os.kill(pid, 0)  # Check if still alive
                        logger.warning("vLLM subprocess %d still alive after shutdown, sending SIGKILL", pid)
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Already exited
                    except Exception as e:
                        logger.debug("could not kill pid %d: %s", pid, e)
                # Wait for killed processes to release resources
                await asyncio.sleep(1)
            self._child_pids.clear()

        # Force CUDA memory release
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()  # Second pass after gc
                logger.info("CUDA cache cleared, GPU VRAM released")
        except ImportError:
            pass

    async def _drain_requests(self, timeout: float = 30.0) -> None:
        """Wait for in-flight requests to complete, with timeout."""
        if self._active_requests == 0:
            return

        logger.info("draining %d in-flight requests (timeout=%.0fs)",
                     self._active_requests, timeout)
        deadline = asyncio.get_event_loop().time() + timeout
        while self._active_requests > 0:
            if asyncio.get_event_loop().time() > deadline:
                logger.warning("drain timeout, %d requests still in-flight — aborting",
                               self._active_requests)
                break
            await asyncio.sleep(0.1)

    def _query_gpu_free_ratio(self, gpu_id: int) -> float | None:
        """Query free VRAM ratio for a specific physical GPU via nvidia-smi.

        Uses nvidia-smi instead of torch.cuda.mem_get_info() because the
        latter depends on CUDA context which is bound to whichever GPU was
        first visible — unreliable in multi-instance mode.
        """
        try:
            import subprocess as sp
            result = sp.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits", f"--id={gpu_id}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            parts = result.stdout.strip().split(",")
            free_mib = float(parts[0].strip())
            total_mib = float(parts[1].strip())
            return free_mib / total_mib if total_mib > 0 else None
        except Exception:
            return None

    async def _wait_for_vram_release(self, timeout: float = 20.0) -> None:
        """Wait until GPU VRAM is actually released.

        vLLM v1 runs the engine core in a subprocess (EngineCore_DP0).
        After shutdown(), the subprocess may still hold CUDA memory for a few
        seconds. We poll nvidia-smi until enough memory is free.
        """
        gpu_id = self._device_ids[0] if self._device_ids else 0
        needed_ratio = self._gpu_memory_utilization
        deadline = time.monotonic() + timeout

        # A transient nvidia-smi failure must NOT cause us to immediately proceed
        # and recreate the engine while the GPU is still occupied (old subprocess
        # not yet released, or mining still winding down) — that risks OOM or
        # landing on a mining GPU. Keep polling on failures. Only if nvidia-smi
        # NEVER works (e.g. not installed in dev/mock) do we give up and proceed.
        smi_ever_worked = False
        consecutive_failures = 0
        while time.monotonic() < deadline:
            free_ratio = self._query_gpu_free_ratio(gpu_id)
            if free_ratio is None:
                consecutive_failures += 1
                if not smi_ever_worked and consecutive_failures >= 3:
                    logger.warning("nvidia-smi unavailable for GPU %d — cannot verify VRAM, proceeding", gpu_id)
                    return
                logger.debug("nvidia-smi query failed for GPU %d (%d), retrying",
                             gpu_id, consecutive_failures)
                await asyncio.sleep(0.5)
                continue

            smi_ever_worked = True
            consecutive_failures = 0
            if free_ratio >= needed_ratio:
                logger.info("GPU %d VRAM released: %.1f%% free (need %.0f%%)",
                            gpu_id, free_ratio * 100, needed_ratio * 100)
                return
            logger.debug("GPU %d waiting for VRAM: %.1f%% free, need %.0f%%",
                         gpu_id, free_ratio * 100, needed_ratio * 100)
            await asyncio.sleep(0.5)

        free_ratio = self._query_gpu_free_ratio(gpu_id) or 0.0
        logger.warning("GPU %d VRAM release timeout after %.0fs — %.1f%% free, proceeding anyway",
                       gpu_id, timeout, free_ratio * 100)

    async def _do_reload(self) -> None:
        """Reload the model into GPU. State must already be LOADING.

        Retries up to 3 times with a 5-second delay between attempts.
        vLLM's EngineCore subprocess can fail to initialize if the previous
        process's GPU resources haven't been fully released by the OS/driver.
        A short delay before retry usually resolves this.
        """
        max_attempts = 3
        retry_delay = 5.0
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                t0 = time.monotonic()
                logger.info("reloading model %s into GPU... (attempt %d/%d)",
                            self._model, attempt, max_attempts)
                await self._wait_for_vram_release()
                await self._create_engine()
                elapsed = time.monotonic() - t0
                logger.info("model reloaded in %.1fs", elapsed)

                async with self._transition_lock:
                    if self._pending_pause is not None:
                        # Pause was requested while we were loading — unload again
                        urgency = self._pending_pause
                        self._pending_pause = None
                        logger.info("pending pause detected after reload, unloading again (urgency=%s)",
                                     urgency.name)
                        await self._destroy_engine()
                        self._state = EngineState.PAUSED
                    else:
                        self._state = EngineState.RUNNING
                        logger.info("vLLM engine running — inference available")
                return  # Success

            except Exception as e:
                last_error = e
                if attempt < max_attempts:
                    logger.warning("reload attempt %d/%d failed: %s — retrying in %.0fs",
                                   attempt, max_attempts, e, retry_delay)
                    # Clean up failed engine state before retry
                    self._engine = None
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                    except Exception:
                        pass
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("all %d reload attempts failed: %s", max_attempts, e)

        # All attempts exhausted — the CUDA context is likely corrupted.
        # Only a full process restart can recover. Schedule self-restart
        # and let Docker's restart policy bring us back with a clean state.
        logger.critical(
            "all %d reload attempts failed — CUDA context likely corrupted. "
            "Restarting process in 3s to recover...", max_attempts
        )
        async with self._transition_lock:
            self._pending_pause = None
            self._state = EngineState.PAUSED

        # Give logs time to flush, then exit. Docker restart policy will relaunch.
        await asyncio.sleep(3)
        os._exit(1)

    async def start(self, model_path: str) -> None:
        async with self._transition_lock:
            self._state = EngineState.STARTING
            self._accepting = True  # clear any leftover drain gate

        logger.info("starting vLLM engine: model=%s, tp=%d, devices=%s",
                     model_path, self._tensor_parallel_size, self._device_ids)

        try:
            self._model = model_path
            await self._create_engine()
        except Exception as e:
            async with self._transition_lock:
                self._state = EngineState.STOPPED
                # Drop any pause queued during the failed start — nothing loaded.
                self._pending_pause = None
            logger.error("failed to start vLLM: %s", e)
            raise

        # A yield may have arrived while we were loading (state == STARTING).
        # Honor it now instead of coming up RUNNING and holding the GPU during
        # mining (fixes pause-dropped-during-STARTING).
        async with self._transition_lock:
            if self._pending_pause is not None:
                urgency = self._pending_pause
                self._pending_pause = None
                logger.info("pending pause detected after start, unloading immediately (urgency=%s)",
                             urgency.name)
                await self._destroy_engine()
                self._state = EngineState.PAUSED
                logger.info("vLLM engine paused right after start — GPU VRAM released")
            else:
                self._state = EngineState.RUNNING
                logger.info("vLLM engine started: model=%s", model_path)

    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        async with self._transition_lock:
            if self._state in (EngineState.LOADING, EngineState.STARTING):
                # Model is being loaded (resume) or first-started — flag it; the
                # loader (start()/_do_reload()) unloads after the load completes.
                self._pending_pause = urgency
                logger.info("pause requested during %s (urgency=%s), will unload after load",
                             self._state.value, urgency.name)
                return

            if self._state != EngineState.RUNNING:
                return

            logger.info("pausing vLLM engine (urgency=%s, inflight=%d) — will release GPU VRAM",
                         urgency.name, self._active_requests)
            if urgency == YieldUrgency.NORMAL:
                # Soft yield: refuse NEW requests but stay RUNNING through the
                # drain so in-flight generations truly run to completion. The
                # old code flipped to UNLOADING first, which made every
                # in-flight request abort itself on its next token — the
                # documented "NORMAL = drain 30s" never actually drained.
                self._accepting = False
            else:
                # Immediate yield (WinningPoSt): abort in-flight right now.
                self._state = EngineState.UNLOADING

        if urgency == YieldUrgency.NORMAL:
            await self._drain_requests(timeout=30)
            async with self._transition_lock:
                if self._state != EngineState.RUNNING:
                    # An immediate pause / stop took over mid-drain — it owns
                    # the rest of the teardown.
                    return
                self._state = EngineState.UNLOADING

        # Destroy engine and release VRAM
        async with self._transition_lock:
            await self._destroy_engine()
            # Availability stays gated by state until resume; re-arm accepting
            # so the engine serves again once it is RUNNING.
            self._accepting = True

            if self._pending_resume:
                # Resume was requested while we were unloading — reload immediately
                self._pending_resume = False
                logger.info("pending resume detected, skipping PAUSED → loading immediately")
                self._state = EngineState.LOADING
            else:
                self._state = EngineState.PAUSED
                logger.info("vLLM engine paused — GPU VRAM released")
                return

        # Handle pending resume
        await self._do_reload()

    async def resume(self) -> None:
        async with self._transition_lock:
            if self._state == EngineState.UNLOADING:
                # Engine is being unloaded — flag it, will reload after unload completes
                self._pending_resume = True
                logger.info("resume requested during UNLOADING, will reload after unload")
                return

            if self._state == EngineState.DEAD:
                # Core crashed and a recovery task is already reloading it. Don't
                # silently no-op (the listener would treat resume as done and
                # never retry) — the recovery path brings the engine back to
                # RUNNING on its own, so just note it and let it finish.
                logger.warning("resume requested while engine DEAD — recovery already in progress, will come up RUNNING")
                return

            if self._state != EngineState.PAUSED:
                return

            logger.info("resuming vLLM engine — reloading model %s", self._model)
            self._state = EngineState.LOADING

        await self._do_reload()

    async def stop(self) -> None:
        logger.info("stopping vLLM engine")
        async with self._transition_lock:
            self._state = EngineState.STOPPING
            await self._destroy_engine()
            self._state = EngineState.STOPPED

    async def switch_model(self, new_model_path: str) -> None:
        """Switch to a different model by destroying and recreating the engine.

        Unlike a mining yield, a model switch is not urgent: stop ACCEPTING new
        requests on this engine (the pool routes around it) but let in-flight
        requests run to completion BEFORE tearing the engine down. The old
        behavior paused immediately, which aborted in-flight work mid-token
        under load (soak finding: switching while serving crashed cores)."""
        if self._model == new_model_path and self._state == EngineState.RUNNING:
            return  # Already loaded

        old_model = self._model
        logger.info("switching model: %s -> %s", old_model, new_model_path)

        self._accepting = False
        try:
            # Drain while still RUNNING: the mid-generation state check stays
            # green, so requests genuinely complete instead of self-aborting.
            await self._drain_requests(timeout=30)

            # Pause to destroy current engine and release VRAM
            await self.pause(YieldUrgency.NORMAL)

            # Update model path
            self._model = new_model_path

            # Resume to create new engine with new model
            try:
                await self.resume()
                logger.info("model switch complete: %s -> %s", old_model, new_model_path)
            except Exception as e:
                logger.error("model switch failed, rolling back to %s: %s", old_model, e)
                self._model = old_model
                try:
                    await self.resume()
                    logger.info("rollback successful, restored: %s", old_model)
                except Exception as rollback_err:
                    logger.error("rollback also failed: %s", rollback_err)
                raise
        finally:
            self._accepting = True

    @property
    def current_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        # _accepting goes False while a model switch drains this engine:
        # no NEW requests are routed here, but in-flight ones finish normally.
        return self._state == EngineState.RUNNING and self._accepting

    @staticmethod
    def _is_engine_dead_error(e: BaseException) -> bool:
        """Detect vLLM's EngineDeadError without a hard import dependency
        (the class lives in vllm.v1.engine.exceptions; mock/test envs lack it)."""
        return "enginedead" in type(e).__name__.lower()

    def _mark_dead(self, err: BaseException) -> None:
        """The engine core subprocess crashed. Take this engine out of routing
        and schedule one automatic recovery (destroy remnants + reload model).

        Without this, /health kept reporting the engine as running and the
        pool kept routing requests into the dead core — every one failed with
        a 500 until the container was manually restarted (soak finding)."""
        logger.critical(
            "vLLM engine core died (devices=%s, model=%s): %s — "
            "marking engine DEAD and scheduling automatic recovery",
            self._device_ids, self._model, err)
        self._state = EngineState.DEAD
        if not self._recovering:
            self._recovering = True
            asyncio.get_running_loop().create_task(self._recover_dead_engine())

    async def _recover_dead_engine(self) -> None:
        """Background recovery: reap the dead core, then reload the model.
        _do_reload retries 3× and falls back to process exit (container
        restart) if the CUDA context is unrecoverable — same policy as resume."""
        try:
            logger.info("recovering dead engine (devices=%s, model=%s)",
                        self._device_ids, self._model)
            async with self._transition_lock:
                if self._state != EngineState.DEAD:
                    return  # someone else already transitioned us
                await self._destroy_engine()
                self._state = EngineState.LOADING
            await self._do_reload()
            logger.info("dead engine recovered (devices=%s)", self._device_ids)
        except Exception as e:
            logger.error("dead-engine recovery failed (devices=%s): %s",
                         self._device_ids, e)
        finally:
            self._recovering = False

    def status(self) -> EngineStatus:
        gpu_id = self._device_ids[0] if self._device_ids and len(self._device_ids) == 1 else -1
        return EngineStatus(
            state=self._state,
            active_requests=self._active_requests,
            loaded_model=self._model,
            gpu_id=gpu_id,
        )

    async def _iter_outputs(self, results_generator, request_id: str):
        """Iterate a vLLM result generator, bounding each await so a wedged or dead engine
        core can't hang the request until the client's own timeout. If no token arrives
        within _GEN_STALL_TIMEOUT, abort the request and raise a RuntimeError, which the API
        layer maps to a retryable 503 so the gateway fails over to another worker (soak
        finding: a SIGKILL'd EngineCore left in-flight requests hanging instead of fast-
        failing). A genuine EngineDeadError still propagates and is handled by the callers."""
        aiter = results_generator.__aiter__()
        while True:
            try:
                out = await asyncio.wait_for(aiter.__anext__(), timeout=_GEN_STALL_TIMEOUT)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                try:
                    await self._engine.abort(request_id)
                except Exception:
                    pass
                logger.warning("request %s stalled — no output for %.0fs, failing fast (dead/wedged core)",
                               request_id, _GEN_STALL_TIMEOUT)
                raise RuntimeError(
                    "engine stalled — no output; request failed, retry on another worker"
                )
            yield out

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Engine not available — GPU yielded to mining")

        import uuid as _uuid
        from vllm import SamplingParams

        request_id = f"req-{_uuid.uuid4().hex[:12]}"

        sp_kwargs = dict(
            max_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
        )
        stop = kwargs.get("stop")
        if stop is not None:
            if isinstance(stop, str):
                stop = [stop]
            sp_kwargs["stop"] = stop

        sampling_params = SamplingParams(**sp_kwargs)

        self._active_requests += 1
        self._inflight_request_ids.add(request_id)
        try:
            results_generator = self._engine.generate(prompt, sampling_params, request_id)
            final_output = None
            async for request_output in self._iter_outputs(results_generator, request_id):
                # Check if we got paused mid-generation
                if self._state != EngineState.RUNNING:
                    if self._engine is not None:
                        try:
                            await self._engine.abort(request_id)
                        except Exception:
                            pass
                    raise RuntimeError("Engine paused during generation — request aborted")
                final_output = request_output

            if final_output is None:
                raise RuntimeError("vLLM returned no output")

            # Collect output text, token counts, and finish reason from vLLM's RequestOutput
            output = final_output.outputs[0]
            output_text = output.text
            prompt_tokens = len(final_output.prompt_token_ids) if hasattr(final_output, 'prompt_token_ids') and final_output.prompt_token_ids else 0
            completion_tokens = len(output.token_ids) if hasattr(output, 'token_ids') and output.token_ids else 0
            cached_tokens = _extract_cached_tokens(final_output, prompt_tokens)

            # Map vLLM's finish reason to OpenAI format
            finish_reason = "stop"
            if hasattr(output, 'finish_reason') and output.finish_reason:
                vllm_reason = str(output.finish_reason).lower()
                if "length" in vllm_reason:
                    finish_reason = "length"

            return GenerateResult(
                text=output_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                cached_tokens=cached_tokens,
            )
        except Exception as e:
            if self._is_engine_dead_error(e):
                # Core subprocess crashed: take the engine out of routing and
                # surface a RuntimeError, which the API layer maps to 503
                # (retryable) instead of the unhandled-500 this used to be.
                self._mark_dead(e)
                raise RuntimeError(
                    "engine core died — request failed, retry on another worker"
                ) from e
            raise
        finally:
            self._active_requests -= 1
            self._inflight_request_ids.discard(request_id)

    async def generate_stream(self, prompt: str, **kwargs):
        """Generate a completion, yielding StreamChunk objects as tokens arrive.

        Same setup as generate(), but yields incremental text deltas instead
        of waiting for the full output.
        """
        if not self.is_available():
            raise RuntimeError("Engine not available — GPU yielded to mining")

        import uuid as _uuid
        from vllm import SamplingParams

        request_id = f"req-{_uuid.uuid4().hex[:12]}"

        sp_kwargs = dict(
            max_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
        )
        stop = kwargs.get("stop")
        if stop is not None:
            if isinstance(stop, str):
                stop = [stop]
            sp_kwargs["stop"] = stop

        sampling_params = SamplingParams(**sp_kwargs)

        self._active_requests += 1
        self._inflight_request_ids.add(request_id)
        prev_text = ""
        try:
            results_generator = self._engine.generate(prompt, sampling_params, request_id)
            final_output = None
            async for request_output in self._iter_outputs(results_generator, request_id):
                if self._state != EngineState.RUNNING:
                    if self._engine is not None:
                        try:
                            await self._engine.abort(request_id)
                        except Exception:
                            pass
                    raise RuntimeError("Engine paused during generation — request aborted")

                final_output = request_output
                current_text = request_output.outputs[0].text
                delta = current_text[len(prev_text):]
                if delta:
                    yield StreamChunk(text_delta=delta)
                prev_text = current_text

            if final_output is None:
                return

            # Final chunk with usage and finish_reason
            output = final_output.outputs[0]
            prompt_tokens = len(final_output.prompt_token_ids) if hasattr(final_output, 'prompt_token_ids') and final_output.prompt_token_ids else 0
            completion_tokens = len(output.token_ids) if hasattr(output, 'token_ids') and output.token_ids else 0
            cached_tokens = _extract_cached_tokens(final_output, prompt_tokens)

            finish_reason = "stop"
            if hasattr(output, 'finish_reason') and output.finish_reason:
                vllm_reason = str(output.finish_reason).lower()
                if "length" in vllm_reason:
                    finish_reason = "length"

            yield StreamChunk(
                text_delta="",
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
        except Exception as e:
            if self._is_engine_dead_error(e):
                self._mark_dead(e)
                raise RuntimeError(
                    "engine core died — request failed, retry on another worker"
                ) from e
            raise
        finally:
            self._active_requests -= 1
            self._inflight_request_ids.discard(request_id)


class MockEngine(InferenceEngine):
    """Mock engine for local development without GPU.

    Simulates model unload/reload delays to match VLLMEngine behavior.
    """

    def __init__(self, latency_sec: float = 0.5, gpu_id: int = 0):
        self._state = EngineState.STOPPED
        self._model = ""
        self._active_requests = 0
        self._latency = latency_sec
        self._gpu_id = gpu_id

    async def start(self, model_path: str) -> None:
        self._state = EngineState.STARTING
        logger.info("mock engine starting", extra={"model": model_path, "gpu_id": self._gpu_id})
        await asyncio.sleep(0.1)  # Simulate startup
        self._model = model_path
        self._state = EngineState.RUNNING
        logger.info("mock engine started on GPU %d", self._gpu_id)

    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        if self._state != EngineState.RUNNING:
            return

        logger.info("mock engine GPU %d pausing (urgency=%s) — simulating VRAM release",
                     self._gpu_id, urgency.name)

        self._state = EngineState.UNLOADING
        if urgency == YieldUrgency.NORMAL:
            await asyncio.sleep(0.5)  # Simulate drain + unload
        else:
            await asyncio.sleep(0.1)  # Simulate immediate unload

        self._state = EngineState.PAUSED
        logger.info("mock engine GPU %d paused — VRAM released", self._gpu_id)

    async def resume(self) -> None:
        if self._state != EngineState.PAUSED:
            return

        logger.info("mock engine GPU %d resuming — simulating model reload", self._gpu_id)
        self._state = EngineState.LOADING
        await asyncio.sleep(2.0)  # Simulate model reload time
        self._state = EngineState.RUNNING
        logger.info("mock engine GPU %d running", self._gpu_id)

    async def stop(self) -> None:
        logger.info("mock engine GPU %d stopping", self._gpu_id)
        self._state = EngineState.STOPPED
        self._model = ""

    async def switch_model(self, new_model_path: str) -> None:
        if self._model == new_model_path and self._state == EngineState.RUNNING:
            return
        old = self._model
        logger.info("mock engine GPU %d switching model: %s -> %s", self._gpu_id, old, new_model_path)
        await self.pause()
        self._model = new_model_path
        await self.resume()

    @property
    def current_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return self._state == EngineState.RUNNING

    def status(self) -> EngineStatus:
        return EngineStatus(
            state=self._state,
            active_requests=self._active_requests,
            loaded_model=self._model,
            gpu_id=self._gpu_id,
        )

    async def generate(self, prompt: str, **kwargs) -> GenerateResult:
        if not self.is_available():
            raise RuntimeError("Engine not available for inference")

        self._active_requests += 1
        try:
            await asyncio.sleep(self._latency)
            max_tokens = kwargs.get("max_tokens", 50)
            text = (f"[GPU-{self._gpu_id} Mock response to: {prompt[:50]}...] "
                    + "token " * min(max_tokens, 20))
            return GenerateResult(
                text=text,
                prompt_tokens=len(prompt.split()),
                completion_tokens=min(max_tokens, 20),
            )
        finally:
            self._active_requests -= 1

    async def generate_stream(self, prompt: str, **kwargs):
        if not self.is_available():
            raise RuntimeError("Engine not available for inference")

        self._active_requests += 1
        try:
            words = ["Hello ", "from ", "mock ", "engine. "]
            for w in words:
                await asyncio.sleep(0.1)
                yield StreamChunk(text_delta=w)
            yield StreamChunk(
                text_delta="",
                finish_reason="stop",
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(words),
            )
        finally:
            self._active_requests -= 1


def create_engine(engine_type: str, **kwargs) -> InferenceEngine:
    """Factory function to create a single inference engine.

    Args:
        engine_type: "mock" or "vllm"
        device_ids: List of CUDA device IDs this engine should use
        gpu_id: Mock GPU identifier (mock mode only)
        gpu_memory_utilization, max_model_len, tensor_parallel_size: vLLM params
    """
    if engine_type == "mock":
        return MockEngine(
            latency_sec=kwargs.get("latency_sec", 0.5),
            gpu_id=kwargs.get("gpu_id", 0),
        )
    elif engine_type == "vllm":
        return VLLMEngine(
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.85),
            max_model_len=kwargs.get("max_model_len", 4096),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            device_ids=kwargs.get("device_ids"),
            enforce_eager=kwargs.get("enforce_eager", False),
        )
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")

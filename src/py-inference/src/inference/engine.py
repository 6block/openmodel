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


@dataclass
class GenerateResult:
    """Result from engine.generate() including token usage and finish reason."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"  # "stop" or "length"


@dataclass
class StreamChunk:
    """A single streaming chunk yielded during generation."""
    text_delta: str              # New text since last chunk
    finish_reason: str | None = None  # Set only on the last chunk
    prompt_tokens: int = 0       # Set only on the last chunk
    completion_tokens: int = 0   # Set only on the last chunk


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

    async def _create_engine(self) -> None:
        """Create and initialize the vLLM AsyncLLMEngine."""
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        # Set CUDA_VISIBLE_DEVICES every time (not just on start),
        # so that resume also targets the correct GPU.
        if self._device_ids is not None:
            cuda_devices = ",".join(str(d) for d in self._device_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            logger.info("set CUDA_VISIBLE_DEVICES=%s", cuda_devices)

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

            # Collect child PIDs before shutdown (vLLM v1 spawns subprocesses)
            import multiprocessing
            child_pids = set()
            try:
                parent_pid = os.getpid()
                for child in multiprocessing.active_children():
                    child_pids.add(child.pid)
                logger.info("vLLM child processes before shutdown: %s", child_pids)
            except Exception:
                pass

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

        while time.monotonic() < deadline:
            free_ratio = self._query_gpu_free_ratio(gpu_id)
            if free_ratio is None:
                logger.debug("nvidia-smi query failed for GPU %d, skipping wait", gpu_id)
                return
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
        self._state = EngineState.STARTING

        logger.info("starting vLLM engine: model=%s, tp=%d, devices=%s",
                     model_path, self._tensor_parallel_size, self._device_ids)

        try:
            self._model = model_path
            await self._create_engine()
            self._state = EngineState.RUNNING
            logger.info("vLLM engine started: model=%s", model_path)
        except Exception as e:
            self._state = EngineState.STOPPED
            logger.error("failed to start vLLM: %s", e)
            raise

    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        async with self._transition_lock:
            if self._state == EngineState.LOADING:
                # Model is being loaded — flag it, will unload after load completes
                self._pending_pause = urgency
                logger.info("pause requested during LOADING (urgency=%s), will unload after load",
                             urgency.name)
                return

            if self._state != EngineState.RUNNING:
                return

            logger.info("pausing vLLM engine (urgency=%s, inflight=%d) — will release GPU VRAM",
                         urgency.name, self._active_requests)
            self._state = EngineState.UNLOADING

        # Outside lock: drain or skip based on urgency
        if urgency == YieldUrgency.NORMAL:
            await self._drain_requests(timeout=30)

        # Destroy engine and release VRAM
        async with self._transition_lock:
            await self._destroy_engine()

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
        """Switch to a different model by destroying and recreating the engine."""
        if self._model == new_model_path and self._state == EngineState.RUNNING:
            return  # Already loaded

        old_model = self._model
        logger.info("switching model: %s -> %s", old_model, new_model_path)

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

    @property
    def current_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return self._state == EngineState.RUNNING

    def status(self) -> EngineStatus:
        gpu_id = self._device_ids[0] if self._device_ids and len(self._device_ids) == 1 else -1
        return EngineStatus(
            state=self._state,
            active_requests=self._active_requests,
            loaded_model=self._model,
            gpu_id=gpu_id,
        )

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
            async for request_output in results_generator:
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
            )
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
            async for request_output in results_generator:
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
            )
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

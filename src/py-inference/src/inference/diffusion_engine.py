"""Diffusion (image generation) engine — the second engine class alongside VLLMEngine.

WHY A SEPARATE CLASS rather than folding image generation into vLLM: SD/Flux run on
diffusers, a different stack with a different execution model. Keeping them apart means a
fault in one cannot take down the other, and each can be scheduled on its own devices.

WHAT IS SHARED WITH VLLMEngine (and therefore must not be forgotten here — the pool and
the scheduler listener touch all of it):

  * the InferenceEngine ABC's methods, plus `generate_stream`, which is NOT in the ABC
    but is called by EnginePool and the streaming API path;
  * `switch_model` + `current_model`, called on every request that names a model;
  * the `_model` and `_state` attributes, which EnginePool.start_paused() assigns to
    directly — these are the hard implicit contract;
  * `pause()` must leave `status().state == EngineState.PAUSED`, or the scheduler
    listener reports the GPU as still held and retries the yield forever;
  * a `request_pause()` no-op, so the listener's preemption signal is safe;
  * `is_vlm` / `max_images_per_prompt`, which EnginePool reads with getattr defaults.

THE DIFFERENCE THAT MATTERS: granularity of interruption. A token generator can be
aborted at any token, so `IMMEDIATE` really does stop it instantly. A diffusion step loop
cannot: each denoising step is a single forward pass, and abandoning one mid-pass leaves
the CUDA context in an undefined state. So this engine interrupts between steps, not
inside them. The gap is one step's latency, not the whole generation — but it is real, and
it is why `_drain_requests` uses a diffusion-sized timeout here rather than the token
engine's 30 s.
"""

import asyncio
import gc
import io
import logging
import os
import time

from .engine import (
    EngineState,
    EngineStatus,
    GenerateResult,
    InferenceEngine,
    YieldUrgency,
    query_gpu_free_ratio,  # shared nvidia-smi helper (see engine.py)
)
from .torch_globals import guarded_call, guarded_load

logger = logging.getLogger(__name__)

# Seconds to wait for in-flight image generations to finish on a soft yield. A single
# image at 4 steps on a 10 GB card takes on the order of 10-30 s, and a queue of them can
# take minutes — far past the token engine's 30 s. Env-tunable because the right value
# depends on the model and the card.
_DIFFUSION_DRAIN_TIMEOUT = float(os.environ.get("DIFFUSION_DRAIN_TIMEOUT_SEC", "180"))

# How often the step loop checks whether a yield arrived.
_STEP_CHECK_INTERVAL = 1

# How long an IMMEDIATE yield waits for an interrupted render to return before it
# releases the pipeline. The render stops at the next step boundary (one step of SDXL
# at 1024² is ~0.5 s on a 3080) and then decodes; this bounds a pathological case.
_IMMEDIATE_STOP_TIMEOUT = float(os.environ.get("DIFFUSION_IMMEDIATE_STOP_SEC", "20"))

# VAE tile size in pixels. Small enough that a 1024x1024 decode is split (SDXL's own
# default tile is 1024, exactly the render size, so it would never tile); large enough
# that tile seams stay rare. Env-tunable for cards with more headroom.
_VAE_TILE_PX = int(os.environ.get("DIFFUSION_VAE_TILE_PX", "512"))


def _has_fp16_variant(model_dir: str) -> bool:
    """True when the model directory ships the `fp16` weight variant.

    Diffusers names variant weights `<name>.fp16.safetensors`. A directory holding
    ONLY those (the usual download for a 10 GB card — the fp32 set is twice the size)
    cannot be loaded without `variant="fp16"`: from_pretrained looks for the default
    file name, finds nothing, and fails. Detected from the files rather than
    configured, so the same engine loads either layout.
    """
    if not os.path.isdir(model_dir):
        return False
    for root, _dirs, files in os.walk(model_dir):
        for name in files:
            if name.endswith(".fp16.safetensors") or name.endswith(".fp16.bin"):
                return True
    return False


def _precision_kwargs(model_dir: str) -> dict:
    """from_pretrained arguments for the weights present in `model_dir`.

    An fp16 variant is loaded as fp16 (variant + torch_dtype): the variant files are
    stored in half precision, and upcasting them to fp32 on load would double the
    VRAM for no gain in quality. Without a variant the pipeline loads as shipped.
    """
    if not _has_fp16_variant(model_dir):
        return {}
    import torch  # only reached on a worker that has fp16 weights, i.e. a GPU box
    return {"variant": "fp16", "torch_dtype": torch.float16}


def _accepts_step_callback(pipe) -> bool:
    """Whether the pipeline's __call__ takes `callback_on_step_end` by name.

    Checked by name (not via **kwargs) so a pipeline that would silently swallow the
    argument is treated as not supporting it.
    """
    try:
        import inspect
        return "callback_on_step_end" in inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False


def _shrink_vae_tiles(vae, tile_px: int | None = None) -> None:
    """Lower the VAE's tiling threshold so standard render sizes are actually tiled.

    Only ever shrinks: a VAE whose tiles are already smaller is left alone. The latent
    tile is derived from the VAE's own pixel/latent ratio rather than assumed to be 8,
    so a VAE with a different downsampling factor stays consistent.
    """
    tile_px = tile_px or _VAE_TILE_PX
    sample = getattr(vae, "tile_sample_min_size", None)
    latent = getattr(vae, "tile_latent_min_size", None)
    if not isinstance(sample, int) or not isinstance(latent, int) or latent <= 0:
        return
    if sample <= tile_px:
        return
    factor = sample // latent
    if factor <= 0:
        return
    vae.tile_sample_min_size = tile_px
    vae.tile_latent_min_size = tile_px // factor


class DiffusionEngine(InferenceEngine):
    """Image generation engine backed by diffusers (SD / Flux).

    Lifecycle mirrors VLLMEngine: pause destroys the pipeline and frees VRAM, resume
    rebuilds it. The pipeline is in-process (no engine-core subprocess), so unlike the
    vLLM engine there are no child PIDs to reap — releasing VRAM is `del pipeline` plus
    an empty_cache().
    """

    def _bind_device(self) -> None:  # pragma: no cover - needs a GPU
        """Make this engine's card the current CUDA device of the calling thread.

        Renders run on asyncio's thread pool, whose threads start on card 0. The
        pipeline itself is placed explicitly, but any allocation a library makes on
        the *current* device would land on card 0 — full of a vLLM engine on a
        multi-pool worker. The speech engines crashed exactly this way (CTranslate2);
        the same guard is applied here so no engine relies on the thread default."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(self._device_ids[0])
        except ImportError:
            pass

    def _release_cuda_cache(self) -> None:  # pragma: no cover - needs a GPU
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                with torch.cuda.device(self._device_ids[0]):
                    torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _on_device(self, fn, *args, **kwargs):
        self._bind_device()
        return guarded_call(fn, *args, **kwargs)

    def _load_on_device(self, fn, *args, **kwargs):
        self._bind_device()
        return guarded_load(fn, *args, **kwargs)

    def __init__(self, device_ids: list[int] | None = None,
                 gpu_memory_utilization: float = 0.85,
                 max_images_per_request: int = 1, max_steps: int = 50, default_steps: int = 0):
        self._state = EngineState.STOPPED
        self._pipe = None
        self._model = ""  # EnginePool.start_paused() writes this directly
        self._device_ids = device_ids or [0]
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_images_per_request = max(1, int(max_images_per_request))
        self._max_steps = max(1, int(max_steps))
        self._default_steps = max(0, int(default_steps or 0))
        self._active_requests = 0

        self._transition_lock = asyncio.Lock()
        # One render on the pipeline at a time. A diffusers pipeline is stateful — its
        # scheduler keeps the step index — so two renders at once corrupt each other
        # ("index 9 is out of bounds for dimension 0 with size 9" under a soak test),
        # and two 1024x1024 renders do not fit on a 10 GB card anyway.
        self._render_lock = asyncio.Lock()
        self._pending_pause: YieldUrgency | None = None
        self._pending_resume = False
        # A yield that arrived while a generation was in flight. The step loop reads it
        # between steps; it is the only interruption point available to us.
        self._interrupt_requested = False
        self._accepting = True

    # --- advertised capabilities (read by EnginePool and the gateway) -------------

    @property
    def is_vlm(self) -> bool:
        """False: this engine GENERATES images, it does not read them. The gateway's
        "vision" feature (image input) must not be advertised by an image generator."""
        return False

    @property
    def is_image_gen(self) -> bool:
        """True: the loaded model generates images. Advertised as the "image_gen"
        feature so the gateway only routes /v1/images/generations here."""
        return True

    @property
    def max_images_per_prompt(self) -> int:
        """0: not an image-input model. Kept so the pool's getattr has a real value."""
        return 0

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def default_steps(self) -> int:
        """Denoising steps for a request that names none (OpenAI clients never do).

        The configured default_steps when set; otherwise by model: distilled few-step
        models (turbo / lightning / schnell / LCM) are made for ~4 steps, full models
        such as SDXL base or SSD-1B need ~25 for a sharp picture."""
        if self._default_steps:
            return min(self._default_steps, self._max_steps)
        name = (self._model or "").lower()
        few = any(k in name for k in ("turbo", "lightning", "schnell", "lcm", "hyper"))
        return min(4 if few else 25, self._max_steps)

    @property
    def max_images_per_request(self) -> int:
        """Cap on how many images one generation request may ask for."""
        return self._max_images_per_request

    @property
    def current_model(self) -> str:
        return self._model

    # --- lifecycle ----------------------------------------------------------------

    async def start(self, model_path: str) -> None:
        async with self._transition_lock:
            self._state = EngineState.STARTING
            self._model = model_path
        await self._load_pipeline()
        await self._after_load()

    async def _load_pipeline(self) -> None:
        """Build the diffusers pipeline on this engine's device(s)."""
        from diffusers import DiffusionPipeline  # imported lazily: only image workers pay for it

        device = f"cuda:{self._device_ids[0]}"
        logger.info("loading diffusion pipeline model=%s device=%s gpu_memory_utilization=%.2f",
                    self._model, device, self._gpu_memory_utilization)
        t0 = time.time()
        # local_files_only: the model manager has already resolved and (if needed)
        # downloaded the weights; reaching the network here would defeat FOC pinning.
        load_kwargs = {"local_files_only": True}
        load_kwargs.update(_precision_kwargs(self._model))
        # guarded_load: from_pretrained flips torch's process-wide default dtype while
        # it runs; see torch_globals. _load_on_device: it also calls
        # torch.cuda.empty_cache() part-way, which acts on the thread's CURRENT device —
        # card 0 on a fresh pool thread, full of the 32B on a test worker — and failed there with
        # "CUDA error: out of memory" (1–2 of the first ~40 SDXL <-> SSD-1B switches
        # after each restart, while the card the pipeline goes to had 9 GB free).
        pipe = await asyncio.to_thread(
            self._load_on_device,
            DiffusionPipeline.from_pretrained,
            self._model,
            **load_kwargs,
        )
        pipe = pipe.to(device)
        # Attention slicing trades a little speed for a lot of VRAM, which is what makes
        # a 10 GB card viable at all. Enablement is best-effort: not every pipeline in
        # every diffusers version exposes it.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            logger.debug("attention slicing unavailable for this pipeline (ignoring)")
        # The VAE decode is the peak: on a 10 GB card SDXL renders a 1024x1024 latent
        # fine but runs out of memory decoding it in one piece (measured: ~7.4 GB for
        # the loaded pipeline, +1 GB requested by the decode). Tiled decoding bounds that
        # peak at a small, size-independent cost. Same best-effort rule as above.
        #
        # Enabling tiling is NOT enough on its own: the VAE only tiles a latent strictly
        # LARGER than tile_latent_min_size, and SDXL ships 1024 px tiles — so the
        # 1024x1024 render (latent 128 = the threshold) still decodes in one piece. On
        # top of that, SDXL's VAE has force_upcast (fp16 overflows), so the decode runs
        # in fp32 and doubles its memory. Shrinking the tile to _VAE_TILE_PX makes the
        # standard sizes actually tile.
        vae = getattr(pipe, "vae", None)
        for name in ("enable_tiling", "enable_slicing"):
            try:
                getattr(vae, name)()
            except Exception:
                logger.debug("vae %s unavailable for this pipeline (ignoring)", name)
        _shrink_vae_tiles(vae)
        self._pipe = pipe
        logger.info("diffusion pipeline ready in %.1fs", time.time() - t0)

    async def _after_load(self) -> None:
        """Mark the engine running, then honour a yield that arrived during the load.

        Called after EVERY load path (start, resume, switch, post-unload reload) so that
        a pause-during-load is never dropped — the same contract VLLMEngine implements.
        Kept separate from _load_pipeline so a caller that overrides the loading itself
        still gets this handled, and so the pending-pause rule lives in one place.
        """
        async with self._transition_lock:
            self._state = EngineState.RUNNING
            pending = self._pending_pause
            self._pending_pause = None
        if pending is not None:
            logger.info("pause was requested during load — unloading now")
            await self.pause(pending)

    async def _destroy_pipeline(self) -> None:
        if self._pipe is None:
            return
        # In-process teardown: drop the reference, then collect. No subprocess to reap.
        self._pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("torch cache clear skipped", exc_info=True)
        # Confirm the memory actually came back before reporting PAUSED: the scheduler
        # will not start mining until we say the GPU is free, and reporting early would
        # hand a still-occupied card to WindowPoSt.
        await self._wait_for_vram_release()

    async def _wait_for_vram_release(self, timeout: float = 20.0) -> None:
        """Poll nvidia-smi until this device's free ratio clears the configured budget.

        Deliberately NOT torch.cuda.mem_get_info(): in a multi-instance worker the CUDA
        context binds to whichever device was first initialised, so it can report a
        sibling's memory. nvidia-smi queries the physical device directly.
        """
        gpu_id = self._device_ids[0]
        deadline = time.time() + timeout
        while time.time() < deadline:
            ratio = query_gpu_free_ratio(gpu_id)
            if ratio is None:
                # No nvidia-smi (dev box / mock). Nothing to confirm against; do not
                # block a yield on a tool that is not there.
                logger.debug("nvidia-smi unavailable — skipping VRAM release confirmation")
                return
            if ratio >= (1.0 - self._gpu_memory_utilization):
                return
            await asyncio.sleep(0.5)
        logger.warning("VRAM release not confirmed within %.0fs on gpu %d — proceeding "
                       "(the scheduler's own check is authoritative)", timeout, gpu_id)

    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        async with self._transition_lock:
            if self._state in (EngineState.LOADING, EngineState.STARTING):
                self._pending_pause = urgency
                logger.info("pause requested during %s (urgency=%s), will unload after load",
                            self._state.value, urgency.name)
                return
            if self._state != EngineState.RUNNING:
                return

            logger.info("pausing diffusion engine (urgency=%s, inflight=%d)",
                        urgency.name, self._active_requests)
            # Stop accepting either way; NORMAL additionally lets in-flight work finish.
            self._accepting = False
            # Signal the step loops. They check this BETWEEN steps — see the module
            # docstring for why a step cannot be abandoned mid-pass.
            self._interrupt_requested = True
            if urgency != YieldUrgency.NORMAL:
                self._state = EngineState.UNLOADING

        if urgency == YieldUrgency.NORMAL:
            await self._drain_requests(timeout=_DIFFUSION_DRAIN_TIMEOUT)
            async with self._transition_lock:
                if self._state != EngineState.RUNNING:
                    return  # an immediate pause took over
                self._state = EngineState.UNLOADING

        async with self._transition_lock:
            # IMMEDIATE skipped the drain, but the pipeline cannot be released while a
            # render thread still holds it: dropping our reference does not free the
            # tensors the thread is using, and the empty_cache() that follows runs too
            # early to return them. The step callback makes that render stop within one
            # step, so a short wait here is enough — and without it the engine reports
            # PAUSED while the card is still full (seen on real hardware: 9480 MiB held
            # after an IMMEDIATE yield).
            if self._active_requests:
                await self._drain_requests(timeout=_IMMEDIATE_STOP_TIMEOUT)
            await self._destroy_pipeline()
            self._interrupt_requested = False
            self._accepting = True
            if self._pending_resume:
                self._pending_resume = False
                self._state = EngineState.LOADING
            else:
                self._state = EngineState.PAUSED
                logger.info("diffusion engine paused — GPU VRAM released")
                return
        await self._load_pipeline()
        await self._after_load()

    async def resume(self) -> None:
        async with self._transition_lock:
            if self._state == EngineState.UNLOADING:
                self._pending_resume = True
                return
            if self._state in (EngineState.RUNNING, EngineState.LOADING, EngineState.STARTING):
                return
            self._state = EngineState.LOADING
        try:
            await self._load_pipeline()
        except BaseException:
            # Back to PAUSED so the next resume retries; staying LOADING would make
            # every later resume return early and the pool would never come back.
            async with self._transition_lock:
                self._pipe = None
                self._state = EngineState.PAUSED
            raise
        await self._after_load()

    async def stop(self) -> None:
        async with self._transition_lock:
            self._state = EngineState.STOPPING
            await self._destroy_pipeline()
            self._state = EngineState.STOPPED

    async def switch_model(self, new_model_path: str) -> None:
        if new_model_path == self._model and self._pipe is not None:
            return
        old_model = self._model
        await self.pause(YieldUrgency.NORMAL)
        async with self._transition_lock:
            self._model = new_model_path
            self._state = EngineState.LOADING
        try:
            await self._load_pipeline()
        except BaseException as e:
            # Roll back like the vLLM engine: the pool keeps serving the model it had
            # rather than staying down on one that cannot load.
            logger.error("model switch failed, rolling back to %s: %s", old_model, e)
            async with self._transition_lock:
                self._pipe = None
                self._model = old_model
                self._state = EngineState.PAUSED
            try:
                await self.resume()
            except Exception as rollback_err:
                logger.error("rollback also failed: %s", rollback_err)
            raise
        await self._after_load()

    def request_pause(self) -> None:
        """Non-blocking preemption signal from the scheduler listener.

        The listener calls this the moment a yield is detected, even while a long resume
        is running. For this engine the only long operation is a generation, which is
        interrupted by the same flag.
        """
        self._interrupt_requested = True

    def is_available(self) -> bool:
        return self._state == EngineState.RUNNING and self._pipe is not None and self._accepting

    def status(self) -> EngineStatus:
        return EngineStatus(
            state=self._state,
            active_requests=self._active_requests,
            gpu_utilization_pct=0.0,
            loaded_model=self._model,
            gpu_id=self._device_ids[0] if self._device_ids else -1,
        )

    async def _drain_requests(self, timeout: float) -> None:
        if self._active_requests == 0:
            return
        logger.info("draining %d in-flight image generations (timeout=%.0fs)",
                    self._active_requests, timeout)
        deadline = time.time() + timeout
        while self._active_requests > 0:
            if time.time() > deadline:
                logger.warning("drain timeout, %d image generations still in flight",
                               self._active_requests)
                return
            await asyncio.sleep(0.2)

    # --- generation ---------------------------------------------------------------

    async def generate(self, prompt: str, **kwargs) -> GenerateResult:
        """Render one or more images. Returns a GenerateResult whose `quantity` is the
        number of images produced.

        The response's asset bytes are carried in `GenerateResult.images` as PNG bytes;
        the caller hashes them for the receipt's asset_hash.
        """
        if not self.is_available():
            raise RuntimeError("Engine not available — GPU yielded to mining")

        async with self._render_lock:
            if not self.is_available():
                raise RuntimeError("Engine not available — GPU yielded to mining")
            images = await self._render(prompt, **kwargs)
        return GenerateResult(
            text="",
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stop",
            images=images,
        )

    async def generate_stream(self, prompt: str, **kwargs):
        """Image generation is not incremental — there is no partial image to stream.

        Implemented anyway (rather than left absent) because EnginePool and the API's
        streaming branch call it unconditionally: a missing method here would surface as
        an AttributeError on the first streaming request instead of a clear refusal. It
        yields nothing and lets the caller fall back to the non-streaming path.
        """
        raise NotImplementedError("image generation does not support streaming")
        yield  # pragma: no cover — makes this an async generator

    def _make_generator(self, seed: int):
        """Build a seeded RNG for the pipeline.

        A method (rather than an inline torch call) so a test can substitute a
        torch-free stand-in: torch is only present on a GPU worker, and the render
        plumbing should be exercisable without it. torch is imported here rather than at
        module scope for the same reason — importing this module must not require torch.
        """
        import torch
        return torch.Generator(device="cpu").manual_seed(seed)

    async def _render(self, prompt: str, **kwargs) -> list[bytes]:
        """Run the denoising loop for one request, honouring a yield between steps."""
        n = int(kwargs.get("n", 1) or 1)
        if n < 1:
            n = 1
        if n > self._max_images_per_request:
            raise ValueError(
                f"too many images: {n} requested, at most {self._max_images_per_request}")

        steps = int(kwargs.get("steps") or self.default_steps)
        guidance = kwargs.get("guidance")
        height = int(kwargs.get("height", 0) or 0)
        width = int(kwargs.get("width", 0) or 0)
        seed = kwargs.get("seed")

        gen_kwargs = {"num_inference_steps": steps, "num_images_per_prompt": n}
        if guidance is not None:
            gen_kwargs["guidance_scale"] = float(guidance)
        if height:
            gen_kwargs["height"] = height
        if width:
            gen_kwargs["width"] = width

        if seed is not None:
            gen_kwargs["generator"] = self._make_generator(int(seed))

        self._active_requests += 1
        interrupted = False
        try:
            # Stop between denoising steps when a yield arrives. diffusers pipelines that
            # support it call `callback_on_step_end` after every step and skip the rest
            # once `_interrupt` is set — so an IMMEDIATE yield frees the card within one
            # step instead of after the whole image. Pipelines without the hook (or a
            # test stub) simply run to completion; the yield then waits for them.
            pipe = self._pipe
            if _accepts_step_callback(pipe):
                def _on_step(p, _step, _t, cb_kwargs):
                    if self._interrupt_requested and self._state != EngineState.RUNNING:
                        p._interrupt = True
                    return cb_kwargs
                gen_kwargs["callback_on_step_end"] = _on_step
            try:
                result = await asyncio.to_thread(self._on_device, pipe, prompt, **gen_kwargs)
            except Exception as e:  # noqa: BLE001
                if "out of memory" not in str(e).lower():
                    raise
                # Return the card's cache before anything else runs on it: left
                # fragmented, the NEXT renders failed too (three 500s in a row on a test worker).
                # A RuntimeError reaches the client as 503, so the gateway retries.
                self._release_cuda_cache()
                raise RuntimeError("image generation ran out of GPU memory; retry") from None
            # A pipeline that was interrupted still decodes whatever latents it had and
            # returns them as an image — a half-denoised picture. That is not what was
            # asked for and must never be returned or billed.
            interrupted = bool(getattr(pipe, "_interrupt", False))
            del pipe
        finally:
            self._active_requests -= 1

        if interrupted:
            raise RuntimeError("Engine not available — GPU yielded to mining during generation")

        # Encode each PIL image to PNG. The bytes are the billable artifact: the gateway
        # re-hashes them, and the receipt commits that hash.
        out: list[bytes] = []
        for img in result.images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out

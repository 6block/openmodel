"""DiffusionEngine: the image-generation engine class.

Exercised with a stub pipeline, because the real one needs diffusers and a GPU. What is
tested here is the LIFECYCLE AND CONTRACT the pool and the scheduler depend on — not the
denoising maths — plus the one behaviour that is genuinely new: a diffusers step loop
cannot be abandoned mid-step, so a yield interrupts between steps rather than inside one.
"""
import asyncio

import pytest

from src.inference.diffusion_engine import DiffusionEngine
from src.inference.engine import EngineState, YieldUrgency


class _StubPipeline:
    """Stands in for a diffusers pipeline.

    Records the call, sleeps for a configurable time (so a test can hold a generation
    open and yield during it), and returns objects with `.images` like a real call.
    """

    def __init__(self, delay=0.0, n_images=1, fail=None):
        self.delay = delay
        self.n_images = n_images
        self.fail = fail
        self.calls = []

    def __call__(self, prompt, num_inference_steps=4, callback_on_step_end=None, **kwargs):
        # Models diffusers' step loop: _interrupt is reset per call, the callback runs
        # after every step, and once it sets _interrupt the remaining steps are skipped
        # (the call still returns an image, as diffusers does).
        self.calls.append({"prompt": prompt, "num_inference_steps": num_inference_steps, **kwargs})
        import time
        self._interrupt = False
        steps = max(1, int(num_inference_steps))
        self.steps_run = 0
        for i in range(steps):
            if self._interrupt:
                continue
            time.sleep(self.delay / steps)
            self.steps_run += 1
            if callback_on_step_end is not None:
                callback_on_step_end(self, i, 0, {})
        if self.fail:
            raise self.fail

        class _Img:
            def __init__(self, i):
                self._i = i

            def save(self, buf, format=None):  # noqa: A002 — diffusers' own signature
                # Real PNG bytes, so the test exercises the same encoding path a real
                # render would (and the digest is of decodable image data).
                from src.inference.engine import _mock_png
                buf.write(_mock_png(self._i))

        class _Result:
            images = [_Img(i) for i in range(self.n_images)]

        return _Result()


def _engine(**kw) -> DiffusionEngine:
    return DiffusionEngine(device_ids=[0], max_images_per_request=kw.pop("max_images_per_request", 4), **kw)


async def _started(pipe=None, **kw) -> DiffusionEngine:
    e = _engine(**kw)
    # Inject the stub instead of loading diffusers: _load_pipeline is the only thing
    # that touches the real stack.
    async def _fake_load():
        e._pipe = pipe if pipe is not None else _StubPipeline()

    e._load_pipeline = _fake_load
    # A torch-free seeded RNG: torch is a GPU-worker dependency and the render
    # plumbing must be testable without it.
    e._make_generator = lambda seed: ("seeded", seed)
    await e.start("/models/fake-flux")
    return e


# --- the implicit contract with EnginePool ----------------------------------------

def test_pool_contract_attributes_exist():
    """EnginePool.start_paused() assigns `_model` and `_state` directly. If either name
    changes, that path silently misbehaves — so pin them."""
    e = _engine()
    assert hasattr(e, "_model")
    assert hasattr(e, "_state")
    assert e._state is EngineState.STOPPED


def test_capability_flags_are_the_generator_shape():
    """An image generator must advertise image_gen and NOT vision: the two route
    different request types, and conflating them would send chat-with-image traffic to a
    worker that cannot read images."""
    e = _engine()
    assert e.is_image_gen is True
    assert e.is_vlm is False
    assert e.max_images_per_prompt == 0
    assert e.max_images_per_request == 4


def test_pause_and_switch_model_methods_exist():
    """The listener calls request_pause() (via hasattr) and every request path calls
    switch_model(); both must be present on the engine, not only on a base class."""
    e = _engine()
    for name in ("request_pause", "switch_model", "current_model", "pause", "resume",
                 "stop", "is_available", "status", "generate", "generate_stream"):
        assert hasattr(e, name), f"missing {name}"


def test_request_pause_is_a_safe_noop_that_signals():
    """request_pause() is called by the listener the instant a yield is detected. It must
    not raise, and it must set the flag the step loop reads."""
    e = _engine()
    e.request_pause()
    assert e._interrupt_requested is True


# --- state transitions the scheduler depends on ------------------------------------

def test_pause_leaves_state_paused():
    """After pause() the reported state must be PAUSED, or the listener tells the
    scheduler the GPU is still held and the yield is retried forever."""
    async def run():
        e = await _started()
        assert e.status().state is EngineState.RUNNING
        await e.pause(YieldUrgency.NORMAL)
        assert e.status().state is EngineState.PAUSED
        assert e.is_available() is False
    asyncio.run(run())


def test_resume_reloads_to_running():
    async def run():
        e = await _started()
        await e.pause(YieldUrgency.NORMAL)
        await e.resume()
        assert e.status().state is EngineState.RUNNING
        assert e.is_available() is True
    asyncio.run(run())


def test_pause_during_load_defers_until_loaded():
    """A yield arriving while weights are loading must be honoured AFTER the load, not
    dropped — the same contract VLLMEngine implements with _pending_pause."""
    async def run():
        e = _engine()
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_load():
            started.set()
            await release.wait()
            e._pipe = _StubPipeline()

        e._load_pipeline = _slow_load
        task = asyncio.ensure_future(e.start("/models/fake-flux"))
        await started.wait()
        # A yield arrives mid-load.
        await e.pause(YieldUrgency.NORMAL)
        release.set()
        await task
        assert e.status().state is EngineState.PAUSED
    asyncio.run(run())


def test_stop_releases_and_marks_stopped():
    async def run():
        e = await _started()
        await e.stop()
        assert e.status().state is EngineState.STOPPED
        assert e._pipe is None
    asyncio.run(run())


def test_switch_model_changes_current_model():
    async def run():
        e = await _started()
        await e.switch_model("/models/other")
        assert e.current_model == "/models/other"
        assert e.status().state is EngineState.RUNNING
    asyncio.run(run())


def test_switch_to_same_model_is_a_noop():
    async def run():
        e = await _started()
        pipe = e._pipe
        await e.switch_model("/models/fake-flux")
        # Same model, pipeline present → nothing was torn down.
        assert e._pipe is pipe
    asyncio.run(run())


# --- generation --------------------------------------------------------------------

def test_generate_returns_one_png_per_image():
    async def run():
        e = await _started(_StubPipeline(n_images=3))
        res = await e.generate("a red cube", n=3)
        assert len(res.images) == 3
        # Each is real PNG bytes, not a placeholder.
        for b in res.images:
            assert b.startswith(b"\x89PNG\r\n\x1a\n")
    asyncio.run(run())


def test_generate_passes_render_parameters_through():
    async def run():
        pipe = _StubPipeline()
        e = await _started(pipe)
        await e.generate("a cat", n=2, steps=8, guidance=3.5, height=512, width=512, seed=7)
        call = pipe.calls[0]
        assert call["prompt"] == "a cat"
        assert call["num_inference_steps"] == 8
        assert call["guidance_scale"] == 3.5
        assert call["height"] == 512 and call["width"] == 512
        assert call["num_images_per_prompt"] == 2
        assert "generator" in call
    asyncio.run(run())


def test_generate_rejects_more_images_than_allowed():
    """The per-request cap is enforced by the engine too, not only at the API edge — a
    direct engine caller must not be able to exceed it."""
    async def run():
        e = await _started(_StubPipeline(), max_images_per_request=2)
        with pytest.raises(ValueError, match="too many images"):
            await e.generate("x", n=3)
    asyncio.run(run())


def test_generate_refuses_when_paused():
    async def run():
        e = await _started()
        await e.pause(YieldUrgency.NORMAL)
        with pytest.raises(RuntimeError, match="not available"):
            await e.generate("x", n=1)
    asyncio.run(run())


def test_generate_stream_refuses_clearly():
    """Image generation has no partial output, so streaming is refused — but the method
    must EXIST (EnginePool and the API call it unconditionally; an AttributeError would
    be a confusing way to discover this)."""
    async def run():
        e = await _started()
        with pytest.raises(NotImplementedError):
            async for _ in e.generate_stream("x"):
                pass
    asyncio.run(run())


# --- the yield-granularity difference ----------------------------------------------

def test_yield_waits_for_the_in_flight_generation():
    """A soft yield must let an in-flight render finish: a diffusion step cannot be
    abandoned mid-pass, so the engine finishes the step rather than corrupting the CUDA
    context. This is the deliberate difference from the token engine."""
    async def run():
        # A generation that takes long enough to still be running when the yield lands.
        pipe = _StubPipeline(delay=1.0, n_images=1)
        e = await _started(pipe)

        gen = asyncio.ensure_future(e.generate("x", n=1))
        await asyncio.sleep(0.2)  # let it enter the render
        assert e._active_requests == 1

        pause = asyncio.ensure_future(e.pause(YieldUrgency.NORMAL))
        await asyncio.sleep(0.2)
        # The generation is still running and the engine has stopped accepting.
        assert e._accepting is False

        await gen   # it completes rather than being killed
        await pause
        assert e.status().state is EngineState.PAUSED
        assert e._active_requests == 0
    asyncio.run(run())


def test_immediate_yield_stops_between_steps_and_frees_the_card():
    """IMMEDIATE stops the render at the next step boundary, waits for that thread to
    return, and only then releases the pipeline. Found on real hardware: releasing while
    the render thread still ran left the card full (9480 MiB) under a PAUSED state."""
    async def run():
        pipe = _StubPipeline(delay=2.0, n_images=1)
        e = await _started(pipe)
        gen = asyncio.ensure_future(e.generate("x", n=1, steps=20))
        await asyncio.sleep(0.3)
        await e.pause(YieldUrgency.IMMEDIATE)
        # By the time pause returns the render thread has already exited.
        assert e._active_requests == 0
        assert e.status().state is EngineState.PAUSED
        assert e._pipe is None
        assert pipe.steps_run < 20, "the render must stop between steps, not run to the end"
        with pytest.raises(RuntimeError):
            await gen
    asyncio.run(run())


def test_interrupted_render_is_not_returned():
    """A pipeline interrupted mid-loop still decodes its partial latents into an image.
    That half-denoised picture must never reach the caller (it would be billed)."""
    async def run():
        pipe = _StubPipeline(delay=1.0, n_images=1)
        e = await _started(pipe)
        gen = asyncio.ensure_future(e.generate("x", n=1, steps=10))
        await asyncio.sleep(0.2)
        await e.pause(YieldUrgency.IMMEDIATE)
        with pytest.raises(RuntimeError, match="yielded"):
            await gen
    asyncio.run(run())


def test_uninterrupted_render_runs_every_step():
    async def run():
        pipe = _StubPipeline(delay=0.05, n_images=1)
        e = await _started(pipe)
        imgs = await e._render("x", n=1, steps=6)
        assert len(imgs) == 1 and pipe.steps_run == 6
    asyncio.run(run())


def test_drain_timeout_is_generous_for_diffusion():
    """The image drain timeout must be far longer than the token engine's 30s: one image
    is tens of seconds and a queue of them is minutes."""
    from src.inference import diffusion_engine as de
    assert de._DIFFUSION_DRAIN_TIMEOUT >= 120


def test_renders_are_serialized_on_one_pipeline():
    """A diffusers pipeline is stateful (its scheduler holds the step index). Two
    renders at once corrupted each other on a test worker ("index 9 is out of bounds"), so one
    engine renders one request at a time; the rest wait their turn and all succeed."""
    import threading

    class _Tracking(_StubPipeline):
        def __init__(self):
            super().__init__(delay=0.05)
            self.now = 0
            self.peak = 0
            self.mu = threading.Lock()

        def __call__(self, *a, **kw):
            with self.mu:
                self.now += 1
                self.peak = max(self.peak, self.now)
            try:
                return super().__call__(*a, **kw)
            finally:
                with self.mu:
                    self.now -= 1

    async def run():
        pipe = _Tracking()
        e = await _started(pipe)
        rs = await asyncio.gather(*[e.generate("x", n=1, steps=4) for _ in range(5)])
        return pipe.peak, rs
    peak, rs = asyncio.run(run())
    assert peak == 1, f"{peak} renders ran on the pipeline at once"
    assert all(len(r.images) == 1 for r in rs)


def test_render_runs_with_the_engine_card_current():
    """Renders run on pool threads that start on card 0; the engine's card must be
    made current in the rendering thread before the pipeline runs (same guard as the
    speech engines, whose libraries allocated on the thread's current device)."""
    import threading

    seen = []

    class _Pipe(_StubPipeline):
        def __call__(self, *a, **kw):
            seen.append(("render", threading.get_ident()))
            return super().__call__(*a, **kw)

    async def run():
        e = await _started(pipe=_Pipe())
        e._bind_device = lambda: seen.append(("bind", threading.get_ident()))
        await e.generate("a cube", n=1)
        renders = [i for i, (k, _) in enumerate(seen) if k == "render"]
        assert renders, seen
        for i in renders:
            assert i > 0 and seen[i - 1] == ("bind", seen[i][1]), seen
    asyncio.run(run())


def test_pipeline_load_runs_with_the_engine_card_current(monkeypatch):
    """from_pretrained empties the CUDA cache part-way, on the thread's CURRENT device; a
    fresh pool thread starts on card 0 (full of the 32B on a test worker), where that failed with
    "CUDA error: out of memory". The load must bind the engine's card in its own thread."""
    import sys
    import threading
    import types

    seen = []

    class _Loaded(_StubPipeline):
        def to(self, device):
            return self

    class _DP:
        @staticmethod
        def from_pretrained(path, **kw):
            seen.append(("load", threading.get_ident()))
            return _Loaded()

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(DiffusionPipeline=_DP))

    async def run():
        e = _engine()
        e._bind_device = lambda: seen.append(("bind", threading.get_ident()))
        await e.start("/models/fake-sdxl")
        loads = [i for i, (k, _) in enumerate(seen) if k == "load"]
        assert loads, seen
        for i in loads:
            assert i > 0 and seen[i - 1] == ("bind", seen[i][1]), seen
    asyncio.run(run())

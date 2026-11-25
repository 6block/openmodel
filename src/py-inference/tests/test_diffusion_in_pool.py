"""The diffusion engine behind EnginePool.

The pool is how the engine is actually driven, and it reads several things off its
members that are NOT on the InferenceEngine ABC: `is_vlm`, `is_image_gen`,
`max_images_per_prompt`, `max_images_per_request`, and the `_model`/`_state` fields that
start_paused() assigns. A diffusion engine missing any of them would fail at runtime, not
at construction — so pin them through the pool, not just on the class.
"""
import asyncio

from src.inference.diffusion_engine import DiffusionEngine
from src.inference.engine import EngineState, YieldUrgency
from src.inference.engine_pool import EnginePool


class _StubPipeline:
    def __call__(self, prompt, **kwargs):
        class _Img:
            def save(self, buf, format=None):  # noqa: A002
                from src.inference.engine import _mock_png
                buf.write(_mock_png(0))

        class _Result:
            images = [_Img() for _ in range(kwargs.get("num_images_per_prompt", 1))]

        return _Result()


async def _started_engine():
    e = DiffusionEngine(device_ids=[0], max_images_per_request=4)

    async def _fake_load():
        e._pipe = _StubPipeline()

    e._load_pipeline = _fake_load
    e._make_generator = lambda seed: ("seeded", seed)
    await e.start("/models/fake-flux")
    return e


def test_pool_reports_image_gen_capability():
    async def run():
        e = await _started_engine()
        p = EnginePool(mode="tensor_parallel", engines=[e])
        assert p.is_image_gen is True
        assert p.is_vlm is False
        # EnginePool reads the output cap so the gateway can publish it.
        assert p.max_images_per_request == 4
    asyncio.run(run())


def test_pool_generates_through_the_diffusion_engine():
    async def run():
        e = await _started_engine()
        p = EnginePool(mode="tensor_parallel", engines=[e])
        res = await p.generate("a red cube", n=2)
        assert len(res.images) == 2
    asyncio.run(run())


def test_start_paused_writes_the_fields_the_pool_expects():
    """EnginePool.start_paused() assigns `_model` and `_state` on each member DIRECTLY.
    This is the hardest implicit contract in the codebase — pin that a diffusion engine
    tolerates it and reports the right state afterwards."""
    async def run():
        e = DiffusionEngine(device_ids=[0])
        p = EnginePool(mode="tensor_parallel", engines=[e])
        await p.start_paused("/models/fake-flux")
        assert e.status().state is EngineState.PAUSED
        assert p.current_model == "/models/fake-flux"
        assert p.is_available() is False
    asyncio.run(run())


def test_pool_pause_releases_and_reports_paused():
    """The listener's yield path goes through the POOL. After a pool pause the reported
    state must be PAUSED, or the scheduler retries the yield indefinitely."""
    async def run():
        e = await _started_engine()
        p = EnginePool(mode="tensor_parallel", engines=[e])
        await p.pause(YieldUrgency.NORMAL)
        assert p.status().state is EngineState.PAUSED
        assert p.is_available() is False
        await p.resume()
        assert p.status().state is EngineState.RUNNING
    asyncio.run(run())


def test_pool_switch_model_reaches_the_engine():
    async def run():
        e = await _started_engine()
        p = EnginePool(mode="tensor_parallel", engines=[e])
        await p.switch_model("/models/other")
        assert p.current_model == "/models/other"
    asyncio.run(run())


def test_create_engine_pool_builds_a_diffusion_engine():
    """The config → factory path: engine: diffusion must produce a DiffusionEngine, and
    carry the image cap through the pool builder."""
    from src.config import AppConfig

    cfg = AppConfig()
    cfg.mode = "prod"
    cfg.inference.engine = "diffusion"
    cfg.inference.model = "/models/flux-schnell"
    cfg.inference.max_images_per_generation = 3
    cfg.inference.multi_gpu.mode = "tensor_parallel"
    cfg.inference.multi_gpu.gpu_count = 1
    cfg.inference.multi_gpu.device_ids = [0]

    p = create_engine_pool_from(cfg)
    assert isinstance(p._engines[0], DiffusionEngine)
    assert p.max_images_per_request == 3


def create_engine_pool_from(cfg):
    from src.inference.engine_pool import create_engine_pool
    return create_engine_pool(cfg)

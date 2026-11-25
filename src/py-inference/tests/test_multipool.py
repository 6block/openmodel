"""Multi-pool aggregator: one worker, several modality pools on disjoint GPUs.

The yield contract is what matters most here. The scheduler hands the cards to mining
only when the worker reports PAUSED, and mining needs every card — so PAUSED must mean
"every pool released", never "most of them".
"""
import asyncio

import pytest

from src.inference.engine import EngineState, MockEngine, YieldUrgency
from src.inference.engine_pool import EnginePool
from src.inference.multipool import MultiPoolEngine, Pool, normalize_model_id

TEXT = "/models/Qwen--Qwen3-32B-AWQ"
IMAGE = "/models/stabilityai--stable-diffusion-xl-base-1.0"
VL = "/models/Qwen--Qwen3-VL-8B-Instruct"


def _pool(name, model, devices, *, image_gen=False, supported=None):
    e = MockEngine(latency_sec=0.01, gpu_id=devices[0], image_gen=image_gen)
    e._model = model
    return Pool(name=name, engine=EnginePool("tensor_parallel", [e]), devices=devices,
                supported_models=supported or [model])


def _multi():
    return MultiPoolEngine([
        _pool("text", TEXT, [0, 1, 2, 3], supported=[TEXT, VL]),
        _pool("image", IMAGE, [4], image_gen=True),
    ])


def test_normalize_matches_gateway_spellings():
    assert normalize_model_id("/models/Qwen--Qwen3-32B-AWQ") == normalize_model_id("Qwen/Qwen3-32B-AWQ")


def test_overlapping_devices_are_refused():
    with pytest.raises(ValueError, match="GPU 4"):
        MultiPoolEngine([_pool("a", TEXT, [3, 4]), _pool("b", IMAGE, [4])])


def test_card_sharing_needs_both_pools_to_opt_in():
    def sp(name, model, share):
        p = _pool(name, model, [5])
        p.share_devices = share
        return p
    asr, tts = "/models/Systran--faster-whisper-small", "/models/hexgrad--Kokoro-82M"
    m = MultiPoolEngine([sp("asr", asr, True), sp("tts", tts, True)])
    assert len(m.pools) == 2
    with pytest.raises(ValueError, match="GPU 5"):
        MultiPoolEngine([sp("asr", asr, True), sp("tts", tts, False)])
    with pytest.raises(ValueError, match="GPU 5"):
        MultiPoolEngine([sp("asr", asr, False), sp("tts", tts, True)])


def test_shared_card_pauses_and_resumes_both_pools():
    async def run():
        a = _pool("asr", "/models/a", [5]); a.share_devices = True
        t = _pool("tts", "/models/t", [5]); t.share_devices = True
        m = MultiPoolEngine([_pool("text", TEXT, [0]), a, t])
        await m.start(TEXT)
        await m.pause(YieldUrgency.IMMEDIATE)
        assert m.status().state is EngineState.PAUSED
        assert all(p.engine.status().state is EngineState.PAUSED for p in m.pools)
        await m.resume()
        assert all(p.engine.status().state is EngineState.RUNNING for p in m.pools)
    asyncio.run(run())


def test_requests_route_to_the_owning_pool():
    async def run():
        m = _multi()
        await m.start(TEXT)
        r = await m.generate("x", model="stabilityai/stable-diffusion-xl-base-1.0", n=1)
        assert r.images, "an image-model request must reach the image pool"
        r = await m.generate("hello", model="Qwen/Qwen3-32B-AWQ")
        assert r.text and not r.images
        # A model the text pool can switch to is owned by it, not by the image pool.
        assert m.owner(VL).name == "text"
    asyncio.run(run())


def test_unknown_model_is_an_error_not_pool_zero():
    """Falling through to the first pool would hand e.g. a speech request to a text
    model. An unknown model must fail clearly."""
    async def run():
        m = _multi()
        await m.start(TEXT)
        assert m.owner("hexgrad/Kokoro-82M") is None
        with pytest.raises(ValueError, match="no pool"):
            await m.generate("x", model="hexgrad/Kokoro-82M")
    asyncio.run(run())


def test_pause_fans_out_and_paused_means_every_pool():
    async def run():
        m = _multi()
        await m.start(TEXT)
        assert m.status().state is EngineState.RUNNING
        await m.pause(YieldUrgency.IMMEDIATE)
        assert all(p.engine.status().state is EngineState.PAUSED for p in m.pools)
        assert m.status().state is EngineState.PAUSED
        await m.resume()
        assert all(p.engine.status().state is EngineState.RUNNING for p in m.pools)
    asyncio.run(run())


def test_one_pool_still_running_is_not_paused():
    """The aggregate must never claim more than its slowest pool."""
    async def run():
        m = _multi()
        await m.start(TEXT)
        await m.pools[0].engine.pause(YieldUrgency.IMMEDIATE)   # only the text pool
        assert m.status().state is not EngineState.PAUSED
    asyncio.run(run())


def test_a_pool_stuck_unloading_blocks_paused():
    async def run():
        m = _multi()
        await m.start(TEXT)
        await m.pause(YieldUrgency.IMMEDIATE)
        m.pools[1].engine._engines[0]._state = EngineState.UNLOADING
        assert m.status().state is EngineState.UNLOADING
    asyncio.run(run())


def test_failed_pool_pause_still_pauses_the_others():
    """One pool raising must not leave the rest on cards mining is about to take."""
    async def run():
        m = _multi()
        await m.start(TEXT)

        async def boom(_u=YieldUrgency.NORMAL):
            raise RuntimeError("stuck")
        m.pools[0].engine.pause = boom
        with pytest.raises(RuntimeError, match="text"):
            await m.pause(YieldUrgency.IMMEDIATE)
        assert m.pools[1].engine.status().state is EngineState.PAUSED
    asyncio.run(run())


def test_resume_stops_when_a_yield_arrives_midway(monkeypatch):
    # Both pools are vLLM-type here, so they resume one after the other.
    import src.inference.multipool as mp
    monkeypatch.setattr(mp, "_spawns_subprocess", lambda eng: True)

    async def run():
        m = _multi()
        await m.start(TEXT)
        await m.pause(YieldUrgency.IMMEDIATE)
        orig = m.pools[0].engine.resume

        async def resume_then_yield():
            await orig()
            m.request_pause()   # mining arrives while the first pool is loading
        m.pools[0].engine.resume = resume_then_yield
        await m.resume()
        assert m.pools[1].engine.status().state is EngineState.PAUSED, \
            "the next pool must not be loaded onto a card mining now needs"
    asyncio.run(run())


def test_switch_only_within_the_owning_pool():
    async def run():
        m = _multi()
        await m.start(TEXT)
        await m.switch_model(VL)
        assert m.pools[0].engine.current_model == VL
        assert m.pools[1].engine.current_model == IMAGE
        with pytest.raises(ValueError):
            await m.switch_model("hexgrad/Kokoro-82M")
    asyncio.run(run())


def test_capabilities_and_loaded_models_are_the_union():
    m = _multi()
    assert m.is_image_gen
    assert set(m.loaded_models) == {TEXT, IMAGE}
    assert m.current_model == TEXT   # primary pool, what old gateways read


def test_health_reports_every_resident_model():
    from fastapi.testclient import TestClient
    from src.inference.api_server import create_app
    m = _multi()
    asyncio.run(m.start(TEXT))
    body = TestClient(create_app(m)).get("/health").json()
    assert body["loaded_model"] == TEXT
    assert set(body["loaded_models"]) == {TEXT, IMAGE}
    assert "image_gen" in body["features"]
    assert body["multi_gpu"]["mode"] == "multi_pool"
    assert {p["name"] for p in body["multi_gpu"]["pools"]} == {"text", "image"}


def test_images_endpoint_reaches_the_image_pool_on_a_multi_pool_worker():
    from fastapi.testclient import TestClient
    from src.inference.api_server import create_app
    m = _multi()
    asyncio.run(m.start(TEXT))
    r = TestClient(create_app(m)).post("/v1/images/generations",
                                       json={"model": IMAGE, "prompt": "a red circle", "n": 1})
    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) == 1


def test_config_pools_build_a_multi_pool_engine(tmp_path):
    from src.config import load_config
    from src.inference.engine_pool import create_engine_pool
    p = tmp_path / "s.yaml"
    p.write_text(f"""
mode: dev
inference:
  api_token: shared
  pools:
    - name: text
      engine: mock
      model: {TEXT}
      device_ids: [0, 1]
    - name: image
      engine: mock
      model: {IMAGE}
      device_ids: [2]
""")
    cfg = load_config(str(p))
    assert [pc.name for pc in cfg.pools] == ["text", "image"]
    assert cfg.pools[0].inference.multi_gpu.device_ids == [0, 1]
    assert cfg.pools[1].inference.api_token == "shared", "shared keys are inherited"
    eng = create_engine_pool(cfg)
    assert isinstance(eng, MultiPoolEngine)
    assert eng.owner(IMAGE).name == "image"


def test_vllm_create_restores_cuda_visible_devices(monkeypatch):
    """A vLLM pool narrows CUDA_VISIBLE_DEVICES only while spawning its core. Left
    narrowed, an in-process pool created afterwards (diffusion on cuda:4, speech on
    cuda:5) would initialize CUDA seeing just cards 0-3 and fail to find its card."""
    import os
    from src.inference.engine import VLLMEngine

    seen = {}

    async def fake_inner(self):
        seen["during"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    monkeypatch.setattr(VLLMEngine, "_create_engine_inner", fake_inner)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    eng = VLLMEngine(device_ids=[0, 1, 2, 3], tensor_parallel_size=4)
    asyncio.run(eng._create_engine())
    assert seen["during"] == "0,1,2,3"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    asyncio.run(eng._create_engine())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_vllm_shutdown_runs_off_the_loop(monkeypatch):
    """vLLM's shutdown() blocks while its core exits. Run on the event loop it froze
    every other pool's pause; on a test worker the image pool started releasing 5 s late."""
    import time as _t
    from src.inference.engine import VLLMEngine

    class SlowEngine:
        def shutdown(self):
            _t.sleep(0.5)

    async def run():
        eng = VLLMEngine(device_ids=[0])
        eng._engine = SlowEngine()
        stamps = []

        async def ticker():
            t0 = _t.monotonic()
            while _t.monotonic() - t0 < 0.4:
                await asyncio.sleep(0.05)
                stamps.append(_t.monotonic() - t0)
        t = asyncio.ensure_future(ticker())
        await asyncio.sleep(0)          # ticker is running before the pause begins
        await eng._destroy_engine()
        await t
        return stamps
    stamps = asyncio.run(run())
    gaps = [b - a for a, b in zip([0.0] + stamps, stamps)]
    assert max(gaps) < 0.3, f"event loop stalled {max(gaps):.2f}s during shutdown"


def test_fast_pools_resume_before_vllm_pools(monkeypatch):
    """In-process pools (image, speech) reload in seconds; they must not wait behind
    vLLM reloads. Measured on a test worker: an image request queued during a yield timed out
    because SDXL resumed ~45 s after the yield ended, after two vLLM pools."""
    import src.inference.multipool as mp
    order = []

    cards = {"text": 0, "image": 4, "asr": 5}   # fixed: str hash() is randomized per process

    def mk(name, slow):
        p = _pool(name, f"/models/{name}", [cards[name]])
        orig = p.engine.resume

        async def resume():
            if slow:
                await asyncio.sleep(0.2)
            order.append(name)
            await orig()
        p.engine.resume = resume
        p._slow = slow
        return p

    monkeypatch.setattr(mp, "_spawns_subprocess", lambda eng: getattr(eng, "_slow_marker", False))
    pools = [mk("text", True), mk("image", False), mk("asr", False)]
    for p in pools:
        p.engine._slow_marker = p._slow

    async def run():
        m = MultiPoolEngine(pools)
        await m.start("/models/text")
        await m.pause(YieldUrgency.IMMEDIATE)
        await m.resume()
        assert order[:2] == ["image", "asr"], order
        assert order[-1] == "text"
        assert all(p.engine.status().state is EngineState.RUNNING for p in m.pools)
    asyncio.run(run())


def test_yield_during_fast_resume_skips_vllm_pools(monkeypatch):
    import src.inference.multipool as mp
    monkeypatch.setattr(mp, "_spawns_subprocess", lambda eng: getattr(eng, "_slow_marker", False))
    text = _pool("text", TEXT, [0]); text.engine._slow_marker = True
    img = _pool("image", IMAGE, [4])

    async def run():
        m = MultiPoolEngine([text, img])
        await m.start(TEXT)
        await m.pause(YieldUrgency.IMMEDIATE)
        orig = img.engine.resume

        async def resume_then_yield():
            await orig()
            m.request_pause()
        img.engine.resume = resume_then_yield
        await m.resume()
        assert text.engine.status().state is EngineState.PAUSED
    asyncio.run(run())


def test_one_pool_failing_to_resume_does_not_block_the_others(monkeypatch):
    import src.inference.multipool as mp
    monkeypatch.setattr(mp, "_spawns_subprocess", lambda eng: getattr(eng, "_slow_marker", False))
    text = _pool("text", TEXT, [0]); text.engine._slow_marker = True
    bad = _pool("tts", "/models/k", [6])
    img = _pool("image", IMAGE, [4])

    async def boom():
        raise RuntimeError("load failed")

    async def run():
        m = MultiPoolEngine([text, bad, img])
        await m.start(TEXT)
        await m.pause(YieldUrgency.IMMEDIATE)
        bad.engine.resume = boom
        with pytest.raises(RuntimeError, match="tts"):
            await m.resume()
        assert img.engine.status().state is EngineState.RUNNING
        assert text.engine.status().state is EngineState.RUNNING
    asyncio.run(run())


def test_vlm_image_pixel_cap_reaches_vllm(monkeypatch, tmp_path):
    """max_image_pixels must reach vLLM as mm_processor_kwargs.max_pixels for a VLM
    (vLLM profiles with one maximum-size image; a 4-bit 8B VL only fits a 10 GB card
    once that is bounded), and must be left out when unset or for a text model."""
    import sys, types, yaml
    from src.config import load_config
    from src.inference import engine as eng_mod
    from src.inference.engine_pool import create_engine_pool

    captured = []

    class FakeArgs:
        def __init__(self, **kw):
            captured.append(kw)

    class FakeLLM:
        @staticmethod
        def from_engine_args(args):
            return object()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(AsyncLLMEngine=FakeLLM))
    monkeypatch.setitem(sys.modules, "vllm.engine", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", types.SimpleNamespace(AsyncEngineArgs=FakeArgs))
    monkeypatch.setattr(eng_mod, "_config_is_vlm", lambda c: True)

    class Cfg:
        max_position_embeddings = 1 << 20
    fake_tf = types.SimpleNamespace(AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: Cfg()))
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    for pixels, expect in ((401408, {"max_pixels": 401408, "size": {"longest_edge": 401408, "shortest_edge": 65536}}),
                           (0, None)):
        captured.clear()
        p = tmp_path / f"c{pixels}.yaml"
        p.write_text(yaml.safe_dump({"inference": {"engine": "vllm", "model": "/models/vl",
                                                   "max_image_pixels": pixels,
                                                   "multi_gpu": {"device_ids": [0], "gpu_count": 1}}}))
        cfg = load_config(str(p))
        pool = create_engine_pool(cfg)
        e = pool._engines[0]
        e._model = "/models/vl"
        asyncio.run(e._create_engine_inner())
        assert captured, "engine args not built"
        assert captured[-1].get("mm_processor_kwargs") == expect, captured[-1]


def test_health_reports_per_model_request_limits(monkeypatch):
    """Two workers can serve one model with different limits (1 image / 4k on a 10 GB
    card, 32 / 8k on a 4-card pool). /health must report them per model so the gateway
    can route a request only where it fits: per pool on a multi-pool worker, top-level
    on a single-engine one; nothing for engines without such limits."""
    from src.inference.multipool import _engine_limits

    class VL:
        is_vlm = True
        max_images_per_prompt = 1
        context_limit = 4096

    class Text:
        is_vlm = False
        max_images_per_prompt = 0
        context_limit = 8192

    class Speech:
        is_vlm = False
        max_images_per_prompt = 0
        context_limit = 0

    assert _engine_limits(VL()) == {"max_images_per_prompt": 1, "context_limit": 4096}
    assert _engine_limits(Text()) == {"context_limit": 8192}
    assert _engine_limits(Speech()) == {}

    text = _pool("text", TEXT, [0])
    m = MultiPoolEngine([text, _pool("image", IMAGE, [4], image_gen=True)])
    pools = m.detailed_status()["pools"]
    assert all("limits" in p for p in pools)


def test_vllm_engine_context_limit_is_the_effective_length():
    from src.inference.engine import VLLMEngine
    e = VLLMEngine(device_ids=[0], max_model_len=8192)
    assert e.context_limit == 8192
    e._effective_max_model_len = 4096    # lowered to the model's own limit at load
    assert e.context_limit == 4096


def test_switches_in_different_pools_run_at_the_same_time():
    """A switch in one pool must not wait for a reload in another: on a test worker an ASR switch
    (5 s) waited 42 s behind a 32B -> VL reload of the text pool."""
    import threading
    import time as _time
    from fastapi.testclient import TestClient
    from src.inference.api_server import create_app

    TEXT2 = "/models/Qwen--Qwen3-4B-Instruct-2507"
    IMAGE2 = "/models/segmind--SSD-1B"
    m = MultiPoolEngine([
        _pool("text", TEXT, [0, 1], supported=[TEXT, TEXT2]),
        _pool("image", IMAGE, [4], image_gen=True, supported=[IMAGE, IMAGE2]),
    ])
    asyncio.run(m.start(TEXT))
    spans = {}
    for p in m.pools:
        e = p.engine._engines[0]
        orig = e.switch_model

        async def slow(path, _orig=orig, _name=p.name):
            t0 = _time.monotonic()
            await asyncio.sleep(0.6)
            await _orig(path)
            spans[_name] = (t0, _time.monotonic())
        e.switch_model = slow

    class _MM:
        async def ensure_model(self, model_id):
            return model_id

    client = TestClient(create_app(m, model_manager=_MM()))
    out = {}
    ths = [threading.Thread(target=lambda: out.setdefault("text", client.post(
               "/v1/chat/completions", json={"model": TEXT2, "messages": [{"role": "user", "content": "hi"}]}).status_code)),
           threading.Thread(target=lambda: out.setdefault("image", client.post(
               "/v1/images/generations", json={"model": IMAGE2, "prompt": "a cube", "n": 1}).status_code))]
    [t.start() for t in ths]
    [t.join(timeout=10) for t in ths]
    assert out == {"text": 200, "image": 200}, out
    (a0, a1), (b0, b1) = spans["text"], spans["image"]
    assert a0 < b1 and b0 < a1, f"the two pools' switches did not overlap: {spans}"

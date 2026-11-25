"""/v1/images/generations: the worker-side endpoint.

Exercised against MockEngine in image_gen mode, so the response shape, the per-request
cap and the receipt are all covered without diffusers or a GPU.
"""
import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from src.inference.api_server import create_app
from src.inference.engine import EngineState, MockEngine


def _client(max_images: int = 4):
    e = MockEngine(latency_sec=0.0, image_gen=True)
    e._max_images_per_request = max_images
    import asyncio
    asyncio.run(e.start("/models/fake-flux"))
    return TestClient(create_app(e)), e


def _body(n=1, **kw):
    b = {"model": "/models/fake-flux", "prompt": "a red cube", "n": n, "size": "512x512"}
    b.update(kw)
    return b


def test_generates_images_and_reports_usage():
    c, _ = _client()
    r = c.post("/v1/images/generations", json=_body(n=2))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["usage"]["images"] == 2
    assert len(body["data"]) == 2
    # Every payload is real, decodable PNG bytes.
    for item in body["data"]:
        raw = base64.b64decode(item["b64_json"])
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_rejects_more_images_than_cap_with_400():
    """Over-cap is a client mistake: a clean 400 at the edge, not an engine raise."""
    c, _ = _client(max_images=2)
    r = c.post("/v1/images/generations", json=_body(n=3))
    assert r.status_code == 400
    assert "too many images" in r.json()["detail"]


def test_rejects_a_model_that_does_not_generate_images():
    """A text worker must refuse this endpoint plainly, rather than failing obscurely
    deep inside an engine that has no image output."""
    import asyncio
    e = MockEngine(latency_sec=0.0, image_gen=False)
    asyncio.run(e.start("/models/fake-text"))
    c = TestClient(create_app(e))
    r = c.post("/v1/images/generations", json=_body())
    assert r.status_code == 400
    assert "does not generate images" in r.json()["detail"]


def test_503_when_yielded_to_mining():
    """A paused engine (GPU yielded) must answer 503 with Retry-After, exactly like the
    chat path, so the gateway queues rather than treating it as a failure."""
    c, e = _client()
    import asyncio
    asyncio.run(e.pause())
    r = c.post("/v1/images/generations", json=_body())
    assert r.status_code == 503
    assert "Retry-After" in r.headers


def test_503_when_a_yield_stops_the_render_midway():
    """A yield that lands mid-render returns nothing and bills nothing. The answer is
    503 (retry elsewhere) — the same as a yield before the render — not a 500."""
    c, e = _client()

    async def interrupted(*a, **kw):
        raise RuntimeError("Engine not available — GPU yielded to mining during generation")
    e.generate = interrupted
    r = c.post("/v1/images/generations", json=_body())
    assert r.status_code == 503, r.text
    assert "X-OM-Receipt" not in r.headers
    assert r.headers.get("Retry-After")


def test_receipt_carries_quantity_and_asset_hash():
    """The receipt must be the ASSET shape: quantity + asset_hash, and the hash must be
    the sha256 of the bytes actually returned — that is what the gateway re-derives and
    cross-checks before billing."""
    c, _ = _client()
    r = c.post("/v1/images/generations", json=_body(n=2))
    assert r.status_code == 200
    rcpt_hdr = r.headers.get("X-OM-Receipt")
    assert rcpt_hdr, "no receipt header"
    rcpt = json.loads(base64.b64decode(rcpt_hdr))

    raw = [base64.b64decode(i["b64_json"]) for i in r.json()["data"]]
    want_hash = hashlib.sha256(b"".join(raw)).hexdigest()

    assert rcpt["quantity"] == 2
    assert rcpt["asset_hash"] == want_hash
    assert rcpt["sig"], "receipt must be signed"
    # Token counts are absent/zero: an image generation produces no tokens, and
    # attesting a token count would be a fabricated measurement.
    assert rcpt.get("prompt_tokens", 0) == 0
    assert rcpt.get("completion_tokens", 0) == 0


def test_health_advertises_image_gen_not_vision():
    """The two capability flags route different traffic: a generator must advertise
    image_gen and NOT vision."""
    c, _ = _client()
    h = c.get("/health").json()
    assert "image_gen" in h["features"]
    assert "vision" not in h["features"]


def test_health_features_absent_for_a_text_worker():
    import asyncio
    e = MockEngine(latency_sec=0.0, image_gen=False)
    asyncio.run(e.start("/models/fake-text"))
    c = TestClient(create_app(e))
    h = c.get("/health").json()
    assert "image_gen" not in h["features"]
    assert "vision" not in h["features"]


@pytest.mark.parametrize("size,expect", [
    ("512x512", (512, 512)),
    ("1024x768", (1024, 768)),
    ("auto", (1024, 1024)),
    ("", (1024, 1024)),
    (None, (1024, 1024)),
    # Larger OpenAI sizes are scaled down to fit, keeping the aspect ratio.
    ("1792x1024", (1024, 584)),
    ("1024x1792", (584, 1024)),
    ("1536x1024", (1024, 680)),
    ("4096x4096", (1024, 1024)),
])
def test_size_parsing(size, expect):
    from src.inference.api_server import _parse_size
    assert _parse_size(size) == expect


@pytest.mark.parametrize("size", ["nonsense", "4100x4096", "100x100", "513x512", "512", "1024x200"])
def test_invalid_sizes_are_a_clean_400(size):
    """Found on a test worker: 4096x4096 ran the card out of memory (500) and "abc" silently
    rendered 1024x1024. Both are client mistakes: 400 with the accepted range."""
    c, _ = _client()
    r = c.post("/v1/images/generations", json=_body(size=size))
    assert r.status_code == 400, (size, r.status_code, r.text)
    assert "multiple of 8" in r.json()["detail"]


def test_empty_prompt_is_a_clean_400():
    c, _ = _client()
    for p in ("", "   "):
        r = c.post("/v1/images/generations", json=_body(prompt=p))
        assert r.status_code == 400 and "prompt" in r.json()["detail"], r.text


def test_out_of_memory_render_is_503_and_frees_the_cache():
    """An OOM mid-render must not be a 500 (the gateway does not retry those) and must
    release the card's cache, or the next renders fail as well."""
    import asyncio
    from src.inference.diffusion_engine import DiffusionEngine

    class OOMPipe:
        def __call__(self, *a, **kw):
            raise RuntimeError("CUDA out of memory. Tried to allocate 160.00 MiB")

    e = DiffusionEngine(device_ids=[0], max_images_per_request=2)

    async def _load():
        e._pipe = OOMPipe()
    e._load_pipeline = _load
    e._bind_device = lambda: None
    released = []
    e._release_cuda_cache = lambda: released.append(1)
    asyncio.run(e.start("/models/fake-sdxl"))
    c = TestClient(create_app(e))
    r = c.post("/v1/images/generations", json=_body(model="/models/fake-sdxl"))
    assert r.status_code == 503, r.text
    assert released == [1]


def test_steps_over_the_cap_are_a_clean_400():
    """Each denoising step holds the whole card; an uncapped request could hold it for
    as long as it asked."""
    c, e = _client()
    e.max_steps = 50
    r = c.post("/v1/images/generations", json=_body(steps=51))
    assert r.status_code == 400 and "at most 50" in r.json()["detail"], r.text
    assert c.post("/v1/images/generations", json=_body(steps=50)).status_code == 200


def test_diffusion_engine_and_pool_report_the_step_cap():
    from src.inference.diffusion_engine import DiffusionEngine
    from src.inference.engine_pool import EnginePool
    e = DiffusionEngine(device_ids=[0], max_steps=30)
    assert e.max_steps == 30 and EnginePool("tensor_parallel", [e]).max_steps == 30
    assert DiffusionEngine(device_ids=[0]).max_steps == 50


def test_config_reads_max_steps(tmp_path):
    from src.config import load_config
    p = tmp_path / "s.yaml"
    p.write_text("mode: dev\ninference:\n  engine: diffusion\n  model: /models/x\n  max_steps: 24\n")
    assert load_config(str(p)).inference.max_steps == 24
    p.write_text("mode: dev\ninference:\n  engine: diffusion\n  model: /models/x\n")
    assert load_config(str(p)).inference.max_steps == 50


def test_url_response_format_is_refused():
    c, _ = _client()
    r = c.post("/v1/images/generations", json=_body(response_format="url"))
    assert r.status_code == 400 and "b64_json" in r.json()["detail"], r.text

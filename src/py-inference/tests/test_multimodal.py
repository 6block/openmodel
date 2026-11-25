"""Unit tests for multimodal content parsing (image understanding).

These test the PURE parsing layer (multimodal.parse_content) — no GPU, no vLLM.
They pin the contract the API server relies on: text-only content is byte-
identical to before, image parts decode out-of-band, and bad input fails clean.
"""
import base64
import io

import pytest


def _png_bytes(color=(255, 0, 0), size=(2, 2)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _data_uri(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def test_text_content_untouched():
    from src.inference.multimodal import parse_content
    parsed = parse_content("hello")
    assert parsed.rendered == "hello"
    assert parsed.images == []
    assert parsed.has_image is False


def test_image_part_decodes_and_marks():
    from src.inference.multimodal import parse_content
    parsed = parse_content([
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": _data_uri(_png_bytes())}},
    ])
    assert parsed.has_image is True
    assert len(parsed.images) == 1
    assert parsed.rendered == [
        {"type": "text", "text": "what is this?"},
        {"type": "image"},
    ]


def test_image_part_loose_spelling():
    from src.inference.multimodal import parse_content
    parsed = parse_content([{"type": "image", "url": _data_uri(_png_bytes())}])
    assert parsed.has_image and len(parsed.images) == 1


def test_all_text_array_collapses_to_string():
    from src.inference.multimodal import parse_content
    parsed = parse_content([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ])
    assert parsed.rendered == "ab"
    assert parsed.has_image is False


def test_bad_data_uri_raises_valueerror():
    from src.inference.multimodal import parse_content
    with pytest.raises(ValueError):
        parse_content([{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}])


def test_missing_url_raises():
    from src.inference.multimodal import parse_content
    with pytest.raises(ValueError):
        parse_content([{"type": "image_url", "image_url": {}}])


def test_non_dict_parts_ignored():
    from src.inference.multimodal import parse_content
    parsed = parse_content(["just a string part"])
    assert parsed.has_image is False
    assert parsed.rendered == ""


def test_corrupt_image_bytes_raise():
    from src.inference.multimodal import parse_content
    bad = _data_uri(b"not a real png")
    with pytest.raises(ValueError):
        parse_content([{"type": "image_url", "image_url": {"url": bad}}])


def test_oversize_image_rejected(monkeypatch):
    from src.inference import multimodal
    monkeypatch.setattr(multimodal, "MAX_IMAGE_BYTES", 8)  # tiny cap
    from src.inference.multimodal import parse_content
    with pytest.raises(ValueError):
        parse_content([{"type": "image_url", "image_url": {"url": _data_uri(_png_bytes())}}])


# --- API-level: a multimodal request through /v1/chat/completions -------------
# The parsing layer above is pure; these pin the wiring: the decoded images reach
# the engine as gen_kwargs["images"], the rendered prompt carries the image marker
# (via the template) or falls back to the text-only join (no template), and the
# signed receipt still hashes request bytes + response TEXT — never pixels.

def _mm_body(text="what colour?", n_images=1):
    parts = [{"type": "text", "text": text}]
    for _ in range(n_images):
        parts.append({"type": "image_url", "image_url": {"url": _data_uri(_png_bytes())}})
    return {"messages": [{"role": "user", "content": parts}], "max_tokens": 8}


def _probe_engine():
    """MockEngine that records what generate()/generate_stream() were handed."""
    import asyncio
    from src.inference.engine import MockEngine, StreamChunk

    captured = {}

    class Probe(MockEngine):
        # This stand-in models a vision worker: it must SAY so, exactly as the real
        # VLLMEngine does. MockEngine is not a VLM by default (it reports is_vlm=False),
        # and these tests exercise the image-input path.
        @property
        def is_vlm(self):
            return True

        async def generate(self, prompt, **kw):
            captured["prompt"] = prompt
            captured["images"] = kw.get("images")
            return await super().generate(prompt, **kw)

        async def generate_stream(self, prompt, **kw):
            captured["prompt"] = prompt
            captured["images"] = kw.get("images")
            yield StreamChunk(text_delta="red")
            yield StreamChunk(text_delta="", finish_reason="stop", prompt_tokens=89, completion_tokens=1)

    e = Probe(latency_sec=0.0)
    asyncio.run(e.start("/models/fake-vlm"))
    return e, captured


def test_chat_completions_passes_images_to_engine(monkeypatch):
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app

    e, captured = _probe_engine()
    # No template loadable for the fake path → bare join; the marker path is pinned
    # by the template test below and by the real-GPU smoke.
    monkeypatch.setitem(api_server._tokenizers, "/models/fake-vlm", False)
    c = TestClient(create_app(e))

    r = c.post("/v1/chat/completions", json=_mm_body(n_images=2))
    assert r.status_code == 200, r.text
    assert captured["images"] is not None and len(captured["images"]) == 2
    from PIL import Image
    assert all(isinstance(i, Image.Image) for i in captured["images"])
    # Bare join uses the TEXT of the content array, not its repr.
    assert captured["prompt"] == "user: what colour?"
    assert "image_url" not in captured["prompt"] and "base64" not in captured["prompt"]


def test_chat_completions_text_only_has_no_images_kwarg(monkeypatch):
    """Text-only requests must not grow an images kwarg (engine path unchanged)."""
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app

    e, captured = _probe_engine()
    monkeypatch.setitem(api_server._tokenizers, "/models/fake-vlm", False)
    c = TestClient(create_app(e))
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert r.status_code == 200
    assert captured["images"] is None
    assert captured["prompt"] == "user: hi"


def test_chat_completions_template_renders_image_marker(monkeypatch):
    """With a chat template, image parts reach it as {"type":"image"} markers and
    text parts as {"type":"text"} — the shape Qwen-VL templates expand into their
    vision placeholder tokens."""
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app

    class Tok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            content = messages[0]["content"]
            assert isinstance(content, list)
            assert content[0] == {"type": "text", "text": "what colour?"}
            assert content[1] == {"type": "image"}
            return "TPL<image>what colour?<|assistant|>"

    e, captured = _probe_engine()
    monkeypatch.setitem(api_server._tokenizers, "/models/fake-vlm", Tok())
    c = TestClient(create_app(e))
    r = c.post("/v1/chat/completions", json=_mm_body())
    assert r.status_code == 200, r.text
    assert captured["prompt"] == "TPL<image>what colour?<|assistant|>"
    assert len(captured["images"]) == 1


def test_chat_completions_streaming_with_image(monkeypatch):
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app
    import json as _json

    e, captured = _probe_engine()
    monkeypatch.setitem(api_server._tokenizers, "/models/fake-vlm", False)
    c = TestClient(create_app(e))
    body = _mm_body()
    body["stream"] = True
    with c.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    frames = [_json.loads(x) for x in lines[:-1]]
    text = "".join(f["choices"][0]["delta"].get("content", "") for f in frames if f.get("choices"))
    assert text == "red"
    assert len(captured["images"]) == 1
    usage = [f["usage"] for f in frames if "usage" in f][0]
    assert usage["prompt_tokens"] == 89  # visual tokens are part of the billable prompt


def test_receipt_hashes_request_bytes_and_response_text_not_pixels(monkeypatch, tmp_path):
    """The signed receipt attests sha256(exact request body) and sha256(response
    text) — identical semantics to text-only requests. Pixels never enter it."""
    import base64 as _b64
    import hashlib
    import json as _json
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app
    import src.inference.receipt as receipt_mod

    monkeypatch.setenv("WORKER_RECEIPT_KEY", str(tmp_path / "k.key"))
    receipt_mod._signer = None
    e, _ = _probe_engine()
    monkeypatch.setitem(api_server._tokenizers, "/models/fake-vlm", False)
    c = TestClient(create_app(e))

    raw = _json.dumps(_mm_body()).encode()
    r = c.post("/v1/chat/completions", content=raw,
               headers={"Content-Type": "application/json", "X-Request-ID": "req-vlm"})
    assert r.status_code == 200
    rcpt = _json.loads(_b64.b64decode(r.headers["X-OM-Receipt"]))
    assert rcpt["request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert rcpt["response_sha256"] == hashlib.sha256(
        r.json()["choices"][0]["message"]["content"].encode()).hexdigest()


def test_bad_image_in_request_is_a_400_not_500(monkeypatch):
    """A corrupt/oversized image is a client error: the request must fail clean
    (4xx) before touching the engine, not surface as an unhandled 500."""
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app

    e, captured = _probe_engine()
    monkeypatch.setitem(api_server._tokenizers, "/models/fake-vlm", False)
    c = TestClient(create_app(e))
    bad = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}], "max_tokens": 8}
    r = c.post("/v1/chat/completions", json=bad)
    assert r.status_code == 400, r.text
    assert "image" in r.json()["detail"].lower()
    assert "images" not in captured, "engine must not be reached with a bad image"


# --- VLM detection, per-request image cap, capability advertisement ------------

def test_config_is_vlm_detection():
    from types import SimpleNamespace
    from src.inference.engine import _config_is_vlm
    assert _config_is_vlm(SimpleNamespace(vision_config=object(), architectures=["Qwen3VLForConditionalGeneration"]))
    assert _config_is_vlm(SimpleNamespace(vision_config=None, architectures=["Qwen2_5_VLForConditionalGeneration"]))
    assert not _config_is_vlm(SimpleNamespace(vision_config=None, architectures=["Qwen3ForCausalLM"]))
    assert not _config_is_vlm(SimpleNamespace(architectures=None))


def test_config_reads_max_images_per_prompt(tmp_path):
    from src.config import load_config
    cfg = tmp_path / "s.yaml"
    cfg.write_text("mode: dev\ninference:\n  engine: mock\n  max_images_per_prompt: 3\n")
    assert load_config(str(cfg)).inference.max_images_per_prompt == 3
    cfg.write_text("mode: dev\ninference:\n  engine: mock\n")
    assert load_config(str(cfg)).inference.max_images_per_prompt == 1


def test_health_advertises_vision_only_for_vlm():
    import asyncio
    from fastapi.testclient import TestClient
    from src.inference.api_server import create_app
    from src.inference.engine import MockEngine

    class VLM(MockEngine):
        is_vlm = True

    class Text(MockEngine):
        is_vlm = False

    for cls, expect in ((VLM, True), (Text, False)):
        e = cls(latency_sec=0.0)
        asyncio.run(e.start("/models/x"))
        feats = TestClient(create_app(e)).get("/health").json()["features"]
        assert ("vision" in feats) is expect, (cls.__name__, feats)
        assert "continuation" in feats and "receipt" in feats


def test_image_request_to_text_model_is_400(monkeypatch):
    """A text-only engine (is_vlm=False) must refuse image parts up front, not let
    vLLM blow up inside generate()."""
    import asyncio
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app
    from src.inference.engine import MockEngine

    hit = {"n": 0}

    class Text(MockEngine):
        is_vlm = False

        async def generate(self, prompt, **kw):
            hit["n"] += 1
            return await super().generate(prompt, **kw)

    e = Text(latency_sec=0.0)
    asyncio.run(e.start("/models/text-only"))
    monkeypatch.setitem(api_server._tokenizers, "/models/text-only", False)
    c = TestClient(create_app(e))
    r = c.post("/v1/chat/completions", json=_mm_body())
    assert r.status_code == 400, r.text
    assert "does not accept image input" in r.json()["detail"]
    assert hit["n"] == 0
    # …while plain text on the same engine still works.
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    assert r.status_code == 200 and hit["n"] == 1


def test_continuation_with_image_keeps_images_and_appends_prefix(monkeypatch):
    """Stream resume (om_continuation) on a vision request: the gateway re-sends the
    ORIGINAL multimodal body plus the delivered prefix. The worker must still decode
    the image AND append the prefix verbatim to the rendered prompt."""
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app

    e, captured = _probe_engine()
    monkeypatch.setitem(api_server._tokenizers, "/models/fake-vlm", False)
    c = TestClient(create_app(e))
    body = _mm_body(text="describe")
    body["om_continuation"] = "It is a red"
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    assert captured["prompt"] == "user: describeIt is a red"
    assert len(captured["images"]) == 1


def test_too_many_images_is_400_at_the_edge(monkeypatch):
    """The per-request image cap is enforced BEFORE generate(): vLLM's own check
    surfaces as an unhandled 500 (observed live), so the API must refuse first."""
    import asyncio
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app
    from src.inference.engine import MockEngine

    hit = {"n": 0}

    class VLM(MockEngine):
        is_vlm = True
        max_images_per_prompt = 1

        async def generate(self, prompt, **kw):
            hit["n"] += 1
            return await super().generate(prompt, **kw)

    e = VLM(latency_sec=0.0)
    asyncio.run(e.start("/models/vlm"))
    monkeypatch.setitem(api_server._tokenizers, "/models/vlm", False)
    c = TestClient(create_app(e))
    r = c.post("/v1/chat/completions", json=_mm_body(n_images=2))
    assert r.status_code == 400, r.text
    assert "too many images" in r.json()["detail"] and "at most 1" in r.json()["detail"]
    assert hit["n"] == 0
    r = c.post("/v1/chat/completions", json=_mm_body(n_images=1))
    assert r.status_code == 200 and hit["n"] == 1


def test_vllm_image_limit_valueerror_maps_to_400(monkeypatch):
    """Backstop: if the engine itself raises vLLM's limit error, it is a 400, not 500."""
    import asyncio
    from fastapi.testclient import TestClient
    from src.inference import api_server
    from src.inference.api_server import create_app
    from src.inference.engine import MockEngine

    class VLM(MockEngine):
        is_vlm = True
        max_images_per_prompt = 0  # cap unknown at the edge → engine decides

        async def generate(self, prompt, **kw):
            raise ValueError("At most 1 image(s) may be provided in one prompt.")

    e = VLM(latency_sec=0.0)
    asyncio.run(e.start("/models/vlm"))
    monkeypatch.setitem(api_server._tokenizers, "/models/vlm", False)
    c = TestClient(create_app(e))
    r = c.post("/v1/chat/completions", json=_mm_body(n_images=2))
    assert r.status_code == 400, r.text
    assert "too many images" in r.json()["detail"]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "/etc/passwd", "relative/img.png",
                                 "ftp://example.com/a.png", "FILE:///etc/hosts", "gopher://x"])
def test_image_url_refuses_anything_but_data_and_http(url, tmp_path):
    """The URL comes from the client: a file:// URL or a bare path would make the worker
    read its own filesystem on the client's behalf."""
    from src.inference.multimodal import load_image_url
    with pytest.raises(ValueError, match="data: URI or an http"):
        load_image_url(url)


def test_image_url_reading_a_real_local_file_is_refused(tmp_path):
    from src.inference.multimodal import load_image_url
    p = tmp_path / "secret.png"
    p.write_bytes(_png_bytes())
    for url in (str(p), "file://" + str(p)):
        with pytest.raises(ValueError):
            load_image_url(url)


def test_file_url_in_a_chat_request_is_a_400(monkeypatch):
    from fastapi.testclient import TestClient
    from src.inference.api_server import create_app
    from src.inference.engine import MockEngine
    import asyncio
    e = MockEngine(latency_sec=0.0)
    e._is_vlm = True
    asyncio.run(e.start("/models/vl"))
    monkeypatch.setattr(type(e), "is_vlm", property(lambda self: True), raising=False)
    r = TestClient(create_app(e)).post("/v1/chat/completions", json={"model": "/models/vl", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"},
                                     {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]}]})
    assert r.status_code == 400, r.text

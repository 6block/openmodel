"""Unit tests for the OpenAI-compatible API server: non-streaming body, SSE
streaming, /v1/completions, and /v1/models — all against MockEngine (no GPU)."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from src.inference.api_server import create_app
from src.inference.engine import MockEngine


@pytest.fixture
def client():
    e = MockEngine(latency_sec=0.0)
    asyncio.run(e.start("test-model"))
    return TestClient(create_app(e))  # auth disabled


def test_chat_completion_nonstreaming_body(client):
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    assert "usage" in body
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_chat_completion_streaming(client):
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 8, "stream": True}) as r:
        assert r.status_code == 200
        data_lines = [ln[6:] for ln in r.iter_lines() if ln and ln.startswith("data: ")]

    assert data_lines, "expected SSE data lines"
    assert data_lines[-1] == "[DONE]"
    objs = [json.loads(c) for c in data_lines if c != "[DONE]"]
    assert any(o.get("object") == "chat.completion.chunk" for o in objs)


def test_completions_endpoint(client):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert "choices" in body
    assert body["choices"][0].get("finish_reason") in ("stop", "length")
    assert "usage" in body


def test_models_endpoint_lists_loaded_model(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "test-model" in ids


# --- B2 stream resume: om_continuation + feature advertisement -----------------

def test_om_continuation_appended_to_prompt():
    """The gateway resumes an interrupted stream by re-issuing the request with
    om_continuation = the already-delivered text; the worker must append it VERBATIM
    to the rendered prompt (no separator) so generation continues mid-sentence."""
    captured = {}

    class Probe(MockEngine):
        async def generate(self, prompt, **kw):
            captured["prompt"] = prompt
            return await super().generate(prompt, **kw)

    e = Probe(latency_sec=0.0)
    asyncio.run(e.start("m"))
    c = TestClient(create_app(e))

    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 8, "om_continuation": "Once upon"})
    assert r.status_code == 200
    assert captured["prompt"].startswith("user: hi")
    assert captured["prompt"].endswith("Once upon"), \
        "delivered prefix must be appended verbatim to the rendered prompt"

    r = c.post("/v1/completions",
               json={"prompt": "Story:", "max_tokens": 8, "om_continuation": " a time"})
    assert r.status_code == 200
    assert captured["prompt"] == "Story: a time"


def test_health_advertises_continuation_feature(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "continuation" in r.json().get("features", []), \
        "gateway relies on this to avoid resuming onto old workers"


# --- A1 signed receipts -------------------------------------------------------

def test_receipt_header_on_nonstreaming(tmp_path, monkeypatch):
    """Non-streaming responses carry X-OM-Receipt: a base64 JSON receipt whose ed25519
    signature verifies against the pubkey advertised on /health, attesting request
    hash + response hash + token counts."""
    import base64, hashlib
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    import src.inference.receipt as receipt_mod
    monkeypatch.setenv("WORKER_RECEIPT_KEY", str(tmp_path / "k.key"))
    receipt_mod._signer = None  # force re-init with tmp key

    e = MockEngine(latency_sec=0.0)
    asyncio.run(e.start("m"))
    c = TestClient(create_app(e))

    body = b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}'
    r = c.post("/v1/chat/completions", content=body,
               headers={"Content-Type": "application/json", "X-Request-ID": "req-abc"})
    assert r.status_code == 200
    rcpt = json.loads(base64.b64decode(r.headers["X-OM-Receipt"]))
    assert rcpt["request_id"] == "req-abc"
    assert rcpt["request_sha256"] == hashlib.sha256(body).hexdigest()
    assert rcpt["response_sha256"] == hashlib.sha256(
        r.json()["choices"][0]["message"]["content"].encode()).hexdigest()
    # signature verifies against the advertised pubkey over the canonical payload
    health = c.get("/health").json()
    assert rcpt["pubkey"] == health["receipt_pubkey"]
    assert "receipt" in health["features"]
    payload = receipt_mod.canonical_payload(
        rcpt["request_id"], rcpt["model"], rcpt["request_sha256"], rcpt["response_sha256"],
        rcpt["prompt_tokens"], rcpt["completion_tokens"], rcpt["cached_tokens"],
        rcpt["ts"], rcpt["pubkey"])
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(rcpt["pubkey"])).verify(
        bytes.fromhex(rcpt["sig"]), payload)  # raises on failure


def test_receipt_stream_event_only_when_requested(tmp_path, monkeypatch):
    """Streaming: the om_receipt event appears right before [DONE] ONLY when the
    gateway asks (X-OM-Receipt-Req: 1) — plain clients never see it."""
    import src.inference.receipt as receipt_mod
    monkeypatch.setenv("WORKER_RECEIPT_KEY", str(tmp_path / "k.key"))
    receipt_mod._signer = None

    e = MockEngine(latency_sec=0.0)
    asyncio.run(e.start("m"))
    c = TestClient(create_app(e))
    req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "stream": True}

    with c.stream("POST", "/v1/chat/completions", json=req,
                  headers={"X-OM-Receipt-Req": "1", "X-Request-ID": "req-s1"}) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    payloads = [ln[6:] for ln in lines]
    rcpt_lines = [p for p in payloads if "om_receipt" in p]
    assert len(rcpt_lines) == 1, "exactly one receipt event when requested"
    assert payloads.index(rcpt_lines[0]) == len(payloads) - 2, "receipt sits right before [DONE]"
    rcpt = json.loads(rcpt_lines[0])["om_receipt"]
    assert rcpt["request_id"] == "req-s1" and rcpt["sig"]

    with c.stream("POST", "/v1/chat/completions", json=req) as r:
        lines = [ln for ln in r.iter_lines() if "om_receipt" in ln]
    assert lines == [], "no receipt event without the gateway header"

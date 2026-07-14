"""P6 regression: /v1/* endpoints honor an optional bearer token, reject
invalid max_tokens, and enforce a max_tokens cap; /health stays open."""
import pytest
from fastapi.testclient import TestClient

from src.inference.api_server import create_app
from src.inference.engine import MockEngine


@pytest.fixture
def engine():
    import asyncio
    e = MockEngine(latency_sec=0.0)
    asyncio.run(e.start("test-model"))
    return e


def test_health_open_without_token(engine):
    app = create_app(engine, api_token="secret")
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_v1_requires_token_when_configured(engine):
    app = create_app(engine, api_token="secret")
    client = TestClient(app)

    # Missing token → 401
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401

    # Wrong token → 401
    r = client.post("/v1/chat/completions",
                    headers={"Authorization": "Bearer wrong"},
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401

    # Correct token → 200
    r = client.post("/v1/chat/completions",
                    headers={"Authorization": "Bearer secret"},
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_no_token_means_open(engine):
    """Backward compat: no api_token → endpoints work without auth."""
    app = create_app(engine, api_token="")
    client = TestClient(app)
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_invalid_max_tokens_rejected(engine):
    app = create_app(engine, api_token="")
    client = TestClient(app)
    # max_tokens <= 0 is rejected by Pydantic (422)
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0})
    assert r.status_code == 422
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": -5})
    assert r.status_code == 422


def test_max_tokens_cap_enforced(engine):
    app = create_app(engine, api_token="", max_tokens_limit=100)
    client = TestClient(app)
    # Over the cap → 400
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 500})
    assert r.status_code == 400
    # Within the cap → 200
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50})
    assert r.status_code == 200

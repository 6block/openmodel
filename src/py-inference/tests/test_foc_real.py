"""Unit tests for RealFOCClient — registry normalization, check_availability
status matrix, and resolve_model — using an httpx MockTransport (no network)."""
import asyncio

import httpx
import pytest

from src.model.foc_real import RealFOCClient


def make(registry, handler=None):
    c = RealFOCClient("http://bridge/", registry)
    if handler is not None:
        c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def test_registry_normalizes_str_and_list():
    c = RealFOCClient("http://bridge/", {"a": "cid1", "b": ["c1", "c2"]})
    assert c._get_cids("a") == ["cid1"]   # str → single-element list
    assert c._get_cids("b") == ["c1", "c2"]
    assert c._get_cids("missing") is None


def test_check_availability_no_cids():
    assert asyncio.run(make({}).check_availability("unknown")) is False


def test_check_availability_200_available():
    c = make({"m": ["cid1"]}, lambda req: httpx.Response(200, json={"available": True}))
    assert asyncio.run(c.check_availability("m")) is True


def test_check_availability_200_not_available():
    c = make({"m": ["cid1"]}, lambda req: httpx.Response(200, json={"available": False}))
    assert asyncio.run(c.check_availability("m")) is False


def test_check_availability_404():
    c = make({"m": ["cid1"]}, lambda req: httpx.Response(404, json={}))
    assert asyncio.run(c.check_availability("m")) is False


def test_check_availability_other_status():
    c = make({"m": ["cid1"]}, lambda req: httpx.Response(500, json={}))
    assert asyncio.run(c.check_availability("m")) is False


def test_check_availability_connect_error():
    def boom(req):
        raise httpx.ConnectError("connection refused")
    c = make({"m": ["cid1"]}, boom)
    assert asyncio.run(c.check_availability("m")) is False


def test_resolve_model_no_cids_raises():
    with pytest.raises(ValueError):
        asyncio.run(make({}).resolve_model("unknown"))


def test_resolve_model_builds_manifest():
    manifest = asyncio.run(make({"m": ["c1", "c2", "c3"]}).resolve_model("m"))
    assert manifest.model_id == "m"
    assert manifest.source_uri == "c1,c2,c3"  # all CIDs joined


# --- A2 weight-integrity: sha256 pinning plumbed to the bridge -----------------

def test_registry_dict_entry_carries_sha256():
    c = RealFOCClient("http://bridge/", {
        "m": {"cids": ["c1", "c2"], "sha256": "AB" * 32},
    })
    assert c._get_cids("m") == ["c1", "c2"]
    assert c._sha256["m"] == ("ab" * 32)  # lowercased


def test_resolve_model_carries_pinned_digest():
    c = make({"m": {"cids": ["c1"], "sha256": "cd" * 32}})
    manifest = asyncio.run(c.resolve_model("m"))
    assert manifest.checksum_sha256 == "cd" * 32


def test_download_forwards_sha256_and_maps_422(tmp_path):
    seen = {}

    def handler(req):
        if req.url.path == "/download-model":
            seen["body"] = req.read().decode()
            return httpx.Response(422, json={
                "error": "sha256 mismatch: downloaded weights do NOT match the pinned digest (file deleted)",
                "expected": "ee" * 32, "computed": "ff" * 32})
        return httpx.Response(404)

    c = make({"m": {"cids": ["c1"], "sha256": "ee" * 32}}, handler)
    manifest = asyncio.run(c.resolve_model("m"))
    with pytest.raises(RuntimeError, match="weight integrity check failed"):
        asyncio.run(c.download_weights(manifest, tmp_path / "m"))
    assert ("ee" * 32) in seen["body"], "pinned digest must be forwarded to the bridge"


def test_download_unpinned_omits_sha256(tmp_path):
    seen = {}

    def handler(req):
        if req.url.path == "/download-model":
            seen["body"] = req.read().decode()
            # Fail after capture (avoids needing a real tarball to extract).
            return httpx.Response(500, json={"error": "stop"})
        return httpx.Response(404)

    c = make({"m": ["c1"]}, handler)  # no sha256 pinned
    manifest = asyncio.run(c.resolve_model("m"))
    with pytest.raises(Exception):
        asyncio.run(c.download_weights(manifest, tmp_path / "m"))
    assert '"sha256"' not in seen["body"], "unpinned model must not send an sha256 field"

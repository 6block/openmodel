"""Prefetch-on-claim: claimed models are fetched at startup, off the request path.

The serving-path download happens inside a request-triggered switch, where the
triggering request times out long before a multi-minute download finishes.
These tests pin the prefetch resolution order (local → FOC → HF), the
"present" short-circuits, the per-model lock that stops a prefetch and a
switch from downloading the same weights twice, and the config plumbing that
feeds gateway.supported_models to the inference side.
"""
import asyncio
import json

import pytest

from src.config import load_config
from src.model.foc_client import FOCClient, ModelManifest
from src.model.manager import ModelManager


def write_complete_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_text("weights")


class ScriptedFOC(FOCClient):
    """FOC fake: serves the models in `available`, records download calls."""

    def __init__(self, available=(), delay=0.0):
        self.available = set(available)
        self.downloads = []
        self.delay = delay

    async def check_availability(self, model_id):
        return model_id in self.available

    async def resolve_model(self, model_id):
        return ModelManifest(model_id=model_id, size_bytes=0, checksum_sha256="",
                             source_uri="cid", quantization=None)

    async def download_weights(self, manifest, dest):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.downloads.append(manifest.model_id)
        write_complete_model(dest)
        return dest


@pytest.fixture()
def cache(tmp_path):
    return tmp_path / "models"


def test_prefetch_present_local_path(cache):
    write_complete_model(cache / "Org--M")
    mgr = ModelManager(ScriptedFOC(), cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch(f"{cache}/Org--M")) == "present"
    assert asyncio.run(mgr.prefetch("Org/M")) == "present"  # hub-form maps to the same dir


def test_prefetch_downloads_from_foc(cache):
    foc = ScriptedFOC(available={"Org/M"})
    mgr = ModelManager(foc, cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch("Org/M")) == "foc-downloaded"
    assert foc.downloads == ["Org/M"]
    # Second call: weights are on disk now — no second download.
    assert asyncio.run(mgr.prefetch("Org/M")) == "present"
    assert foc.downloads == ["Org/M"]


def test_prefetch_hf_form_falls_back_to_snapshot(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(ModelManager, "_hf_snapshot",
                        staticmethod(lambda model_id: calls.append(model_id)))
    mgr = ModelManager(ScriptedFOC(), cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch("Org/M")) == "hf-ensured"
    assert calls == ["Org/M"]


def test_prefetch_path_form_falls_back_to_hf(cache, monkeypatch):
    """FOC has nothing (or its only provider is down): a derivable path-form
    claim falls back to HF via the canonical basename transform, into the
    canonical dir. Live motivation: the single mainnet provider went dark and
    bricked every first boot."""
    fetched = []
    def fake_fetch(hf_id, dest):
        fetched.append(hf_id)
        write_complete_model(dest)
    monkeypatch.setattr(ModelManager, "_hf_fetch_dir", staticmethod(fake_fetch))
    mgr = ModelManager(ScriptedFOC(), cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch("/models/Org--M")) == "hf-downloaded"
    assert fetched == ["Org/M"]
    assert (cache / "Org--M" / "config.json").exists()


def test_prefetch_foc_error_falls_back_to_hf(cache, monkeypatch):
    """A registered model whose FOC download RAISES (provider outage) must fall
    back, not die."""
    class DeadFOC(ScriptedFOC):
        async def download_weights(self, manifest, dest):
            raise RuntimeError("provider TLS handshake failed")
    fetched = []
    def fake_fetch(hf_id, dest):
        fetched.append(hf_id)
        write_complete_model(dest)
    monkeypatch.setattr(ModelManager, "_hf_fetch_dir", staticmethod(fake_fetch))
    mgr = ModelManager(DeadFOC(available={"/models/Org--M"}), cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch("/models/Org--M")) == "hf-downloaded"
    assert fetched == ["Org/M"]


def test_prefetch_underivable_path_stays_unavailable(cache, monkeypatch):
    # No "--" in the basename → no HF derivation → honest "unavailable".
    monkeypatch.setattr(ModelManager, "_hf_fetch_dir",
                        staticmethod(lambda *_: pytest.fail("HF must not be called")))
    mgr = ModelManager(ScriptedFOC(), cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch("/models/localonly")) == "unavailable"


def test_prefetch_hf_fallback_kill_switch(cache, monkeypatch):
    monkeypatch.setenv("HF_FALLBACK", "0")
    monkeypatch.setattr(ModelManager, "_hf_fetch_dir",
                        staticmethod(lambda *_: pytest.fail("fallback disabled")))
    mgr = ModelManager(ScriptedFOC(), cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch("/models/Org--M")) == "unavailable"


def test_ensure_model_foc_error_falls_back_to_hf(cache, monkeypatch):
    """The boot/switch path: FOC provider down → HF fallback fetches into the
    canonical dir and the engine gets a loadable path (previously: crash loop)."""
    class DeadFOC(ScriptedFOC):
        async def download_weights(self, manifest, dest):
            raise RuntimeError("provider TLS handshake failed")
    def fake_fetch(hf_id, dest):
        write_complete_model(dest)
    monkeypatch.setattr(ModelManager, "_hf_fetch_dir", staticmethod(fake_fetch))
    mgr = ModelManager(DeadFOC(available={"/models/Org--M"}), cache_dir=str(cache))
    path = asyncio.run(mgr.ensure_model("/models/Org--M"))
    assert path.endswith("Org--M")
    from pathlib import Path as P
    assert (P(path) / "config.json").exists()


def test_prefetch_replaces_partial_cache(cache):
    partial = cache / "Org--M"
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")  # config but no weights = incomplete
    foc = ScriptedFOC(available={"Org/M"})
    mgr = ModelManager(foc, cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch("Org/M")) == "foc-downloaded"
    assert (partial / "model.safetensors").exists()


def test_switch_waits_for_inflight_prefetch_single_download(cache):
    """ensure_model for a model mid-prefetch waits on the lock and takes the
    cache hit — exactly one download for the pair."""
    foc = ScriptedFOC(available={"Org/M"}, delay=0.05)
    mgr = ModelManager(foc, cache_dir=str(cache))

    async def run_pair():
        pre = asyncio.create_task(mgr.prefetch("Org/M"))
        await asyncio.sleep(0.01)  # prefetch is inside download_weights now
        path = await mgr.ensure_model("Org/M")
        return await pre, path

    pre_result, path = asyncio.run(run_pair())
    assert pre_result == "foc-downloaded"
    assert path.endswith("Org--M")
    assert foc.downloads == ["Org/M"]  # not twice


def test_config_parses_supported_models(tmp_path):
    cfg_file = tmp_path / "sidecar.yaml"
    cfg_file.write_text(json.dumps({
        "inference": {"model": "Org/M"},
        "gateway": {"supported_models": ["/models/Org--M", "Org/N", ""]},
    }))
    cfg = load_config(cfg_file)
    assert cfg.gateway.supported_models == ["/models/Org--M", "Org/N"]


def test_config_supported_models_defaults_empty(tmp_path):
    cfg_file = tmp_path / "sidecar.yaml"
    cfg_file.write_text("inference:\n  model: Org/M\n")
    cfg = load_config(cfg_file)
    assert cfg.gateway.supported_models == []


class HalfWriteFOC(ScriptedFOC):
    """Writes config.json, stalls mid-extract, then writes the weights —
    reproducing the window where the cache dir looks plausible but is broken."""

    async def download_weights(self, manifest, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}")
        (dest / "model.safetensors").write_text("PARTIAL")  # half-written weight
        await asyncio.sleep(0.05)
        (dest / "model.safetensors").write_text("weights-final")
        self.downloads.append(manifest.model_id)
        return dest


def test_switch_never_sees_half_extracted_dir(cache):
    """While a prefetch is mid-extract, ensure_model must queue on the lock and
    return the FINISHED copy — never the plausible-looking partial one."""
    foc = HalfWriteFOC(available={"Org/M"})
    mgr = ModelManager(foc, cache_dir=str(cache))

    async def run_pair():
        pre = asyncio.create_task(mgr.prefetch("Org/M"))
        await asyncio.sleep(0.01)  # inside the half-written window
        path = await mgr.ensure_model("Org/M")
        await pre
        return path

    path = asyncio.run(run_pair())
    from pathlib import Path
    assert (Path(path) / "model.safetensors").read_text() == "weights-final"
    assert foc.downloads == ["Org/M"]


def test_registry_key_tolerates_both_spellings():
    """A path-form claim must find CIDs registered under the HF id and vice
    versa — exact-only lookup made path-form claims invisible to FOC."""
    from src.model.foc_real import RealFOCClient
    c = RealFOCClient("http://127.0.0.1:1", {
        "Qwen/Qwen3-4B-Instruct-2507": {"cids": ["cidA"], "sha256": "a" * 64},
        "/models/Org--PathKeyed": ["cidB"],
    })
    # path-form claim → HF-keyed entry
    assert c._get_cids("/models/Qwen--Qwen3-4B-Instruct-2507") == ["cidA"]
    # HF-form claim → path-keyed entry
    assert c._get_cids("Org/PathKeyed") == ["cidB"]
    # exact forms still work
    assert c._get_cids("Qwen/Qwen3-4B-Instruct-2507") == ["cidA"]
    assert c._get_cids("/models/Org--PathKeyed") == ["cidB"]
    # unknown stays unknown
    assert c._get_cids("/models/Org--Absent") is None
    # sha256 resolves through the same tolerance
    assert c._registry_key("/models/Qwen--Qwen3-4B-Instruct-2507") == "Qwen/Qwen3-4B-Instruct-2507"


def test_path_form_foc_download_lands_in_basename_dir(cache):
    """A path-form claim's FOC download must extract to cache/<basename> —
    where the path-first lookup and every operator convention expect it. Found
    live: it extracted to a '--models--Org--Name' mangled directory instead
    (consistent on both sides, so it served — but the same model fetched later
    under its HF spelling would land in a second directory, doubling disk)."""
    claim = "/models/Org--M"
    foc = ScriptedFOC(available={claim})
    mgr = ModelManager(foc, cache_dir=str(cache))
    assert asyncio.run(mgr.prefetch(claim)) == "foc-downloaded"
    assert (cache / "Org--M" / "config.json").exists()
    assert not (cache / "--models--Org--M").exists()
    # Both spellings now resolve to the SAME directory — no double fetch.
    assert asyncio.run(mgr.prefetch(claim)) == "present"
    assert asyncio.run(mgr.prefetch("Org/M")) == "present"
    path = asyncio.run(mgr.ensure_model(claim))
    assert path.endswith("Org--M")
    assert foc.downloads == [claim]

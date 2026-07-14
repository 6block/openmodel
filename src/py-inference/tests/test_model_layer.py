"""Unit tests for the model layer (ModelManager, find_local_model) — no GPU."""
import asyncio

from src.model.manager import ModelManager
from src.model.local_loader import find_local_model


class FakeFOC:
    def __init__(self, available=False):
        self._available = available

    async def check_availability(self, model_id):
        return self._available

    async def resolve_model(self, model_id):
        return {"model": model_id}

    async def download_weights(self, manifest, dest):
        dest.mkdir(parents=True, exist_ok=True)
        return dest


def test_ensure_model_cache_hit(tmp_path):
    d = tmp_path / "org--m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    mgr = ModelManager(FakeFOC(), cache_dir=str(tmp_path))
    path = asyncio.run(mgr.ensure_model("org/m"))
    assert path == str(d)
    assert mgr.loaded_model == "org/m"


def test_ensure_model_incomplete_cache_is_refetched(tmp_path):
    # A half-written cache dir (config but no weights) must NOT be a cache hit — it
    # is removed and the model re-resolved, instead of returning an unloadable dir
    # that would make vLLM fail forever (audit MEDIUM fix).
    d = tmp_path / "org--m"
    d.mkdir()
    (d / "config.json").write_text("{}")  # config present but NO weight file
    mgr = ModelManager(FakeFOC(available=False), cache_dir=str(tmp_path))
    path = asyncio.run(mgr.ensure_model("org/m"))
    assert path == "org/m"   # fell through to the HF id, not the partial dir
    assert not d.exists()    # the incomplete dir was removed


def test_ensure_model_foc_download(tmp_path):
    mgr = ModelManager(FakeFOC(available=True), cache_dir=str(tmp_path))
    path = asyncio.run(mgr.ensure_model("org/m"))
    assert "org--m" in path           # downloaded into the cache path
    assert mgr.loaded_model == "org/m"


def test_ensure_model_hf_fallback(tmp_path):
    mgr = ModelManager(FakeFOC(available=False), cache_dir=str(tmp_path))
    path = asyncio.run(mgr.ensure_model("org/m"))
    assert path == "org/m"            # not in FOC → used as a direct HF id
    assert mgr.loaded_model == "org/m"


def test_find_local_model_direct_path(tmp_path):
    (tmp_path / "model").mkdir()
    got = find_local_model(str(tmp_path / "model"))
    assert got == str((tmp_path / "model").resolve())


def test_find_local_model_in_search_dir(tmp_path):
    (tmp_path / "test-org--test-model-xyz").mkdir()
    got = find_local_model("test-org/test-model-xyz", search_dirs=[str(tmp_path)])
    assert got == str(tmp_path / "test-org--test-model-xyz")


def test_find_local_model_not_found(tmp_path):
    assert find_local_model("missing-org/missing-model-zzz", search_dirs=[str(tmp_path)]) is None

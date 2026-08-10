"""A gateway-triggered model switch names the target by its local path (that is
how the fleet advertises loaded models). The HF-id mangling turned that path
into "--models--Qwen--…" — a directory that can never exist — so every switch
toward a path-named model ignored the weights already on disk and tried to
download instead. These tests pin the path-first resolution."""
import asyncio
from pathlib import Path

from src.model.manager import ModelManager


def _make_complete_model(dir_: Path):
    dir_.mkdir(parents=True)
    (dir_ / "config.json").write_text("{}")
    (dir_ / "model.safetensors").write_text("x")


def test_ensure_model_accepts_local_path(tmp_path, monkeypatch):
    import src.model.manager as mm
    monkeypatch.setattr(mm, "is_model_complete", lambda p: Path(p).exists())

    model_dir = tmp_path / "Qwen--Qwen2.5-3B-Instruct"
    _make_complete_model(model_dir)

    m = ModelManager(foc_client=None, cache_dir=str(tmp_path))
    got = asyncio.run(m.ensure_model(str(model_dir)))
    assert got == str(model_dir)


def test_ensure_model_resolves_path_basename_in_cache(tmp_path, monkeypatch):
    # The advertised path is the CONTAINER's mount ("/models/X"); resolution must
    # also try cache_dir/X so a differently-mounted host still finds the weights.
    import src.model.manager as mm
    monkeypatch.setattr(mm, "is_model_complete", lambda p: Path(p).exists())

    _make_complete_model(tmp_path / "Qwen--Qwen2.5-3B-Instruct")
    m = ModelManager(foc_client=None, cache_dir=str(tmp_path))
    got = asyncio.run(m.ensure_model("/nonexistent-mount/Qwen--Qwen2.5-3B-Instruct"))
    assert got == str(tmp_path / "Qwen--Qwen2.5-3B-Instruct")

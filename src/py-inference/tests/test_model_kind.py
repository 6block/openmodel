"""A pool's switch targets must be models its own engine can load."""
import asyncio
import json

import pytest

from src.inference.engine import MockEngine
from src.inference.engine_pool import EnginePool
from src.inference.multipool import MultiPoolEngine, Pool
from src.model.model_kind import check_engine_fits, check_switch_targets, model_kind

TEXT = "/models/Qwen--Qwen3-32B-AWQ"
IMAGE = "/models/stabilityai--stable-diffusion-xl-base-1.0"
TURBO = "/models/stabilityai--sdxl-turbo"


def _dir(tmp_path, name, kind):
    d = tmp_path / name
    d.mkdir()
    if kind == "diffusion":
        (d / "model_index.json").write_text("{}")
    elif kind == "asr":
        (d / "model.bin").write_bytes(b"x")
        (d / "vocabulary.txt").write_text("a")
        (d / "config.json").write_text("{}")
    elif kind == "tts":
        (d / "kokoro-v1_0.pth").write_bytes(b"x")
        (d / "voices").mkdir()
        (d / "config.json").write_text(json.dumps({"istftnet": {}}))
    elif kind == "vllm":
        (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    return str(d)


@pytest.mark.parametrize("kind", ["diffusion", "asr", "tts", "vllm"])
def test_kind_is_read_from_the_weights(tmp_path, kind):
    assert model_kind(_dir(tmp_path, "m", kind)) == kind


def test_whisper_large_v3_layout_is_asr(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "model.bin").write_bytes(b"x")
    (d / "vocabulary.json").write_text("[]")
    (d / "config.json").write_text(json.dumps({"alignment_heads": []}))
    assert model_kind(str(d)) == "asr"


def test_unknown_layout_is_not_refused(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    assert model_kind(str(d)) is None
    check_engine_fits("vllm", "x", str(d))


def test_mismatch_is_refused_with_the_right_engine_named(tmp_path):
    with pytest.raises(ValueError, match="diffusion"):
        check_engine_fits("vllm", IMAGE, _dir(tmp_path, "m", "diffusion"))
    with pytest.raises(ValueError, match="asr"):
        check_engine_fits("tts", "Systran/faster-whisper-small", _dir(tmp_path, "w", "asr"))
    check_engine_fits("diffusion", IMAGE, _dir(tmp_path, "i", "diffusion"))


def test_startup_checks_every_downloaded_switch_target(tmp_path):
    from src.config import load_config
    p = tmp_path / "s.yaml"
    p.write_text(f"""
mode: dev
inference:
  pools:
    - name: text
      engine: vllm
      model: {TEXT}
      device_ids: [0]
      supported_models: [{TEXT}, {IMAGE}]
    - name: image
      engine: diffusion
      model: {TURBO}
      device_ids: [1]
""")
    cfg = load_config(str(p))
    disk = {TEXT: _dir(tmp_path, "t", "vllm"), IMAGE: _dir(tmp_path, "i", "diffusion"),
            TURBO: _dir(tmp_path, "u", "diffusion")}
    with pytest.raises(ValueError, match="stable-diffusion-xl"):
        check_switch_targets(cfg, disk.get)
    # Not downloaded yet: left to the switch-time check, startup goes on.
    check_switch_targets(cfg, {TEXT: disk[TEXT], TURBO: disk[TURBO]}.get)


def test_single_engine_worker_checks_its_claimed_models(tmp_path):
    from src.config import load_config
    p = tmp_path / "s.yaml"
    p.write_text(f"""
mode: dev
inference:
  engine: diffusion
  model: {IMAGE}
gateway:
  supported_models: [{IMAGE}, {TEXT}]
""")
    cfg = load_config(str(p))
    disk = {IMAGE: _dir(tmp_path, "i", "diffusion"), TEXT: _dir(tmp_path, "t", "vllm")}
    with pytest.raises(ValueError, match="vllm"):
        check_switch_targets(cfg, disk.get)


class _Manager:
    def __init__(self, paths):
        self.paths = paths

    async def ensure_model(self, model_id):
        return self.paths[model_id]


def test_switch_to_a_model_of_another_kind_is_refused_before_unloading(tmp_path):
    from fastapi.testclient import TestClient
    from src.inference.api_server import create_app
    e = MockEngine(latency_sec=0.01, gpu_id=0)
    e._model = TEXT
    text = EnginePool("tensor_parallel", [e])
    text.engine_kind = "vllm"
    m = MultiPoolEngine([Pool("text", text, [0], supported_models=[TEXT, IMAGE])])
    asyncio.run(m.start(TEXT))
    app = create_app(m, model_manager=_Manager({IMAGE: _dir(tmp_path, "i", "diffusion")}))
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": IMAGE, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 500
    assert "diffusion" in r.text
    assert m.pools[0].engine.current_model == TEXT, "the resident model must stay loaded"
    # and the pool still serves its own model afterwards
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": TEXT, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text


def test_built_pools_carry_their_engine_kind(tmp_path):
    from src.config import load_config
    from src.inference.engine_pool import create_engine_pool
    p = tmp_path / "s.yaml"
    p.write_text(f"""
mode: dev
inference:
  pools:
    - name: text
      engine: mock
      model: {TEXT}
      device_ids: [0]
""")
    eng = create_engine_pool(load_config(str(p)))
    assert eng.pools[0].engine.engine_kind == "mock"

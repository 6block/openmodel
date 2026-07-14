"""Unit tests for config parsing / multi-GPU resolution (no GPU needed)."""
import src.config as config
from src.config import _interpolate_env_vars, _resolve_multi_gpu, load_config


def test_interpolate_env_vars(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    assert _interpolate_env_vars("x=${FOO}") == "x=bar"
    assert _interpolate_env_vars("y=${UNSET_VAR_XYZ}") == "y="  # unset → empty
    assert _interpolate_env_vars("plain text") == "plain text"


def test_resolve_multi_gpu_default():
    mg = _resolve_multi_gpu({}, "dev")
    assert mg.mode == "tensor_parallel"
    assert mg.gpu_count == 1
    assert mg.device_ids == [0]


def test_resolve_multi_gpu_explicit_count():
    mg = _resolve_multi_gpu({"multi_gpu": {"mode": "multi_instance", "gpu_count": 4}}, "prod")
    assert mg.mode == "multi_instance"
    assert mg.gpu_count == 4
    assert mg.device_ids == [0, 1, 2, 3]


def test_resolve_multi_gpu_device_ids_take_precedence():
    mg = _resolve_multi_gpu({"multi_gpu": {"gpu_count": 8, "device_ids": [2, 3]}}, "prod")
    assert mg.device_ids == [2, 3]
    assert mg.gpu_count == 2  # overridden by len(device_ids)


def test_resolve_multi_gpu_auto_dev_uses_mock(monkeypatch):
    monkeypatch.setattr(config, "detect_gpus", lambda: 0)
    mg = _resolve_multi_gpu({"multi_gpu": {"gpu_count": "auto", "mock_gpu_count": 6}}, "dev")
    assert mg.gpu_count == 6


def test_resolve_multi_gpu_auto_prod_no_gpu_defaults_to_1(monkeypatch):
    monkeypatch.setattr(config, "detect_gpus", lambda: 0)
    mg = _resolve_multi_gpu({"multi_gpu": {"gpu_count": "auto"}}, "prod")
    assert mg.gpu_count == 1


def test_resolve_multi_gpu_auto_uses_detected(monkeypatch):
    monkeypatch.setattr(config, "detect_gpus", lambda: 8)
    mg = _resolve_multi_gpu({"multi_gpu": {"gpu_count": "auto"}}, "prod")
    assert mg.gpu_count == 8
    assert mg.device_ids == [0, 1, 2, 3, 4, 5, 6, 7]


def test_load_config_tensor_parallel_sets_tp_size(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("mode: dev\ninference:\n  engine: vllm\n  multi_gpu:\n    mode: tensor_parallel\n    gpu_count: 4\n")
    cfg = load_config(str(p))
    assert cfg.inference.tensor_parallel_size == 4  # tp_size = gpu_count in tensor_parallel mode
    assert cfg.inference.multi_gpu.mode == "tensor_parallel"


def test_load_config_multi_instance_keeps_tp_size(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("mode: prod\ninference:\n  engine: vllm\n  tensor_parallel_size: 1\n  multi_gpu:\n    mode: multi_instance\n    gpu_count: 8\n")
    cfg = load_config(str(p))
    assert cfg.inference.tensor_parallel_size == 1  # NOT overridden in multi_instance
    assert cfg.inference.multi_gpu.gpu_count == 8


def test_load_config_engine_default_by_mode(tmp_path):
    dev = tmp_path / "dev.yaml"
    dev.write_text("mode: dev\ninference: {}\n")
    assert load_config(str(dev)).inference.engine == "mock"
    prod = tmp_path / "prod.yaml"
    prod.write_text("mode: prod\ninference: {}\n")
    assert load_config(str(prod)).inference.engine == "vllm"


def test_load_config_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_API_TOKEN", "sekret")
    monkeypatch.setenv("SCHEDULER_GRPC_ADDR", "sched:9999")
    p = tmp_path / "cfg.yaml"
    p.write_text("mode: prod\ninference: {}\n")
    cfg = load_config(str(p))
    assert cfg.inference.api_token == "sekret"
    assert cfg.grpc.scheduler_addr == "sched:9999"

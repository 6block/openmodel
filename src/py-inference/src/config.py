"""Configuration loader for the inference service."""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MultiGpuConfig:
    mode: str = "tensor_parallel"  # "tensor_parallel" | "multi_instance"
    gpu_count: int = 1             # Resolved count (after auto-detection)
    device_ids: list[int] = field(default_factory=lambda: [0])


@dataclass
class MultiInstanceConfig:
    load_balancer: str = "round_robin"  # "round_robin" | "least_busy"


@dataclass
class InferenceConfig:
    engine: str = "mock"  # "vllm" or "mock"
    model: str = "meta-llama/Llama-3-8B"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    api_port: int = 8000
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    multi_gpu: MultiGpuConfig = field(default_factory=MultiGpuConfig)
    multi_instance: MultiInstanceConfig = field(default_factory=MultiInstanceConfig)


@dataclass
class FOCConfig:
    backend: str = "mock"
    mock_model_registry: dict = field(default_factory=dict)
    simulate_delay: bool = False
    # Real FOC backend settings
    real_bridge_url: str = "http://127.0.0.1:3100"
    real_piece_cid_registry: dict = field(default_factory=dict)


@dataclass
class GRPCConfig:
    scheduler_addr: str = "localhost:50051"


@dataclass
class LoggingConfig:
    level: str = "info"
    format: str = "json"


@dataclass
class AppConfig:
    mode: str = "dev"
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    foc: FOCConfig = field(default_factory=FOCConfig)
    grpc: GRPCConfig = field(default_factory=GRPCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def detect_gpus() -> int:
    """Detect the number of available NVIDIA GPUs.

    Tries torch.cuda first, falls back to nvidia-smi CLI.
    Returns 0 if no GPUs detected.
    """
    # Try torch.cuda
    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            logger.info("detected %d GPU(s) via torch.cuda", count)
            return count
    except ImportError:
        pass

    # Fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            count = len(lines)
            if count > 0:
                logger.info("detected %d GPU(s) via nvidia-smi", count)
                return count
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.info("no GPUs detected")
    return 0


def _interpolate_env_vars(content: str) -> str:
    """Replace ${ENV_VAR} with environment variable values."""
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return re.sub(r"\$\{(\w+)\}", replacer, content)


def _resolve_multi_gpu(inference_raw: dict, app_mode: str) -> MultiGpuConfig:
    """Resolve multi-GPU configuration, handling 'auto' detection."""
    mg_raw = inference_raw.get("multi_gpu", {})
    if not mg_raw:
        # No multi_gpu config: single GPU default
        return MultiGpuConfig(mode="tensor_parallel", gpu_count=1, device_ids=[0])

    mode = mg_raw.get("mode", "tensor_parallel")
    raw_count = mg_raw.get("gpu_count", "auto")
    raw_device_ids = mg_raw.get("device_ids", None)
    mock_gpu_count = mg_raw.get("mock_gpu_count", 4)

    # Resolve gpu_count
    if raw_count == "auto":
        detected = detect_gpus()
        if detected == 0 and app_mode == "dev":
            gpu_count = mock_gpu_count
            logger.info("dev mode: using mock_gpu_count=%d", gpu_count)
        elif detected == 0:
            gpu_count = 1
            logger.warning("no GPUs detected, defaulting to gpu_count=1")
        else:
            gpu_count = detected
    else:
        gpu_count = int(raw_count)

    # Resolve device_ids
    if raw_device_ids is not None:
        device_ids = [int(d) for d in raw_device_ids]
        gpu_count = len(device_ids)  # device_ids takes precedence
    else:
        device_ids = list(range(gpu_count))

    logger.info("multi_gpu resolved: mode=%s, gpu_count=%d, device_ids=%s",
                mode, gpu_count, device_ids)

    return MultiGpuConfig(mode=mode, gpu_count=gpu_count, device_ids=device_ids)


def load_config(path: str | Path) -> AppConfig:
    """Load and parse the sidecar configuration file."""
    with open(path) as f:
        content = _interpolate_env_vars(f.read())

    raw = yaml.safe_load(content)

    cfg = AppConfig()
    cfg.mode = raw.get("mode", "dev")

    # Parse inference config
    inference_raw = raw.get("inference", {})

    multi_gpu = _resolve_multi_gpu(inference_raw, cfg.mode)

    # In tensor_parallel mode, tensor_parallel_size = gpu_count
    tp_size = inference_raw.get("tensor_parallel_size", 1)
    if multi_gpu.mode == "tensor_parallel":
        tp_size = multi_gpu.gpu_count

    mi_raw = inference_raw.get("multi_instance", {})

    cfg.inference = InferenceConfig(
        engine=inference_raw.get("engine", "mock" if cfg.mode == "dev" else "vllm"),
        model=inference_raw.get("model", "meta-llama/Llama-3-8B"),
        gpu_memory_utilization=inference_raw.get("gpu_memory_utilization", 0.85),
        max_model_len=inference_raw.get("max_model_len", 4096),
        api_port=inference_raw.get("api_port", 8000),
        tensor_parallel_size=tp_size,
        enforce_eager=inference_raw.get("enforce_eager", False),
        multi_gpu=multi_gpu,
        multi_instance=MultiInstanceConfig(
            load_balancer=mi_raw.get("load_balancer", "round_robin"),
        ),
    )

    foc_raw = raw.get("foc", {})
    mock_raw = foc_raw.get("mock", {})
    real_raw = foc_raw.get("real", {})
    cfg.foc = FOCConfig(
        backend=foc_raw.get("backend", "mock"),
        mock_model_registry=mock_raw.get("model_registry", {}),
        simulate_delay=mock_raw.get("simulate_delay", False),
        real_bridge_url=real_raw.get("bridge_url", "http://127.0.0.1:3100"),
        real_piece_cid_registry=real_raw.get("piece_cid_registry", {}),
    )

    grpc_raw = raw.get("grpc", {})
    scheduler_addr = os.environ.get(
        "SCHEDULER_GRPC_ADDR",
        f"localhost:{grpc_raw.get('port', 50051)}",
    )
    cfg.grpc = GRPCConfig(scheduler_addr=scheduler_addr)

    logging_raw = raw.get("logging", {})
    cfg.logging = LoggingConfig(
        level=logging_raw.get("level", "info"),
        format=logging_raw.get("format", "json"),
    )

    return cfg

"""Which engine a model directory belongs to, from its files.

A pool's switch targets must be models its own engine can load. A diffusion model
listed as a switch target of a vLLM text pool (or a whisper model of a TTS pool) is a
configuration mistake that otherwise surfaced only as an opaque load failure on the
first request for it. The kind is read from the weights on disk, the same evidence the
engines themselves rely on.
"""
from __future__ import annotations

import json
import os


def model_kind(model_dir: str) -> str | None:
    """'diffusion', 'asr', 'tts', 'vllm', or None when it cannot be told."""
    def has(name):
        return os.path.exists(os.path.join(model_dir, name))
    if has("model_index.json"):
        return "diffusion"                 # diffusers pipeline
    if has("model.bin") and (has("vocabulary.txt") or has("vocabulary.json")):
        return "asr"                       # CTranslate2 whisper (faster-whisper)
    if any(f.endswith(".pth") for f in _ls(model_dir)) and os.path.isdir(os.path.join(model_dir, "voices")):
        return "tts"                       # Kokoro: .pth weights + voices/
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    if cfg.get("architectures") or cfg.get("model_type"):
        return "vllm"                      # a transformers LM / VLM
    return None


def _ls(d: str) -> list[str]:
    try:
        return os.listdir(d)
    except OSError:
        return []


def check_engine_fits(engine_type: str, model_id: str, model_dir: str) -> None:
    """Raise ValueError when model_dir is a model of a different engine kind."""
    kind = model_kind(model_dir)
    if kind is None or engine_type in ("mock",):
        return
    if kind != engine_type:
        raise ValueError(
            f"model {model_id!r} is a {kind} model and cannot be loaded by this pool's "
            f"{engine_type!r} engine; list it only in a pool whose engine is {kind!r}")


def check_switch_targets(cfg, present_path) -> None:
    """check_engine_fits for every switch target already on disk.

    Per pool on a multi-pool worker; the claimed models (gateway.supported_models)
    on a single-engine worker. present_path(model_id) returns the local directory of
    a complete copy, or None for a model not downloaded yet (checked at switch time).
    """
    groups = [(pc.inference.engine, pc.supported_models) for pc in cfg.pools] \
        or [(cfg.inference.engine, cfg.gateway.supported_models)]
    for engine_type, targets in groups:
        for m in targets:
            local = present_path(m)
            if local:
                check_engine_fits(engine_type, m, local)

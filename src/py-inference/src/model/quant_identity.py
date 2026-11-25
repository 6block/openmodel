"""Quantized weights must be served under their own model id.

A 4-bit build of a model answers the admission probes about as well as the full
model (a 4-bit Qwen3-VL-8B passed the vision probe 16/16 on the test gateway), so the
probes cannot tell them apart. Serving it under the full model's id would charge
users the full model's price for lower quality. The rule is therefore enforced on the
worker before anything loads: if the weights on disk are quantized, the model id the
worker serves them under must name the quantization.
"""
from __future__ import annotations

import json
import os
import re

# Markers that a model id names its quantization. Matched case-insensitively as a
# separate token of the repo name ("Qwen3-32B-AWQ", "…-Instruct-AWQ-4bit", "-FP8",
# "-GPTQ-Int4", "-bnb-4bit", "-w4a16", "-int8").
_ID_MARKERS = re.compile(
    r"(?:^|[-_.])(awq|gptq|fp8|fp4|nvfp4|mxfp4|int4|int8|w4a16|w8a8|w8a16|bnb|\d+bit|gguf|exl2|marlin|compressed)(?:$|[-_.])",
    re.IGNORECASE,
)


def weights_quantization(model_dir: str) -> str | None:
    """The quantization method of the weights in model_dir, or None if full precision
    (or unknown: no readable config.json). Read from config.json's
    quantization_config — the field vLLM itself uses to pick a quantized loader."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    q = cfg.get("quantization_config")
    if not q:
        return None
    return str(q.get("quant_method") or q.get("format") or "quantized")


# Models whose OFFICIAL release is quantized (there is no full-precision original to
# pass off): their own id is the honest name for those weights. Keyed by a lower-case
# substring of the id, value = the quantization methods that release uses.
_NATIVE_QUANTIZATION = {
    "gpt-oss": {"mxfp4"},
}


def _native(model_id: str, method: str) -> bool:
    name = model_id.lower()
    return any(k in name and method.lower() in v for k, v in _NATIVE_QUANTIZATION.items())


def id_names_quantization(model_id: str) -> bool:
    name = model_id.rstrip("/").split("/")[-1]
    name = name.split("--", 1)[-1]          # /models/Org--Name → Name
    return bool(_ID_MARKERS.search(name))


def check_quant_identity(model_id: str, model_dir: str) -> None:
    """Raise ValueError when quantized weights would be served under an id that does
    not name a quantization."""
    q = weights_quantization(model_dir)
    if q and not id_names_quantization(model_id) and not _native(model_id, q):
        raise ValueError(
            f"model {model_id!r} has {q}-quantized weights but its id does not name the "
            f"quantization; quantized models must be served under their own id "
            f"(e.g. the quantized repo's name, such as '…-AWQ' or '…-4bit')")

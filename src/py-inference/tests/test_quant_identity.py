"""Quantized weights may only be served under an id that names the quantization."""
import json

import pytest

from src.model.quant_identity import check_quant_identity, id_names_quantization, weights_quantization


def _model(tmp_path, quant=None):
    d = tmp_path / "m"
    d.mkdir()
    cfg = {"architectures": ["Qwen3VLForConditionalGeneration"]}
    if quant:
        cfg["quantization_config"] = {"quant_method": quant}
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


@pytest.mark.parametrize("model_id", [
    "/models/Qwen--Qwen3-32B-AWQ", "/models/Qwen--Qwen3.8-27B-FP8",
    "/models/cyankiwi--Qwen3-VL-8B-Instruct-AWQ-4bit", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "org/model-w4a16", "org/model-bnb-4bit"])
def test_ids_that_name_a_quantization(model_id):
    assert id_names_quantization(model_id)


@pytest.mark.parametrize("model_id", [
    "/models/Qwen--Qwen3-VL-8B-Instruct", "Qwen/Qwen3-4B-Instruct-2507",
    "/models/stabilityai--stable-diffusion-xl-base-1.0", "/models/Systran--faster-whisper-small",
    "/models/hexgrad--Kokoro-82M"])
def test_full_precision_ids(model_id):
    assert not id_names_quantization(model_id)


def test_quantized_weights_under_the_full_models_id_are_refused(tmp_path):
    # The case found on the test gateway: 4-bit weights served as Qwen3-VL-8B.
    d = _model(tmp_path, "compressed-tensors")
    with pytest.raises(ValueError, match="quantized models must be served under their own id"):
        check_quant_identity("/models/Qwen--Qwen3-VL-8B-Instruct", d)


def test_quantized_weights_under_their_own_id_are_accepted(tmp_path):
    d = _model(tmp_path, "awq")
    check_quant_identity("/models/Qwen--Qwen3-32B-AWQ", d)


def test_full_precision_weights_and_unreadable_configs_pass(tmp_path):
    check_quant_identity("/models/Qwen--Qwen3-VL-8B-Instruct", _model(tmp_path))
    assert weights_quantization(str(tmp_path / "missing")) is None


def test_an_officially_quantized_release_keeps_its_own_id(tmp_path):
    # gpt-oss ships only as MXFP4: its own id is the honest name for those weights.
    check_quant_identity("/models/openai--gpt-oss-20b", _model(tmp_path, "mxfp4"))


def test_native_exception_is_per_model_and_per_method(tmp_path):
    d = _model(tmp_path, "awq")
    with pytest.raises(ValueError):
        check_quant_identity("/models/openai--gpt-oss-20b", d)       # not its native method
    d2 = tmp_path / "x"; d2.mkdir(); (d2 / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "mxfp4"}}))
    with pytest.raises(ValueError):
        check_quant_identity("/models/Qwen--Qwen3-VL-8B-Instruct", str(d2))  # not a native model

"""Audit MEDIUM fixes: sampling-param range validation + exact model-name matching."""
import pytest
from pydantic import ValidationError

from src.inference.api_server import ChatCompletionRequest, CompletionRequest, _model_matches


def test_sampling_params_in_range_ok():
    ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], temperature=0.0, top_p=1.0)
    ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], temperature=2.0, top_p=0.01)


@pytest.mark.parametrize("bad", [
    {"temperature": -0.1},
    {"temperature": 2.5},
    {"top_p": 0.0},
    {"top_p": 1.5},
])
def test_sampling_params_out_of_range_rejected(bad):
    # Out-of-range values must be rejected at the edge (HTTP 422) rather than flowing
    # into vLLM and surfacing as a bare 500 / mid-stream crash.
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **bad)


def test_completion_request_sampling_validation():
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="hi", top_p=9.0)


def test_model_matches_exact_component():
    # Path form vs HuggingFace-id form of the SAME model → match.
    assert _model_matches("/models/Qwen--Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")
    assert _model_matches("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")
    # NOT a match: requested name is only a SUFFIX of the loaded one (the endswith bug).
    assert not _model_matches("/models/Foo--Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct")
    # Different model entirely.
    assert not _model_matches("/models/Qwen--Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")
    # Empty current (nothing loaded).
    assert not _model_matches("", "Qwen/Qwen2.5-1.5B-Instruct")

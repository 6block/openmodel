"""P7 regression: proto YieldUrgency value 2 (IMMEDIATE) must map to
YieldUrgency.IMMEDIATE, not NORMAL. Value 1 is HIGH (reserved)."""
from src.scheduler_client.listener import (
    _proto_urgency_to_engine,
    _URGENCY_NORMAL,
    _URGENCY_HIGH,
    _URGENCY_IMMEDIATE,
)
from src.inference.engine import YieldUrgency


def test_urgency_constants_match_proto():
    # proto/sidecar.proto: NORMAL=0, HIGH=1, IMMEDIATE=2
    assert _URGENCY_NORMAL == 0
    assert _URGENCY_HIGH == 1
    assert _URGENCY_IMMEDIATE == 2


def test_immediate_maps_to_immediate():
    # The actual proto IMMEDIATE value (2) must decode to IMMEDIATE.
    assert _proto_urgency_to_engine(2) == YieldUrgency.IMMEDIATE


def test_normal_maps_to_normal():
    assert _proto_urgency_to_engine(0) == YieldUrgency.NORMAL


def test_unknown_defaults_to_normal():
    # HIGH (1) is unused; unknown values fall back to NORMAL (drain).
    assert _proto_urgency_to_engine(99) == YieldUrgency.NORMAL

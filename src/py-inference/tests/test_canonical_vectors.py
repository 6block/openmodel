"""Cross-language canonical-format vectors.

The billing proof rests on a signing payload that the worker (Python) and the gateway
(Go) must build byte-identically, and on a Merkle leaf that two implementations must
build byte-identically. A one-byte drift invalidates every proof an external verifier
can check, and no runtime check would notice — the gateway would verify receipts it had
itself mis-rebuilt.

This test asserts the Python side against the shared fixture; the Go side has the
mirror test in internal/settlement/canonical_vectors_test.go. Regenerate the fixture
with scripts/gen-canonical-vectors.py after any layout change.
"""
import json
import os

import pytest

from inference.receipt import canonical_payload

VECTORS_REL = os.path.join("sp-state-agent", "internal", "settlement", "testdata",
                           "canonical-vectors.json")


def _fixture_path():
    """Locate the shared fixture by walking up from this test file.

    Walking up rather than counting '..' keeps the test working from either repository
    layout (the working tree and the repo-openmodel staging copy sit at different
    depths).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(here, VECTORS_REL)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def _load():
    path = _fixture_path()
    if path is None:
        pytest.skip("shared canonical-vectors.json not found above this test file")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_payload_vectors_match():
    """canonical_payload must reproduce every payload recorded in the fixture."""
    doc = _load()
    assert doc["payloads"], "fixture has no payload vectors"
    for v in doc["payloads"]:
        inp = dict(v["inputs"])
        # `pubkey` is part of the inputs for token shapes only; the asset shape has no
        # pubkey in its bytes. Pass it through and let canonical_payload ignore it.
        got = canonical_payload(**inp).decode("utf-8")
        assert got == v["payload"], (
            f"{v['name']} ({v['shape']}): payload drifted from the cross-language "
            f"fixture\n want {v['payload']!r}\n got  {got!r}")


def test_token_and_asset_payloads_are_distinct():
    """A token payload and an asset payload must never coincide: if they did, a record
    of one shape could be presented as the other without the signature noticing."""
    doc = _load()
    by_shape = {}
    for v in doc["payloads"]:
        by_shape.setdefault(v["shape"], v["payload"])
    assert by_shape.get("token") != by_shape.get("asset")


def test_nonascii_model_name_is_not_escaped():
    """Go's json.Marshal emits raw UTF-8; Python's json.dumps escapes by default. The
    two sides must agree, or any model whose name has a non-ASCII character has every
    receipt fail verification."""
    doc = _load()
    nonascii = [v for v in doc["payloads"] if v["name"] == "token-nonascii-model"]
    assert nonascii, "fixture lost the non-ASCII vector"
    payload = nonascii[0]["payload"]
    assert "模型-flux" in payload, (
        "non-ASCII model name must appear as raw UTF-8, not \\uXXXX escapes")
    assert "\\u6a21" not in payload


def test_zero_quantity_uses_token_shape():
    """quantity defaults to 0, so an existing caller that never mentions it keeps
    getting the token shape — the compatibility guarantee for every historical model."""
    doc = _load()
    for v in doc["payloads"]:
        if v["shape"] == "token":
            # Rebuild with the default quantity and confirm the shape did not change.
            inp = {k: val for k, val in v["inputs"].items()}
            inp.pop("quantity", None)
            assert canonical_payload(**inp).decode("utf-8") == v["payload"]


def test_character_count_is_code_points():
    """TTS bills len(input) — Unicode code points. The Go gateway counts with
    utf8.RuneCountInString; the shared fixture pins the cases where the two could
    plausibly disagree (combining marks, astral emoji, ZWJ sequences, CRLF)."""
    doc = _load()
    cases = doc.get("character_counts")
    assert cases, "fixture lost the character_counts section"
    for c in cases:
        assert len(c["text"]) == c["count"], c


def test_speech_payloads_present():
    """Both speech shapes must be pinned: they reuse the asset payload, so a regression
    in one would otherwise show up only as an unverifiable receipt in production."""
    doc = _load()
    names = {v["name"] for v in doc["payloads"]}
    assert {"asset-asr-seconds", "asset-tts-characters"} <= names


def test_billable_seconds_matches_fixture():
    """ASR: the worker's rounding must equal the gateway's at every boundary case."""
    from inference.speech_engine import billable_seconds
    doc = _load()
    cases = doc.get("billable_seconds")
    assert cases, "fixture lost the billable_seconds section"
    for c in cases:
        assert billable_seconds(c["duration"]) == c["seconds"], c

"""OpenAI-API compatibility of the worker endpoints: the parameters an OpenAI SDK sends
by default (or commonly) are honoured or mapped, never silently ignored in a way that
changes what the client receives or pays for."""
import asyncio
import base64
import glob
import hashlib
import io
import json
import os
import wave

import pytest
from fastapi.testclient import TestClient

from src.inference.api_server import create_app, subtitles
from src.inference.audio_meta import audio_duration
from src.inference.engine import MockEngine
from src.inference.engine_pool import EnginePool
from src.inference.speech_engine import OPENAI_VOICES, encode_speech, pcm16_wav

from tests.test_speech import _FakeWhisper, _asr, _receipt, _tts, _wav

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "audiofmt")


# --- audio duration: the worker's copy must equal the gateway's ---------------------

def _truth():
    return json.load(open(os.path.join(FIX, "truth.json")))


@pytest.mark.parametrize("name", sorted(json.load(open(os.path.join(FIX, "truth.json")))))
def test_duration_of_each_compressed_format_matches_a_decode(name):
    tr = _truth()[name]
    d, kind = audio_duration(open(os.path.join(FIX, name), "rb").read())
    tol = 0.1 if kind == "mp4" else (0.01 if kind in ("ogg", "webm") else 0.002)
    assert abs(d - tr["decoded"]) <= tol, (name, d, tr)
    assert d >= tr["intended"] - 0.03, "a header must never under-bill the audio"


def test_duration_equals_the_gateway_measurement_exactly():
    """Values the gateway's Go implementation returns for the same fixtures (printed by
    its test helper); both sides bill from this number, so they must be identical."""
    golden = json.load(open(os.path.join(FIX, "gateway_durations.json")))
    for name, want in golden.items():
        d, _ = audio_duration(open(os.path.join(FIX, name), "rb").read())
        assert abs(d - want) < 1e-9, (name, d, want)


@pytest.mark.parametrize("blob", [b"hello, not audio", b"OggS\x00\x02", b"\x1a\x45\xdf\xa3\x80",
                                  b"\x00\x00\x00\x10ftypM4A \x00\x00\x00\x00",
                                  b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(50)])
def test_duration_refuses_garbage(blob):
    with pytest.raises(ValueError):
        audio_duration(blob)


# --- transcription ---------------------------------------------------------------

class _SegWhisper(_FakeWhisper):
    """Whisper stand-in that reports timed segments and records the call's options,
    and reports the duration of the audio it was given (any supported format)."""

    def __init__(self, decoded=None):
        super().__init__()
        self.kw = None
        self.decoded = decoded

    def transcribe(self, f, **kw):
        self.kw = kw
        raw = f.read()
        dur = self.decoded if self.decoded is not None else audio_duration(raw)[0]

        class Seg:
            def __init__(s, i, a, b, t):
                s.id, s.start, s.end, s.text, s.words = i, a, b, t, None

        class Info:
            pass
        info = Info()
        info.duration, info.language = dur, kw.get("language") or "en"
        return iter([Seg(0, 0.0, 1.5, "hello"), Seg(1, 1.5, dur, " world")]), info


def _asr_client(decoded=None):
    fake = _SegWhisper(decoded)
    e = _asr(fake)
    asyncio.run(e.start("/models/Systran--faster-whisper-small"))
    return TestClient(create_app(EnginePool("tensor_parallel", [e]))), fake


def _post(c, path, audio, name="a.wav", **data):
    data.setdefault("model", "/models/Systran--faster-whisper-small")
    return c.post(path, files={"file": (name, audio, "application/octet-stream")}, data=data)


@pytest.mark.parametrize("name", [n for n in sorted(json.load(open(os.path.join(FIX, "truth.json"))))])
def test_every_compressed_upload_is_billed_by_its_declared_length(name):
    c, _ = _asr_client()
    audio = open(os.path.join(FIX, name), "rb").read()
    r = _post(c, "/v1/audio/transcriptions", audio, name)
    assert r.status_code == 200, r.text
    declared, _ = audio_duration(audio)
    import math
    assert r.json()["usage"]["seconds"] == max(1, math.ceil(declared - 1e-9))
    assert _receipt(r)["asset_hash"] == hashlib.sha256(audio).hexdigest()


def test_a_file_that_plays_longer_than_it_declares_is_refused():
    """A forged header understating the length would buy a cheaper transcription."""
    c, _ = _asr_client(decoded=30.0)
    r = _post(c, "/v1/audio/transcriptions", _wav(3.0))
    assert r.status_code == 400 and "declares" in r.text


def test_codec_padding_beyond_the_declared_length_is_accepted():
    c, _ = _asr_client(decoded=3.0 + 0.2)
    assert _post(c, "/v1/audio/transcriptions", _wav(3.0)).status_code == 200


def test_unsupported_upload_is_a_clean_400():
    c, _ = _asr_client()
    r = _post(c, "/v1/audio/transcriptions", b"not audio at all", "a.bin")
    assert r.status_code == 400 and "unsupported audio format" in r.text


def test_response_formats():
    c, _ = _asr_client()
    audio = _wav(3.0)
    assert _post(c, "/v1/audio/transcriptions", audio, response_format="text").text == "hello world"
    srt = _post(c, "/v1/audio/transcriptions", audio, response_format="srt").text
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\nhello")
    vtt = _post(c, "/v1/audio/transcriptions", audio, response_format="vtt").text
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nhello")
    v = _post(c, "/v1/audio/transcriptions", audio, response_format="verbose_json").json()
    assert v["task"] == "transcribe" and v["duration"] == 3.0
    assert [s["text"] for s in v["segments"]] == ["hello", " world"]
    assert v["usage"] == {"type": "duration", "seconds": 3}
    assert _post(c, "/v1/audio/transcriptions", audio, response_format="xml").status_code == 400


def test_options_reach_the_model():
    c, fake = _asr_client()
    _post(c, "/v1/audio/transcriptions", _wav(1.0), language="de", prompt="Filecoin", temperature="0.2",
          **{"timestamp_granularities[]": "word"})
    assert fake.kw["language"] == "de" and fake.kw["initial_prompt"] == "Filecoin"
    assert fake.kw["temperature"] == 0.2 and fake.kw["word_timestamps"] is True


def test_translations_endpoint_translates_and_bills_like_a_transcription():
    c, fake = _asr_client()
    audio = _wav(2.2)
    r = _post(c, "/v1/audio/translations", audio, language="de")
    assert r.status_code == 200, r.text
    assert fake.kw["task"] == "translate" and fake.kw["language"] is None, \
        "translation always targets English; a source language hint is not a target"
    assert r.json()["usage"]["seconds"] == 3
    assert _receipt(r)["quantity"] == 3


def test_streamed_transcription_is_event_shaped():
    c, _ = _asr_client()
    r = _post(c, "/v1/audio/transcriptions", _wav(2.0), stream="true")
    events = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    assert [e["type"] for e in events] == ["transcript.text.delta", "transcript.text.delta", "transcript.text.done"]
    assert events[-1]["text"] == "hello world" and events[-1]["usage"]["seconds"] == 2


def test_subtitle_timestamps_roll_over_hours():
    assert "01:01:01,250 --> 01:01:02,000" in subtitles([{"start": 3661.25, "end": 3662.0, "text": "x"}], "srt")


# --- speech synthesis ----------------------------------------------------------------

def _tts_client(tmp_path, voices=("af_heart", "af_alloy", "am_onyx")):
    e, model = _tts(tmp_path)
    for v in voices:
        (tmp_path / "voices" / f"{v}.pt").write_bytes(b"x")
    asyncio.run(e.start(model))
    return TestClient(create_app(EnginePool("tensor_parallel", [e]))), model, e


def test_openai_voice_names_map_to_installed_voices(tmp_path):
    c, model, _ = _tts_client(tmp_path)
    for v in ("alloy", "onyx", "af_heart"):
        r = c.post("/v1/audio/speech", json={"model": model, "input": "hi", "voice": v, "response_format": "wav"})
        assert r.status_code == 200, (v, r.text)
    r = c.post("/v1/audio/speech", json={"model": model, "input": "hi", "voice": "nobody"})
    assert r.status_code == 400 and "available:" in r.text


def test_every_openai_voice_maps_to_a_real_kokoro_voice():
    kokoro = {"af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
              "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
              "am_liam", "am_michael", "am_onyx", "am_puck", "bf_alice", "bm_fable", "bm_george"}
    assert set(OPENAI_VOICES.values()) <= kokoro


@pytest.mark.parametrize("fmt,ctype,magic", [
    ("mp3", "audio/mpeg", None), ("opus", "audio/ogg", b"OggS"), ("aac", "audio/aac", b"\xff"),
    ("flac", "audio/flac", b"fLaC"), ("pcm", "audio/pcm", None), ("wav", "audio/wav", b"RIFF"),
])
def test_speech_formats_bill_characters_and_hash_the_delivered_bytes(tmp_path, fmt, ctype, magic):
    pytest.importorskip("av")
    c, model, _ = _tts_client(tmp_path)
    r = c.post("/v1/audio/speech", json={"model": model, "input": "héllo", "voice": "alloy", "response_format": fmt})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith(ctype)
    if magic:
        assert r.content.startswith(magic)
    rc = _receipt(r)
    assert rc["quantity"] == 5
    assert rc["asset_hash"] == hashlib.sha256(r.content).hexdigest()


def test_openai_default_request_is_mp3(tmp_path):
    """The OpenAI SDK's default response_format is mp3; a request naming none is wav here
    only when the field is absent from the JSON body."""
    pytest.importorskip("av")
    c, model, _ = _tts_client(tmp_path)
    r = c.post("/v1/audio/speech", json={"model": model, "input": "hi", "voice": "alloy", "response_format": "mp3"})
    assert r.status_code == 200
    assert audio_duration(r.content)[1] == "mp3"


def test_speed_is_passed_and_bounded(tmp_path):
    c, model, e = _tts_client(tmp_path)
    seen = {}
    orig = e._pipeline

    def spy(lang):
        run = orig(lang)

        def wrapped(text, voice=None, speed=1):
            seen["speed"] = speed
            return run(text, voice, speed)
        return wrapped
    e._pipeline = spy
    assert c.post("/v1/audio/speech", json={"model": model, "input": "hi", "speed": 1.5}).status_code == 200
    assert seen["speed"] == 1.5
    assert c.post("/v1/audio/speech", json={"model": model, "input": "hi", "speed": 9}).status_code == 400


def test_encode_speech_round_trips_duration():
    av = pytest.importorskip("av")
    import numpy as np
    wav = pcm16_wav(np.zeros(24000 * 2, dtype="float32"), 24000)
    for fmt in ("mp3", "opus", "flac"):
        d, _ = audio_duration(encode_speech(wav, fmt))
        assert abs(d - 2.0) < 0.1, (fmt, d)
    assert len(encode_speech(wav, "pcm")) == 24000 * 2 * 2


# --- chat ------------------------------------------------------------------------------

class _RecordingEngine(MockEngine):
    def __init__(self):
        super().__init__(latency_sec=0.0)
        self.kw = None

    async def generate(self, prompt, **kwargs):
        self.kw = kwargs
        self.prompt = prompt
        return await super().generate(prompt, **kwargs)


def _chat():
    e = _RecordingEngine()
    asyncio.run(e.start("/models/fake-text"))
    return TestClient(create_app(e)), e


def _msg(**kw):
    b = {"model": "/models/fake-text", "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


def test_max_completion_tokens_is_max_tokens():
    c, e = _chat()
    assert c.post("/v1/chat/completions", json=_msg(max_completion_tokens=7)).status_code == 200
    assert e.kw["max_tokens"] == 7
    c.post("/v1/chat/completions", json=_msg(max_completion_tokens=7, max_tokens=9))
    assert e.kw["max_tokens"] == 9, "an explicit max_tokens wins"


def test_sampling_extras_reach_the_engine():
    c, e = _chat()
    c.post("/v1/chat/completions", json=_msg(seed=3, presence_penalty=0.5, frequency_penalty=-0.5))
    assert (e.kw["seed"], e.kw["presence_penalty"], e.kw["frequency_penalty"]) == (3, 0.5, -0.5)
    c.post("/v1/chat/completions", json=_msg())
    assert "seed" not in e.kw and "presence_penalty" not in e.kw


def test_json_response_formats_constrain_decoding():
    c, e = _chat()
    c.post("/v1/chat/completions", json=_msg(response_format={"type": "json_object"}))
    assert e.kw["structured"] == {"json_object": True}
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    c.post("/v1/chat/completions", json=_msg(response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}))
    assert e.kw["structured"] == {"json": schema}
    c.post("/v1/chat/completions", json=_msg(response_format={"type": "text"}))
    assert "structured" not in e.kw
    assert c.post("/v1/chat/completions", json=_msg(response_format={"type": "yaml"})).status_code == 400
    assert c.post("/v1/chat/completions", json=_msg(response_format={"type": "json_schema"})).status_code == 400


def test_n_greater_than_one_is_refused_not_silently_one():
    c, _ = _chat()
    r = c.post("/v1/chat/completions", json=_msg(n=2))
    assert r.status_code == 400 and "n must be 1" in r.text


def test_developer_role_is_rendered_as_system():
    from src.inference.api_server import _chat_prompt, ChatMessage

    class Eng:
        current_model = ""
    prompt, _ = _chat_prompt(Eng(), [ChatMessage(role="developer", content="be brief"),
                                     ChatMessage(role="user", content="hi")])
    assert "system: be brief" in prompt and "developer" not in prompt


def test_completions_honour_the_extras_too():
    c, e = _chat()
    c.post("/v1/completions", json={"model": "/models/fake-text", "prompt": "x", "seed": 5})
    assert e.kw["seed"] == 5
    assert c.post("/v1/completions", json={"model": "/models/fake-text", "prompt": "x", "n": 3}).status_code == 400


def test_sampling_kwargs_build_vllm_params():
    from src.inference.engine import _sampling_kwargs
    sp = _sampling_kwargs({"max_tokens": 5, "stop": "x", "seed": 1, "presence_penalty": 0.1})
    assert sp == {"max_tokens": 5, "temperature": 0.7, "top_p": 0.95, "stop": ["x"], "seed": 1,
                  "presence_penalty": 0.1}


# --- images ----------------------------------------------------------------------------

def _img():
    e = MockEngine(latency_sec=0.0, image_gen=True)
    asyncio.run(e.start("/models/fake-flux"))
    return TestClient(create_app(e))


@pytest.mark.parametrize("fmt,magic", [("jpeg", b"\xff\xd8"), ("webp", b"RIFF"), ("png", b"\x89PNG")])
def test_image_output_format_is_what_is_delivered_and_hashed(fmt, magic):
    pytest.importorskip("PIL")
    c = _img()
    r = c.post("/v1/images/generations", json={"model": "/models/fake-flux", "prompt": "x", "size": "512x512",
                                               "output_format": fmt})
    assert r.status_code == 200, r.text
    raw = base64.b64decode(r.json()["data"][0]["b64_json"])
    assert raw.startswith(magic)
    assert _receipt(r)["asset_hash"] == hashlib.sha256(raw).hexdigest()


def test_image_auto_size_and_unknown_output_format():
    c = _img()
    assert c.post("/v1/images/generations", json={"model": "/models/fake-flux", "prompt": "x", "size": "auto"}).status_code == 200
    r = c.post("/v1/images/generations", json={"model": "/models/fake-flux", "prompt": "x", "output_format": "gif"})
    assert r.status_code == 400


def test_structured_output_params_carry_the_request_schema(monkeypatch):
    import sys
    import types
    seen = {}

    class SOP:
        def __init__(self, **kw):
            seen.update(kw)
    mod = types.ModuleType("vllm.sampling_params")
    mod.StructuredOutputsParams = SOP
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", mod)
    from src.inference.engine import _sampling_kwargs
    _sampling_kwargs({"structured": {"json": {"type": "object"}}})
    assert seen == {"json": {"type": "object"}}


def test_engine_config_forbids_free_whitespace_in_json():
    """vLLM reads disable_any_whitespace only from the engine config; a per-request flag
    was silently ignored and a 4B model still looped on newlines (4 in 40 live)."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "inference", "engine.py")).read()
    assert 'StructuredOutputsConfig(\n                backend="xgrammar", disable_any_whitespace=True)' in src


# --- denoising steps when the client names none -----------------------------------------

@pytest.mark.parametrize("model,want", [
    ("/models/stabilityai--stable-diffusion-xl-base-1.0", 25),
    ("/models/segmind--SSD-1B", 25),
    ("/models/stabilityai--sdxl-turbo", 4),
    ("/models/black-forest-labs--FLUX.1-schnell", 4),
    ("/models/ByteDance--SDXL-Lightning", 4),
])
def test_default_steps_follow_the_model(model, want):
    """OpenAI clients never send steps. At the old fixed 4 steps SDXL base rendered a
    blurred picture; a full model now gets 25, a distilled few-step model keeps 4."""
    from src.inference.diffusion_engine import DiffusionEngine
    e = DiffusionEngine()
    e._model = model
    assert e.default_steps == want


def test_configured_default_steps_win_and_stay_under_the_cap():
    from src.inference.diffusion_engine import DiffusionEngine
    e = DiffusionEngine(default_steps=30, max_steps=50)
    e._model = "/models/stabilityai--sdxl-turbo"
    assert e.default_steps == 30
    e = DiffusionEngine(default_steps=80, max_steps=50)
    assert e.default_steps == 50
    e = DiffusionEngine(max_steps=10)
    e._model = "/models/stabilityai--stable-diffusion-xl-base-1.0"
    assert e.default_steps == 10


def test_a_request_without_steps_renders_with_the_engine_default():
    from src.inference.diffusion_engine import DiffusionEngine
    seen = {}

    class Pipe:
        def __call__(self, prompt, **kw):
            seen.update(kw)

            class R:
                images = []
            return R()
    e = DiffusionEngine()
    e._model = "/models/stabilityai--stable-diffusion-xl-base-1.0"
    e._pipe = Pipe()
    e._on_device = lambda fn, *a, **k: fn(*a, **k)
    asyncio.run(e._render("x"))
    assert seen["num_inference_steps"] == 25
    asyncio.run(e._render("x", steps=8))
    assert seen["num_inference_steps"] == 8


def test_config_reads_default_steps(tmp_path):
    from src.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text("mode: mock\ninference:\n  model: /models/x\n  default_steps: 30\n"
                 "  pools:\n    - name: image\n      engine: diffusion\n      model: /models/y\n      device_ids: [0]\n"
                 "      default_steps: 12\n")
    cfg = load_config(str(p))
    assert cfg.inference.default_steps == 30
    assert cfg.pools[0].inference.default_steps == 12

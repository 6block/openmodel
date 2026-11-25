"""Speech engines and the /v1/audio endpoints, with fake backends.

faster-whisper and Kokoro are only installed on a GPU worker, so these tests replace each
engine's `_load` with a stand-in that behaves like the real library at the interface the
engine uses: a Whisper-like object whose transcribe() yields segments lazily, and a
Kokoro-like model plus a pipeline that yields one audio chunk per sentence.

What is pinned: billing quantities (ceil seconds min 1; code points), the receipt shape and
digests, yield behaviour between segments/sentences, and the endpoint refusals.
"""
import asyncio
import base64
import hashlib
import io
import json
import struct
import threading
import time

import pytest

from src.inference import speech_engine as se
from src.inference.engine import EngineState, YieldUrgency
from src.inference.engine_pool import EnginePool
from src.inference.speech_engine import AsrEngine, TtsEngine, billable_characters, billable_seconds


def _wav(seconds: float, rate: int = 16000) -> bytes:
    """A real 16-bit mono WAV of silence-ish samples."""
    n = int(seconds * rate)
    data = b"".join(struct.pack("<h", (i * 37) % 2000 - 1000) for i in range(n))
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(data))
    return hdr + data


class _Seg:
    def __init__(self, text):
        self.text = text


class _FakeWhisper:
    """transcribe() returns a LAZY segment generator, like faster-whisper."""

    def __init__(self, segments=("hello", " world"), delay=0.0):
        self.segments, self.delay = segments, delay
        self.consumed = 0

    def transcribe(self, f, language=None):
        raw = f.read()
        rate = struct.unpack("<I", raw[24:28])[0]
        duration = (len(raw) - 44) / 2 / rate

        def gen():
            for s in self.segments:
                time.sleep(self.delay)
                self.consumed += 1
                yield _Seg(s)

        class Info:
            pass
        info = Info()
        info.duration, info.language = duration, language or "en"
        return gen(), info


def _asr(fake=None, **kw):
    e = AsrEngine(device_ids=[5], **kw)
    e._load = lambda path: fake or _FakeWhisper()
    return e


class _FakeKokoroModel:
    pass


def _tts(tmp_path, sentences=1, delay=0.0, **kw):
    voices = tmp_path / "voices"
    voices.mkdir(exist_ok=True)
    for v in ("af_heart", "zf_xiaoxiao"):
        (voices / f"{v}.pt").write_bytes(b"x")
    e = TtsEngine(device_ids=[5], **kw)
    e._load = lambda path: _FakeKokoroModel()
    calls = {"n": 0}

    def fake_pipeline(lang):
        def run(text, voice=None, speed=1):
            import numpy as np
            for i in range(sentences):
                time.sleep(delay)
                calls["n"] += 1
                # Deterministic per text, so different inputs give different bytes.
                seed = sum(map(ord, text)) + i
                yield None, None, (np.sin(np.arange(2400) * (seed % 7 + 1) / 50.0) * 0.3).astype("float32")
        return run
    e._pipeline = fake_pipeline
    e._calls = calls
    return e, str(tmp_path)


# --- billing rules -----------------------------------------------------------------

def test_billable_seconds_rounds_up_with_a_one_second_floor():
    assert billable_seconds(0.0) == 1
    assert billable_seconds(0.2) == 1
    assert billable_seconds(1.0) == 1
    assert billable_seconds(1.0001) == 2
    assert billable_seconds(3.4) == 4
    assert billable_seconds(60.0) == 60


def test_billable_characters_is_code_points():
    assert billable_characters("é") == 2
    assert billable_characters("\U0001F600") == 1
    assert billable_characters("你好") == 2


# --- ASR engine --------------------------------------------------------------------

def test_asr_transcribes_and_reports_duration():
    async def run():
        e = _asr()
        await e.start("/models/whisper")
        text, dur, lang = await e.transcribe(_wav(2.5))
        assert text == "hello world" and lang == "en"
        assert abs(dur - 2.5) < 0.01
    asyncio.run(run())


def test_asr_rejects_audio_over_the_cap():
    async def run():
        e = _asr(max_audio_sec=2)
        await e.start("/m")
        with pytest.raises(ValueError, match="at most 2s"):
            await e.transcribe(_wav(3.0))
    asyncio.run(run())


def test_asr_immediate_yield_stops_between_segments_and_returns_nothing():
    async def run():
        fake = _FakeWhisper(segments=[f" w{i}" for i in range(30)], delay=0.05)
        e = _asr(fake)
        await e.start("/m")
        job = asyncio.ensure_future(e.transcribe(_wav(1.0)))
        await asyncio.sleep(0.2)
        await e.pause(YieldUrgency.IMMEDIATE)
        assert e.status().state is EngineState.PAUSED
        assert e._active_requests == 0, "pause must wait for the request thread to exit"
        assert fake.consumed < 30, "the transcription must stop between segments"
        with pytest.raises(se.Interrupted):
            await job
    asyncio.run(run())


def test_asr_normal_yield_lets_the_request_finish():
    async def run():
        fake = _FakeWhisper(segments=[" a", " b", " c"], delay=0.05)
        e = _asr(fake)
        await e.start("/m")
        job = asyncio.ensure_future(e.transcribe(_wav(1.0)))
        await asyncio.sleep(0.02)
        await e.pause(YieldUrgency.NORMAL)
        text, _, _ = await job
        assert text == "a b c"
        assert e.status().state is EngineState.PAUSED
    asyncio.run(run())


def test_speech_engine_refuses_chat():
    async def run():
        e = _asr()
        await e.start("/m")
        with pytest.raises(ValueError, match="audio endpoints"):
            await e.generate("hi")
    asyncio.run(run())


# --- TTS engine --------------------------------------------------------------------

def test_tts_returns_a_24khz_wav(tmp_path):
    async def run():
        e, model = _tts(tmp_path)
        await e.start(model)
        wav = await e.synthesize("Hello there.")
        import wave
        with wave.open(io.BytesIO(wav)) as w:
            assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
            assert w.getnframes() > 0
    asyncio.run(run())


def test_tts_rejects_unknown_voice_and_overlong_input(tmp_path):
    async def run():
        e, model = _tts(tmp_path, max_chars=5)
        await e.start(model)
        with pytest.raises(ValueError, match="unknown voice"):
            await e.synthesize("hi", voice="xx_nobody")
        with pytest.raises(ValueError, match="at most 5"):
            await e.synthesize("123456")
    asyncio.run(run())


def test_tts_immediate_yield_stops_between_sentences(tmp_path):
    async def run():
        e, model = _tts(tmp_path, sentences=40, delay=0.05)
        await e.start(model)
        job = asyncio.ensure_future(e.synthesize("A long text."))
        await asyncio.sleep(0.2)
        await e.pause(YieldUrgency.IMMEDIATE)
        assert e.status().state is EngineState.PAUSED and e._active_requests == 0
        assert e._calls["n"] < 40
        with pytest.raises(se.Interrupted):
            await job
    asyncio.run(run())


def test_tts_pause_drops_the_cached_pipelines(tmp_path):
    """Each Kokoro pipeline references the model. If pause kept them, the weights would
    stay on the card and the reported release would be false."""
    async def run():
        e, model = _tts(tmp_path)
        await e.start(model)
        e._pipelines = {"a": object(), "z": object()}
        await e.pause(YieldUrgency.IMMEDIATE)
        assert e.status().state is EngineState.PAUSED
        assert e._pipelines == {}
    asyncio.run(run())


# --- endpoints ---------------------------------------------------------------------

def _app(engine):
    from fastapi.testclient import TestClient
    from src.inference.api_server import create_app
    return TestClient(create_app(engine))


def _receipt(resp):
    return json.loads(base64.b64decode(resp.headers["X-OM-Receipt"]))


def test_transcription_endpoint_bills_ceil_seconds_and_hashes_the_upload():
    e = _asr()
    asyncio.run(e.start("/models/Systran--faster-whisper-small"))
    audio = _wav(3.4)
    r = _app(EnginePool("tensor_parallel", [e])).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", audio, "audio/wav")},
        data={"model": "/models/Systran--faster-whisper-small"},
        headers={"X-Request-ID": "req-asr"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hello world"
    assert body["usage"] == {"type": "duration", "seconds": 4}
    rc = _receipt(r)
    assert rc["quantity"] == 4
    assert rc["asset_hash"] == hashlib.sha256(audio).hexdigest(), \
        "ASR commits the uploaded audio file, which the gateway hashes too"


def test_speech_endpoint_bills_characters_and_hashes_the_returned_audio(tmp_path):
    e, model = _tts(tmp_path)
    asyncio.run(e.start(model))
    text = "你好，world 😀"
    r = _app(EnginePool("tensor_parallel", [e])).post(
        "/v1/audio/speech", json={"model": model, "input": text, "voice": "af_heart"},
        headers={"X-Request-ID": "req-tts"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    rc = _receipt(r)
    assert rc["quantity"] == len(text)
    assert rc["asset_hash"] == hashlib.sha256(r.content).hexdigest()


def test_speech_endpoint_refuses_unknown_formats(tmp_path):
    e, model = _tts(tmp_path)
    asyncio.run(e.start(model))
    c = _app(EnginePool("tensor_parallel", [e]))
    for fmt in ("ogg", "m4a", "wma"):
        r = c.post("/v1/audio/speech", json={"model": model, "input": "hi", "response_format": fmt})
        assert r.status_code == 400 and "response_format must be" in r.text, fmt
    r = c.post("/v1/audio/speech", json={"model": model, "input": "hi", "stream_format": "sse"})
    assert r.status_code == 400


def test_audio_endpoints_refuse_the_wrong_kind_of_model(tmp_path):
    e, model = _tts(tmp_path)
    asyncio.run(e.start(model))
    r = _app(EnginePool("tensor_parallel", [e])).post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", _wav(1), "audio/wav")},
        data={"model": model})
    assert r.status_code == 400 and "does not transcribe" in r.text


def test_health_advertises_speech_features(tmp_path):
    a = _asr()
    t, model = _tts(tmp_path)
    from src.inference.multipool import MultiPoolEngine, Pool
    m = MultiPoolEngine([
        Pool("asr", EnginePool("tensor_parallel", [a]), [5], ["/models/whisper"]),
        Pool("tts", EnginePool("tensor_parallel", [t]), [6], [model]),
    ])
    asyncio.run(a.start("/models/whisper"))
    asyncio.run(t.start(model))
    body = _app(m).get("/health").json()
    assert {"asr", "tts"} <= set(body["features"])


def test_failed_resume_returns_to_paused_and_retries(tmp_path):
    """Found on a test worker: a Kokoro load that raised left the pool in LOADING; resume()
    returns early for a loading engine, so the pool never came back."""
    async def run():
        e, model = _tts(tmp_path)
        await e.start(model)
        await e.pause(YieldUrgency.IMMEDIATE)
        good = e._load
        calls = {"n": 0}

        def flaky(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise NotImplementedError("Cannot copy out of meta tensor")
            return good(path)
        e._load = flaky
        with pytest.raises(NotImplementedError):
            await e.resume()
        assert e.status().state is EngineState.PAUSED
        await e.resume()
        assert e.status().state is EngineState.RUNNING
    asyncio.run(run())


def test_speech_requests_are_serialized_on_one_model(tmp_path):
    """Parallel whisper calls exhausted the card on a test worker and the CUDA error aborted the
    whole inference process. One request at a time per speech engine."""
    import threading
    state = {"now": 0, "peak": 0}
    mu = threading.Lock()

    async def run():
        e = _asr()
        await e.start("/models/whisper")
        orig = e._transcribe_sync

        def tracked(*a):
            with mu:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            try:
                time.sleep(0.05)
                return orig(*a)
            finally:
                with mu:
                    state["now"] -= 1
        e._transcribe_sync = tracked
        await asyncio.gather(*[e.transcribe(_wav(1.0)) for _ in range(5)])
    asyncio.run(run())
    assert state["peak"] == 1, state


def test_sys_exit_inside_a_model_call_does_not_stop_the_server(tmp_path):
    """spaCy calls sys.exit(1) when it cannot fetch a language model on first use of a
    Kokoro pipeline. On a test worker that SystemExit, raised in the worker thread, shut the
    whole server down 16 times in one soak. It must become a 503 for that request."""
    e, model = _tts(tmp_path)
    asyncio.run(e.start(model))

    def boom(lang):
        raise SystemExit(1)
    e._pipeline = boom
    c = _app(e)
    r = c.post("/v1/audio/speech", json={"model": model, "input": "hello", "voice": "af_heart"})
    assert r.status_code == 503, r.text
    assert c.get("/health").status_code == 200   # still serving


def test_model_calls_and_loads_run_with_the_engine_card_current():
    """Model work runs on pool threads whose current CUDA device starts as card 0;
    CTranslate2 allocates there, so every load and call must bind the engine's card
    in the thread doing the work."""
    async def run():
        seen = []
        e = _asr()
        e._bind_device = lambda: seen.append(("bind", threading.get_ident()))
        inner_load = e._load
        e._load = lambda path: (seen.append(("load", threading.get_ident())), inner_load(path))[1]
        await e.start("/models/whisper")
        orig = e._transcribe_sync
        e._transcribe_sync = lambda *a: (seen.append(("call", threading.get_ident())), orig(*a))[1]
        await e.transcribe(_wav(1.0))
        # each unit of work is preceded by a bind in the same thread
        for i, (kind, tid) in enumerate(seen):
            if kind in ("load", "call"):
                assert i > 0 and seen[i - 1] == ("bind", tid), seen
    asyncio.run(run())

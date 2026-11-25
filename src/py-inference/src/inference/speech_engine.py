"""Speech engines: transcription (ASR, faster-whisper) and synthesis (TTS, Kokoro).

Both follow the same implicit engine contract DiffusionEngine documents, because the pool,
the scheduler listener and the API server touch all of it:

  * `_model` / `_state` attributes (EnginePool.start_paused() writes them directly);
  * `current_model`, `switch_model`, `request_pause`, and a `generate_stream` that refuses
    clearly rather than being absent;
  * `pause()` must leave `status().state == PAUSED`, and only after the device's memory
    is actually back — the scheduler hands the card to mining on that word.

WHY A SHARED BASE. The lifecycle (load / unload / drain / pending pause during a load /
pending resume during an unload) is identical for both speech models and subtle enough
that two hand copies would drift. Each subclass supplies only how to load its model and
how to run one request.

INTERRUPTION. A transcription is processed segment by segment and a synthesis sentence by
sentence; both check the yield flag between those units. A request interrupted that way
raises instead of returning a partial result — a half transcript or half an utterance is
not what was asked for and must never be returned or billed.

BILLING QUANTITIES ARE NOT DECIDED HERE. The engines report what they did (audio duration,
characters spoken), but the receipt's billable quantity is computed by the API layer with
the same rule the gateway uses, so the two cannot drift: ceil(seconds) with a minimum of 1
for ASR, Unicode code points of the input for TTS.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import math
import os
import time

from .engine import EngineState, EngineStatus, GenerateResult, InferenceEngine, YieldUrgency, query_gpu_free_ratio
from .torch_globals import guarded_call, guarded_load

logger = logging.getLogger(__name__)

# Soft-yield drain timeout. A long transcription is tens of seconds on a 3080; a queue of
# them can be minutes, as with image generation.
_SPEECH_DRAIN_TIMEOUT = float(os.environ.get("SPEECH_DRAIN_TIMEOUT_SEC", "120"))
_IMMEDIATE_STOP_TIMEOUT = float(os.environ.get("SPEECH_IMMEDIATE_STOP_SEC", "20"))


def billable_seconds(duration_s: float) -> int:
    """ASR billing quantity: input audio duration rounded UP to whole seconds, minimum 1.

    Same rule as OpenAI whisper-1 and Google STT V2, and the same function the gateway
    applies to the bytes it measured itself.
    """
    if duration_s <= 0:
        return 1
    return max(1, math.ceil(duration_s - 1e-9))


def billable_characters(text: str) -> int:
    """TTS billing quantity: Unicode code points of the input, no trimming or
    normalization (the gateway's utf8.RuneCountInString)."""
    return len(text)


class Interrupted(RuntimeError):
    """A request stopped at a safe point because the GPU is being yielded."""

    def __init__(self):
        super().__init__("Engine not available — GPU yielded to mining during processing")


class _SpeechEngineBase(InferenceEngine):
    """Lifecycle shared by the ASR and TTS engines."""

    kind = "speech"

    def __init__(self, device_ids: list[int] | None = None, gpu_memory_utilization: float = 0.85):
        self._state = EngineState.STOPPED
        self._model = ""
        self._impl = None
        self._device_ids = device_ids or [0]
        self._gpu_memory_utilization = gpu_memory_utilization
        self._active_requests = 0
        self._lock = asyncio.Lock()
        # One request on the model at a time. The model objects are not safe for
        # concurrent calls, and each call allocates its own working memory: under a
        # mixed-load soak, parallel whisper calls exhausted the card and the CUDA
        # error aborted the whole inference process (every pool, every request).
        self._work_lock = asyncio.Lock()
        self._pending_pause: YieldUrgency | None = None
        self._pending_resume = False
        self._interrupt_requested = False
        self._accepting = True

    # --- subclass hooks ------------------------------------------------------------

    def _load(self, model_path: str):  # pragma: no cover - needs the real libraries
        raise NotImplementedError

    @property
    def device(self) -> str:
        return f"cuda:{self._device_ids[0]}"

    # --- capabilities ----------------------------------------------------------------

    @property
    def is_vlm(self) -> bool:
        return False

    @property
    def is_image_gen(self) -> bool:
        return False

    @property
    def max_images_per_prompt(self) -> int:
        return 0

    @property
    def max_images_per_request(self) -> int:
        return 0

    @property
    def current_model(self) -> str:
        return self._model

    # --- lifecycle -------------------------------------------------------------------

    async def start(self, model_path: str) -> None:
        async with self._lock:
            self._state = EngineState.STARTING
            self._model = model_path
        await self._load_impl()
        await self._after_load()

    async def _load_impl(self) -> None:
        t0 = time.time()
        logger.info("loading %s model %s on %s", self.kind, self._model, self.device)
        try:
            self._impl = await asyncio.to_thread(self._load_on_device, self._model)
        except BaseException:
            # Back to PAUSED, not stuck in LOADING: resume() returns early for a loading
            # engine, so a failed load would otherwise never be retried and the pool
            # would report "loading" until the process restarted.
            async with self._lock:
                self._impl = None
                self._state = EngineState.PAUSED
            raise
        logger.info("%s model ready in %.1fs", self.kind, time.time() - t0)

    async def _after_load(self) -> None:
        async with self._lock:
            self._state = EngineState.RUNNING
            pending, self._pending_pause = self._pending_pause, None
        if pending is not None:
            logger.info("%s: pause requested during load — unloading now", self.kind)
            await self.pause(pending)

    async def _destroy(self) -> None:
        if self._impl is None:
            return
        self._impl = None
        self._drop_references()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                # empty_cache acts on the CURRENT device; this pool's card is not it in
                # a multi-pool process, so select it explicitly.
                with torch.cuda.device(self._device_ids[0]):
                    torch.cuda.empty_cache()
        except Exception:
            logger.debug("torch cache clear skipped", exc_info=True)
        await self._wait_for_vram_release()

    def _drop_references(self) -> None:
        """Release engine-specific objects that still reference the model, so the
        garbage collector can free its GPU memory. Overridden where needed."""

    async def _wait_for_vram_release(self, timeout: float = 20.0) -> None:
        # nvidia-smi on the physical device, not torch.cuda.mem_get_info (which binds to
        # whichever device the process initialised first).
        gpu_id = self._device_ids[0]
        deadline = time.time() + timeout
        while time.time() < deadline:
            ratio = query_gpu_free_ratio(gpu_id)
            if ratio is None or ratio >= (1.0 - self._gpu_memory_utilization):
                return
            await asyncio.sleep(0.5)
        logger.warning("%s: VRAM release not confirmed within %.0fs on gpu %d", self.kind, timeout, gpu_id)

    async def pause(self, urgency: YieldUrgency = YieldUrgency.NORMAL) -> None:
        async with self._lock:
            if self._state in (EngineState.LOADING, EngineState.STARTING):
                self._pending_pause = urgency
                return
            if self._state != EngineState.RUNNING:
                return
            self._accepting = False
            self._interrupt_requested = True
            if urgency != YieldUrgency.NORMAL:
                self._state = EngineState.UNLOADING

        if urgency == YieldUrgency.NORMAL:
            # NORMAL lets in-flight requests finish; they check the flag only for an
            # IMMEDIATE yield (state != RUNNING), so clear the request-side signal.
            self._interrupt_requested = False
            await self._drain(_SPEECH_DRAIN_TIMEOUT)
            async with self._lock:
                if self._state != EngineState.RUNNING:
                    return
                self._state = EngineState.UNLOADING
                self._interrupt_requested = True

        async with self._lock:
            # Never release the model while a request thread still holds it: the memory
            # would not come back and PAUSED would be a lie (P2's IMMEDIATE lesson).
            if self._active_requests:
                await self._drain(_IMMEDIATE_STOP_TIMEOUT)
            await self._destroy()
            self._interrupt_requested = False
            self._accepting = True
            if self._pending_resume:
                self._pending_resume = False
                self._state = EngineState.LOADING
            else:
                self._state = EngineState.PAUSED
                logger.info("%s engine paused — GPU memory released", self.kind)
                return
        await self._load_impl()
        await self._after_load()

    async def resume(self) -> None:
        async with self._lock:
            if self._state == EngineState.UNLOADING:
                self._pending_resume = True
                return
            if self._state in (EngineState.RUNNING, EngineState.LOADING, EngineState.STARTING):
                return
            self._state = EngineState.LOADING
        await self._load_impl()
        await self._after_load()

    async def stop(self) -> None:
        async with self._lock:
            self._state = EngineState.STOPPING
            await self._destroy()
            self._state = EngineState.STOPPED

    async def switch_model(self, new_model_path: str) -> None:
        if new_model_path == self._model and self._impl is not None:
            return
        old_model = self._model
        await self.pause(YieldUrgency.NORMAL)
        async with self._lock:
            self._model = new_model_path
            self._state = EngineState.LOADING
        try:
            await self._load_impl()   # a failure leaves the engine PAUSED
        except BaseException as e:
            # Roll back like the vLLM engine: keep serving the old model rather than
            # staying down on one that cannot load.
            logger.error("%s: model switch failed, rolling back to %s: %s", self.kind, old_model, e)
            self._model = old_model
            try:
                await self.resume()
            except Exception as rollback_err:
                logger.error("%s: rollback also failed: %s", self.kind, rollback_err)
            raise
        await self._after_load()

    def request_pause(self) -> None:
        self._interrupt_requested = True

    def is_available(self) -> bool:
        return self._state == EngineState.RUNNING and self._impl is not None and self._accepting

    def status(self) -> EngineStatus:
        return EngineStatus(state=self._state, active_requests=self._active_requests,
                            gpu_utilization_pct=0.0, loaded_model=self._model,
                            gpu_id=self._device_ids[0] if self._device_ids else -1)

    def _load_on_device(self, model_path: str):
        self._bind_device()
        return guarded_load(self._load, model_path)   # see torch_globals

    async def _drain(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while self._active_requests > 0 and time.time() < deadline:
            await asyncio.sleep(0.1)
        if self._active_requests:
            logger.warning("%s: drain timeout, %d request(s) still in flight", self.kind, self._active_requests)

    def _should_stop(self) -> bool:
        """Checked between segments / sentences, from the worker thread."""
        return self._interrupt_requested and self._state != EngineState.RUNNING

    def _bind_device(self) -> None:  # pragma: no cover - needs a GPU
        """Make this engine's card the current CUDA device of the calling thread.

        Model calls run on asyncio's thread pool, and a fresh thread's current device
        is card 0. CTranslate2 (faster-whisper) makes some allocations on the current
        device rather than the model's: with card 0 full of a vLLM engine its first
        transcription aborted the whole process with a C++ "CUDA failed with error out
        of memory" (seen on a test worker after restarts, 2 of 3 times)."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(self._device_ids[0])
        except ImportError:
            pass

    def _contained(self, fn, *args):
        """Run a model call, turning SystemExit/KeyboardInterrupt into an error.

        Libraries under Kokoro call sys.exit() from deep inside a request (spaCy exits
        1 when it cannot download a language model). Raised in a worker thread that
        SystemExit reached the event loop and shut the whole inference server down —
        every pool, every in-flight request — and the container restarted into the
        same failure (seen 16 times in a 30-minute soak test)."""
        try:
            self._bind_device()
            return guarded_call(fn, *args)
        except (SystemExit, KeyboardInterrupt) as e:
            raise RuntimeError(f"model call aborted ({type(e).__name__}: {e})") from None

    async def _run(self, fn, *args):
        if not self.is_available():
            raise RuntimeError("Engine not available — GPU yielded to mining")
        self._active_requests += 1
        try:
            async with self._work_lock:
                # Re-check after the wait: a yield may have unloaded the model.
                if self._impl is None or self._state != EngineState.RUNNING:
                    raise RuntimeError("Engine not available — GPU yielded to mining")
                return await asyncio.to_thread(self._contained, fn, *args)
        finally:
            self._active_requests -= 1

    # Chat/completions do not apply to a speech model; refuse plainly.
    async def generate(self, prompt: str, **kwargs) -> GenerateResult:
        raise ValueError(f"model {self._model!r} is a {self.kind} model; use the /v1/audio endpoints")

    async def generate_stream(self, prompt: str, **kwargs):
        raise NotImplementedError(f"{self.kind} models do not stream chat output")
        yield  # pragma: no cover


class AsrEngine(_SpeechEngineBase):
    """Speech-to-text with faster-whisper (CTranslate2)."""

    kind = "asr"

    def __init__(self, device_ids=None, gpu_memory_utilization: float = 0.85, max_audio_sec: int = 600):
        super().__init__(device_ids, gpu_memory_utilization)
        self._max_audio_sec = int(max_audio_sec)

    @property
    def is_asr(self) -> bool:
        return True

    @property
    def max_audio_sec(self) -> int:
        return self._max_audio_sec

    def _load(self, model_path: str):
        from faster_whisper import WhisperModel
        return WhisperModel(model_path, device="cuda", device_index=self._device_ids[0],
                            compute_type="float16")

    async def transcribe(self, audio: bytes, language: str | None = None) -> tuple[str, float, str]:
        """Return (text, duration_seconds, detected_language)."""
        r = await self.transcribe_full(audio, language)
        return r["text"], r["duration"], r["language"]

    async def transcribe_full(self, audio: bytes, language: str | None = None, *,
                              task: str = "transcribe", prompt: str | None = None,
                              temperature: float | None = None,
                              word_timestamps: bool = False) -> dict:
        """Transcribe (or translate to English, task="translate").

        Returns {"text", "duration", "language", "segments": [{"id", "start", "end",
        "text", "words"?}]}. `duration` is what the decoder produced from the audio.
        """
        return await self._run(self._transcribe_sync, audio, language, task, prompt,
                               temperature, word_timestamps)

    def _transcribe_sync(self, audio: bytes, language, task="transcribe", prompt=None,
                         temperature=None, word_timestamps=False):
        kw = {"language": language or None}
        if task != "transcribe":
            kw["task"] = task
        if prompt:
            kw["initial_prompt"] = prompt
        if temperature is not None:
            kw["temperature"] = float(temperature)
        if word_timestamps:
            kw["word_timestamps"] = True
        segments, info = self._impl.transcribe(io.BytesIO(audio), **kw)
        if info.duration > self._max_audio_sec:
            raise ValueError(f"audio is {info.duration:.1f}s, at most {self._max_audio_sec}s per request")
        out = []
        for seg in segments:          # lazily decoded: each step is one segment
            if self._should_stop():
                raise Interrupted()
            item = {"id": len(out), "start": round(float(getattr(seg, "start", 0.0) or 0.0), 3),
                    "end": round(float(getattr(seg, "end", 0.0) or 0.0), 3), "text": seg.text}
            words = getattr(seg, "words", None)
            if word_timestamps and words:
                item["words"] = [{"word": w.word, "start": round(float(w.start), 3),
                                  "end": round(float(w.end), 3)} for w in words]
            out.append(item)
        return {"text": "".join(x["text"] for x in out).strip(), "duration": float(info.duration),
                "language": info.language, "segments": out}


class TtsEngine(_SpeechEngineBase):
    """Text-to-speech with Kokoro-82M (24 kHz)."""

    kind = "tts"
    SAMPLE_RATE = 24000

    def __init__(self, device_ids=None, gpu_memory_utilization: float = 0.85,
                 max_chars: int = 4000, default_voice: str = ""):
        super().__init__(device_ids, gpu_memory_utilization)
        self._max_chars = int(max_chars)
        self._default_voice = default_voice or "af_heart"
        self._pipelines: dict[str, object] = {}

    @property
    def is_tts(self) -> bool:
        return True

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def voices(self) -> list[str]:
        d = os.path.join(self._model, "voices")
        try:
            return sorted(f[:-3] for f in os.listdir(d) if f.endswith(".pt"))
        except OSError:
            return []

    def _load(self, model_path: str):
        from kokoro import KModel
        model = KModel(config=os.path.join(model_path, "config.json"),
                       model=os.path.join(model_path, "kokoro-v1_0.pth")).to(self.device).eval()
        self._pipelines = {}
        return model

    def _drop_references(self) -> None:
        # Each KPipeline holds the KModel; dropping only _impl would keep the weights
        # on the card through the cached pipelines.
        self._pipelines = {}

    def _pipeline(self, lang_code: str):
        from kokoro import KPipeline
        p = self._pipelines.get(lang_code)
        if p is None:
            p = KPipeline(lang_code=lang_code, model=self._impl, repo_id="hexgrad/Kokoro-82M",
                          device=self.device)
            self._pipelines[lang_code] = p
        return p

    async def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0) -> bytes:
        """Return a 16-bit PCM WAV at 24 kHz."""
        if billable_characters(text) > self._max_chars:
            raise ValueError(f"input is {billable_characters(text)} characters, at most {self._max_chars}")
        if not text.strip():
            raise ValueError("input text is empty")
        voice = self.resolve_voice(voice)
        if not 0.25 <= float(speed) <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        return await self._run(self._synthesize_sync, text, voice, float(speed))

    def resolve_voice(self, voice: str | None) -> str:
        """The installed voice a request's `voice` names: an installed id as is, or one
        of the OpenAI voice names mapped onto an installed voice (OPENAI_VOICES)."""
        voice = voice or self._default_voice
        available = self.voices()
        if voice in available:
            return voice
        mapped = OPENAI_VOICES.get(str(voice).lower())
        if mapped and mapped in available:
            return mapped
        raise ValueError(f"unknown voice {voice!r}; available: {', '.join(available)}")

    def _synthesize_sync(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        import numpy as np
        # Kokoro voice ids encode the language in their first letter (a=US English,
        # b=British, z=Mandarin, j=Japanese, …), which is also its pipeline lang code.
        pipe = self._pipeline(voice[0])
        chunks = []
        for _, _, audio in pipe(text, voice=os.path.join(self._model, "voices", voice + ".pt"), speed=speed):
            if self._should_stop():   # between sentences
                raise Interrupted()
            chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        if not chunks:
            raise ValueError("nothing to synthesize")
        return pcm16_wav(np.concatenate(chunks), self.SAMPLE_RATE)


# OpenAI voice names → installed Kokoro voices, so a client written for the OpenAI API
# works unchanged. Names Kokoro ships directly (alloy, echo, fable, nova, onyx) map to
# that voice; the others to a voice of the same register.
OPENAI_VOICES = {
    "alloy": "af_alloy", "ash": "am_adam", "ballad": "bm_george", "coral": "af_heart",
    "echo": "am_echo", "fable": "bm_fable", "nova": "af_nova", "onyx": "am_onyx",
    "sage": "af_sarah", "shimmer": "af_bella", "verse": "am_michael",
    "marin": "af_kore", "cedar": "am_eric",
}

# response_format → (container, codec, Content-Type). pcm is raw 16-bit little-endian
# mono at 24 kHz, as in the OpenAI API.
SPEECH_FORMATS = {
    "wav": (None, None, "audio/wav"),
    "pcm": (None, None, "audio/pcm"),
    "mp3": ("mp3", "libmp3lame", "audio/mpeg"),
    "opus": ("ogg", "libopus", "audio/ogg"),
    "aac": ("adts", "aac", "audio/aac"),
    "flac": ("flac", "flac", "audio/flac"),
}


def encode_speech(wav: bytes, fmt: str) -> bytes:
    """Re-encode a 24 kHz mono PCM WAV as `fmt` (a SPEECH_FORMATS key)."""
    fmt = fmt.lower()
    if fmt == "wav":
        return wav
    import wave
    with wave.open(io.BytesIO(wav), "rb") as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    if fmt == "pcm":
        return pcm
    import av
    import numpy as np
    container, codec, _ = SPEECH_FORMATS[fmt]
    samples = np.frombuffer(pcm, dtype="<i2").reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    frame.sample_rate = rate
    out_rate = 48000 if codec == "libopus" else rate
    buf = io.BytesIO()
    with av.open(buf, "w", format=container) as c:
        st = c.add_stream(codec, rate=out_rate, layout="mono")
        if codec in ("libmp3lame", "aac"):
            st.bit_rate = 64000
        rs = av.AudioResampler(format=st.codec_context.format.name, layout="mono", rate=out_rate)
        for f in rs.resample(frame) + (rs.resample(None) or []):
            for p in st.encode(f):
                c.mux(p)
        for p in st.encode(None):
            c.mux(p)
    return buf.getvalue()


def pcm16_wav(samples, rate: int) -> bytes:
    """Encode float samples in [-1, 1] as a mono 16-bit PCM WAV.

    Written with the standard-library `wave` module rather than soundfile: the output
    format is fixed and simple, and the gateway measures exactly this format, so there
    is no reason to carry a native dependency for it.
    """
    import wave
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()

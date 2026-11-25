"""OpenAI-compatible REST API server for inference with runtime model switching."""

import asyncio
import contextvars
import hashlib
import hmac
import json
import logging
import time
import uuid

import base64

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from .engine import EngineState, GenerateResult, InferenceEngine
from .multimodal import parse_content
from .receipt import build_receipt, get_signer
from .audio_meta import audio_duration
from .speech_engine import SPEECH_FORMATS, Interrupted, billable_characters, billable_seconds, encode_speech

logger = logging.getLogger(__name__)


_tokenizers: dict = {}


def _content_text(content) -> str:
    """Best-effort text extraction from a message `content` (str | list). Used
    only for affinity-key hashing and the no-template bare join — never for
    the template itself."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _chat_prompt(engine, messages, enable_thinking: bool = False):
    """Render the conversation with the model's own chat template.

    Returns (prompt, images): `prompt` is the rendered text (image parts become
    template markers); `images` is the ordered list of decoded PIL images to hand
    vLLM out-of-band (empty for text-only). Pixels never enter the token path —
    receipts keep hashing the rendered text + response exactly as before.

    The bare "role: content" join below is kept only for models that ship no
    template. Feeding that bare transcript to an instruction-tuned model makes
    it treat the input as text to continue — it completes the user's sentence
    and invents its own turn markers. Qwen2.5 masked this by answering anyway;
    Qwen3 additionally exposes its raw <think> block, so the web user saw
    template noise plus the model's internal reasoning.
    """
    path = getattr(engine, "current_model", "") or ""
    tok = _tokenizers.get(path)
    if tok is None and path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception:
            logger.warning("no tokenizer loadable for %s; falling back to bare prompt", path)
            tok = False  # negative-cache: do not retry on every request
        _tokenizers[path] = tok

    rendered = []
    images = []
    for m in messages:
        parsed = parse_content(m.content)
        if parsed.has_image:
            images.extend(parsed.images)
        # "developer" is the OpenAI name for system instructions on newer models;
        # chat templates only know "system".
        role = "system" if m.role == "developer" else m.role
        rendered.append({"role": role, "content": parsed.rendered})

    if tok and getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(
            rendered,
            tokenize=False,
            add_generation_prompt=True,
            # Default OFF: reasoning models emit a <think> block that multiplies
            # latency and bills, so direct answers unless the client opts in
            # (per-request enable_thinking). Templates without this variable
            # ignore it either way.
            enable_thinking=enable_thinking,
        )
        return prompt, images
    return "\n".join(f"{'system' if m.role == 'developer' else m.role}: {_content_text(m.content)}"
                     for m in messages), images


class _ThinkSplitter:
    """Split a thinking-model response "<think>REASONING</think>ANSWER" into
    (reasoning_delta, content_delta) pairs, tolerating tags that arrive split
    across stream deltas. Instantiated ONLY when the client asked for thinking:
    with the flag off the raw text passes through untouched (default JSON
    responses additionally carry `reasoning_content: null` from the response
    model — semantically identical; receipts hash the raw text either way).

    The receipt hash is always computed over the RAW generated text — the split
    is presentation only, so billing attestation is unaffected.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, pre_opened: bool = False, open_tag: str = None, close_tag: str = None):
        # gpt-oss (harmony) speaks a different dialect: after special tokens are
        # stripped the reasoning channel reads "analysis<REASONING>assistantfinal
        # <ANSWER>" — same machine, different anchors.
        if open_tag:
            self.OPEN = open_tag
        if close_tag:
            self.CLOSE = close_tag
        self._buf = ""
        # Newer Qwen templates (*-2507, 3.8) pre-fill "<think>\n" INSIDE the
        # generation prompt, so the generated text starts straight in the
        # reasoning and only ever contains the closing tag. The caller detects
        # that from the rendered prompt and starts us in the think phase.
        self._phase = "think" if pre_opened else "seek"  # seek → think → content
        self._reasoning_started = False  # swallow the newline right after <think>
        self._content_started = False    # swallow the newline(s) right after </think>

    def feed(self, delta: str) -> tuple[str, str]:
        reasoning, content = "", ""
        self._buf += delta
        if self._phase == "seek":
            probe = self._buf.lstrip()
            if not probe:
                return "", ""
            if probe.startswith(self.OPEN):
                self._buf = probe[len(self.OPEN):]
                self._phase = "think"
            elif self.OPEN.startswith(probe):
                return "", ""  # still might become the tag; keep buffering
            else:
                content, self._buf = self._buf, ""
                self._phase = "content"
                return "", content
        if self._phase == "think":
            if not self._reasoning_started:
                self._buf = self._buf.lstrip("\n")
            i = self._buf.find(self.CLOSE)
            if i >= 0:
                reasoning = self._buf[:i].rstrip("\n")
                self._buf = self._buf[i + len(self.CLOSE):].lstrip("\n")
                self._phase = "content"
                if reasoning:
                    self._reasoning_started = True
            else:
                # Emit all but a tail that could be a split closing tag. Trailing
                # newlines stay buffered too: they may turn out to precede the
                # closing tag, and once emitted they cannot be taken back.
                keep = len(self.CLOSE) - 1
                if len(self._buf) > keep:
                    emit = self._buf[:-keep]
                    stripped = emit.rstrip("\n")
                    self._buf = emit[len(stripped):] + self._buf[-keep:]
                    reasoning = stripped
                if reasoning:
                    self._reasoning_started = True
                return reasoning, ""
        if self._phase == "content":
            chunk = self._buf
            self._buf = ""
            if not self._content_started:
                chunk = chunk.lstrip("\n")
            if chunk:
                self._content_started = True
            content += chunk
        return reasoning, content

    def flush(self) -> tuple[str, str]:
        """Return whatever is still buffered (truncated generations)."""
        buf, self._buf = self._buf, ""
        if self._phase == "think":
            # A truncated generation may end mid-way through the closing tag;
            # that half-written tag is markup, not reasoning.
            for k in range(min(len(buf), len(self.CLOSE) - 1), 0, -1):
                if self.CLOSE.startswith(buf[-k:]):
                    buf = buf[:-k]
                    break
            return buf.rstrip("\n"), ""
        if not self._content_started:
            buf = buf.lstrip("\n")
        return "", buf


def _split_completed(text: str, splitter, want_thinking: bool) -> tuple[str | None, str]:
    """Split a finished generation into (reasoning_content, content).

    Reasoning is withheld from a request that did not ask for it — except when
    the generation ran out of budget inside the reasoning channel and never
    produced an answer. Withholding it there would return an entirely empty
    response for tokens the caller is billed for, so it is surfaced instead;
    finish_reason ("length") tells them why it stopped there.
    """
    if splitter is None:
        return None, text
    r1, c1 = splitter.feed(text)
    r2, c2 = splitter.flush()
    reasoning, content = r1 + r2, c1 + c2
    if want_thinking:
        return (reasoning or None), content
    return (reasoning if reasoning and not content else None), content


def _engine_session_key(http_request, messages=None, prompt=None):
    """Derive an engine-affinity key so a multi-instance worker can route the
    same conversation back to the same engine (reusing its prefix cache).

    Priority: explicit X-Session-Id header → hash of the first two messages
    (the stable conversation prefix) → hash of a completion prompt. Returns
    None when there is nothing stable to pin on (then routing stays balanced).
    """
    sid = http_request.headers.get("x-session-id") if http_request is not None else None
    if sid:
        return "h:" + sid
    parts = []
    if messages:
        for m in messages[:2]:
            content = _content_text(m.content)
            parts.append(f"{m.role}:{content}")
    elif isinstance(prompt, str) and prompt:
        parts.append("p:" + prompt)
    if not parts:
        return None
    return "m:" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def _is_context_length_error(e: Exception) -> bool:
    """True if the error is "prompt longer than the model's context window".
    vLLM raises a ValueError for this; we map it to a clean 400 instead of a 500,
    since it is a client mistake (too-long input), not a server fault."""
    msg = str(e).lower()
    return ("context length" in msg or "longer than the maximum" in msg
            or "maximum model length" in msg or "max_model_len" in msg
            or ("prompt" in msg and "too long" in msg))


def _is_multimodal_limit_error(e: Exception) -> bool:
    """True if vLLM rejected the request for carrying more images than the engine's
    per-prompt cap ("At most N image(s) may be provided in one prompt"). The API layer
    enforces the cap itself before generate(); this catches any path that slipped by so
    the client still gets a 400, not a 500."""
    msg = str(e).lower()
    return "may be provided in one prompt" in msg or ("at most" in msg and "image" in msg)


class ChatMessage(BaseModel):
    role: str
    # Text (str) for text-only, or an OpenAI-style multimodal array of parts
    # ({"type": "text"|"image_url", ...}) for VLM requests.
    content: str | list
    # Present on assistant RESPONSES when thinking was requested; ignored (and
    # never re-rendered into the prompt) when a client echoes it back in history.
    reasoning_content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, gt=0)  # reject <= 0 (invalid for vLLM)
    # Range-validate sampling params so an out-of-range value is rejected at the edge
    # (HTTP 422) instead of flowing into vLLM SamplingParams and raising a ValueError
    # that surfaced as a bare 500 / mid-stream crash (audit MEDIUM fix).
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False
    # Thinking mode (opt-in, default OFF — answers, latency and bills stay as
    # they are unless a client asks). `enable_thinking` is the direct switch;
    # `chat_template_kwargs: {"enable_thinking": true}` is accepted as the
    # OpenAI-ecosystem spelling of the same thing. Models whose template has no
    # such variable (gpt-oss, Qwen2.5) silently ignore it.
    enable_thinking: bool = False
    chat_template_kwargs: dict | None = None
    # GATEWAY-INTERNAL (B2 stream resume): text already delivered to the client by a
    # previous worker whose generation was interrupted (mining yield / crash). It is
    # appended VERBATIM to the rendered prompt so the engine continues exactly where
    # the stream broke. The gateway rejects this field on client requests (400) —
    # only the gateway itself may set it. Advertised via /health features.
    om_continuation: str | None = None
    # OpenAI parameters with a direct equivalent. max_completion_tokens is the newer
    # spelling of max_tokens; n > 1 is refused (one choice per request);
    # response_format json_object / json_schema constrain decoding to valid JSON.
    max_completion_tokens: int | None = Field(default=None, gt=0)
    n: int = Field(default=1, ge=1)
    seed: int | None = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    response_format: dict | None = None

    @model_validator(mode="after")
    def _openai_aliases(self):
        if self.max_completion_tokens is not None and "max_tokens" not in self.model_fields_set:
            self.max_tokens = self.max_completion_tokens
        return self


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    max_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    # GATEWAY-INTERNAL (B2 stream resume): see ChatCompletionRequest.om_continuation.
    om_continuation: str | None = None
    n: int = Field(default=1, ge=1)
    seed: int | None = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    text: str | None = None
    finish_reason: str = "stop"


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0  # Prompt tokens served from the prefix cache (OpenAI-compatible)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails = Field(default_factory=PromptTokensDetails)


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# --- image generation (OpenAI images API shape) -----------------------------------
#
# The response carries the produced bytes as base64. `usage.images` is the BILLABLE
# amount for this request; it is also what the gateway cross-checks the receipt's
# quantity against, so it must state the true count of returned images.

class ImageGenerationRequest(BaseModel):
    model: str = "default"
    prompt: str
    n: int = Field(default=1, ge=1)
    size: str | None = "1024x1024"   # "WxH" or "auto"; parsed into height/width
    # Denoising steps; None = the engine's default for the loaded model (default_steps).
    steps: int | None = Field(default=None, ge=1)
    guidance: float | None = None
    seed: int | None = None
    # "b64_json". The worker has no public file store; the gateway serves "url" by
    # requesting b64_json here and hosting the bytes itself.
    response_format: str | None = "b64_json"
    output_format: str | None = None          # png (default) / jpeg / webp
    output_compression: int | None = Field(default=None, ge=0, le=100)


class ImageObject(BaseModel):
    b64_json: str


class ImageUsage(BaseModel):
    images: int = 0


class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageObject]
    usage: ImageUsage = Field(default_factory=ImageUsage)


class SpeechRequest(BaseModel):
    model: str = "default"
    input: str
    # An installed voice id, or an OpenAI voice name (mapped; speech_engine.OPENAI_VOICES).
    voice: str | None = None
    response_format: str | None = "wav"   # wav / mp3 / opus / aac / flac / pcm
    speed: float = 1.0                    # 0.25 – 4.0
    # Accepted for OpenAI compatibility; this engine has no prompt-controlled style.
    instructions: str | None = None
    stream_format: str | None = None      # "audio" (default); "sse" is not offered


# Image sizes a 10 GB card renders with SDXL: each side a multiple of 8 (the latent
# stride) between these bounds. A 4096x4096 request ran the card out of memory (a 500,
# and the fragmented card failed the next renders too); "abc" silently became 1024x1024.
IMAGE_SIDE_MIN, IMAGE_SIDE_MAX = 256, 1024


def _parse_size(size: str | None) -> tuple[int, int]:
    """Parse "WxH" into (width, height); ValueError (→ 400) for anything else.

    "auto" (or no size) is 1024x1024. A size with a side above IMAGE_SIDE_MAX — the
    1792x1024 / 1536x1024 family OpenAI clients send — is scaled down to fit, keeping
    its aspect ratio, each side rounded to a multiple of 8.
    """
    if size is None or str(size).strip().lower() in ("", "auto"):
        return IMAGE_SIDE_MAX, IMAGE_SIDE_MAX
    try:
        w, sep, h = str(size).lower().partition("x")
        width, height = int(w), int(h)
    except ValueError:
        width = height = 0
        sep = ""
    if sep and width % 8 == 0 and height % 8 == 0 and max(width, height) > IMAGE_SIDE_MAX:
        scale = IMAGE_SIDE_MAX / max(width, height)
        width = max(8, int(round(width * scale / 8)) * 8)
        height = max(8, int(round(height * scale / 8)) * 8)
    if not sep or not (IMAGE_SIDE_MIN <= width <= IMAGE_SIDE_MAX and IMAGE_SIDE_MIN <= height <= IMAGE_SIDE_MAX) \
            or width % 8 or height % 8:
        raise ValueError(f"size must be \"WxH\" with each side a multiple of 8 between "
                         f"{IMAGE_SIDE_MIN} and {IMAGE_SIDE_MAX}, or \"auto\" (got {size!r})")
    return width, height


IMAGE_OUTPUT_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}


def _encode_image(png: bytes, output_format: str | None, compression: int | None) -> bytes:
    """Re-encode a rendered PNG as `output_format` (png / jpeg / webp)."""
    fmt = (output_format or "png").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt == "png":
        return png
    from PIL import Image
    import io as _io
    img = Image.open(_io.BytesIO(png))
    if fmt == "jpeg":
        img = img.convert("RGB")
    buf = _io.BytesIO()
    quality = 100 - int(compression) if compression is not None else 90
    img.save(buf, format=IMAGE_OUTPUT_FORMATS[fmt], quality=max(1, min(100, quality)))
    return buf.getvalue()


def _sampling_extras(request) -> dict:
    """OpenAI sampling parameters beyond max_tokens / temperature / top_p / stop, as
    engine keyword arguments. Raises HTTPException(400) for values that cannot be
    honoured, so a client never receives silently different output from what it asked."""
    if getattr(request, "n", 1) != 1:
        raise HTTPException(status_code=400, detail="n must be 1: one choice per request")
    extras = {}
    if getattr(request, "seed", None) is not None:
        extras["seed"] = int(request.seed)
    for name in ("presence_penalty", "frequency_penalty"):
        v = getattr(request, name, 0.0) or 0.0
        if v:
            extras[name] = float(v)
    rf = getattr(request, "response_format", None)
    if rf:
        kind = rf.get("type") if isinstance(rf, dict) else None
        if kind == "json_object":
            extras["structured"] = {"json_object": True}
        elif kind == "json_schema":
            js = rf.get("json_schema") or {}
            schema = js.get("schema") if isinstance(js, dict) else None
            if not isinstance(schema, dict):
                raise HTTPException(status_code=400,
                                    detail="response_format json_schema needs json_schema.schema (an object)")
            extras["structured"] = {"json": schema}
        elif kind != "text":
            raise HTTPException(status_code=400,
                                detail="response_format type must be \"text\", \"json_object\" or \"json_schema\"")
    return extras


# A decode may run past the declared length by the codec's priming/padding (AAC adds up
# to ~2048 samples) and the decoder's resampling; more than this means the header lies.
DURATION_SLACK_SEC = 0.25
DURATION_SLACK_RATIO = 0.01


def _ts(t: float, sep: str) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}{sep}{ms:03d}"


def subtitles(segments: list, fmt: str) -> str:
    """SRT or WebVTT text from transcription segments."""
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines += [f"{_ts(seg['start'], '.')} --> {_ts(seg['end'], '.')}", seg["text"].strip(), ""]
        return "\n".join(lines)
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{_ts(seg['start'], ',')} --> {_ts(seg['end'], ',')}", seg["text"].strip(), ""]
    return "\n".join(lines)


def _model_matches(current: str, requested: str) -> bool:
    """True if the currently-loaded model IS the requested one. Compares the final
    path/id component EXACTLY (FOC caches a model at '<dir>/Org--Name'); a substring/
    endswith check would false-match a model whose name is merely a suffix of another
    and silently serve the wrong model (audit MEDIUM fix)."""
    if not current:
        return False
    if current == requested:
        return True
    req_name = requested.replace("/", "--").replace("\\", "/").rstrip("/").split("/")[-1]
    cur_name = current.replace("\\", "/").rstrip("/").split("/")[-1]
    return cur_name == req_name


def _retry_after_for_state(state: EngineState) -> str:
    """Return Retry-After header value based on engine state."""
    if state == EngineState.LOADING:
        return "15"   # Model is reloading, will be back soon
    return "60"       # UNLOADING/PAUSED — mining may take minutes


def create_app(engine: InferenceEngine, model_manager=None,
               api_token: str = "", max_tokens_limit: int = 0,
               token_store=None) -> FastAPI:
    """Create the FastAPI application with the given inference engine.

    Args:
        engine: The inference engine (or engine pool).
        model_manager: Optional ModelManager for runtime model switching.
            When provided, requests with a non-default 'model' field will
            trigger automatic model download and switching.
        api_token: If non-empty, /v1/* endpoints require
            'Authorization: Bearer <api_token>'. /health stays open.
        max_tokens_limit: If > 0, reject requests asking for more output tokens.
        token_store: Optional GatewayTokenStore of dynamic per-worker tokens
            issued by SP self-registration (learned from the scheduler over
            gRPC). Accepted IN ADDITION to api_token; auth is enforced as soon
            as either source is configured.
    """
    app = FastAPI(title="OpenModel Inference API")

    # One switch at a time PER ENGINE (pool). A single process-wide lock made a switch in
    # one pool wait for a reload in another: on a test worker an ASR small -> medium switch (5 s)
    # waited 42 s behind a 32B -> VL reload of the text pool. Different pools may switch
    # at once — vLLM core spawns are serialized by VLLMEngine._create_lock and in-process
    # loads by torch_globals.
    _switch_locks: dict[int, asyncio.Lock] = {}

    def _switch_lock(eng) -> asyncio.Lock:
        lock = _switch_locks.get(id(eng))
        if lock is None:
            lock = _switch_locks[id(eng)] = asyncio.Lock()
        return lock
    # Requests that obtained their model and have not finished generating, per engine.
    # A switch of that engine waits for them: without this, a request queued behind a
    # switch could switch the pool BACK before the request that triggered the first
    # switch ever ran — each of two interleaved requests for two models then found its
    # model gone and answered 503 (found on a test worker with 32B and VL-8B in one pool).
    _in_use: dict[int, int] = {}
    _in_use_cond = asyncio.Condition()
    _switching: set[int] = set()   # engines whose switch has passed the in-use wait

    # The engine a request holds, set by _ensure_model_loaded and released by the
    # middleware below once the response (including a whole SSE stream) has been sent.
    _held = contextvars.ContextVar("held_engine", default=None)

    async def _hold(eng) -> None:
        async with _in_use_cond:
            _in_use[id(eng)] = _in_use.get(id(eng), 0) + 1
        box = _held.get()
        if box is not None:
            box.append(eng)

    async def _release(eng) -> None:
        async with _in_use_cond:
            _in_use[id(eng)] = max(0, _in_use.get(id(eng), 0) - 1)
            _in_use_cond.notify_all()

    @app.middleware("http")
    async def _release_held_engines(request: Request, call_next):
        box: list = []
        token = _held.set(box)
        try:
            response = await call_next(request)
        except BaseException:
            for eng in box:
                await _release(eng)
            raise
        finally:
            _held.reset(token)
        if not box:
            return response
        if isinstance(response, StreamingResponse) or hasattr(response, "body_iterator"):
            inner = response.body_iterator

            async def body():
                try:
                    async for chunk in inner:
                        yield chunk
                finally:
                    for eng in box:
                        await _release(eng)
            response.body_iterator = body()
            return response
        for eng in box:
            await _release(eng)
        return response

    def _require_auth(authorization: str = Header(default="")):
        """Bearer-token gate for /v1/* endpoints.

        No-op only when NEITHER a static api_token NOR a dynamic token store is
        configured (trusted-LAN mode). With self-registration, the accepted set
        is {static token} ∪ {last two dynamic tokens} — the previous dynamic
        token stays valid for one rotation so in-flight gateway requests don't
        401 during the ≤3s handover.
        """
        dynamic = token_store.accepted() if token_store is not None else []
        if not api_token and not dynamic:
            return
        candidates = ([api_token] if api_token else []) + dynamic
        for tok in candidates:
            # constant-time compare; reject anything not exactly "Bearer <token>"
            if hmac.compare_digest(authorization, f"Bearer {tok}"):
                return
        raise HTTPException(status_code=401, detail="invalid or missing Authorization")

    def _check_max_tokens(max_tokens: int):
        if max_tokens_limit > 0 and max_tokens > max_tokens_limit:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens {max_tokens} exceeds limit {max_tokens_limit}",
            )

    def _engine_for(requested_model: str):
        """The engine that serves `requested_model`.

        A multi-pool worker routes to the pool owning the model; a single-engine
        worker is its own (only) engine, so existing behaviour is unchanged. A model
        no pool serves falls back to the primary engine, whose normal switch path
        then decides (and reports) whether it can load it.
        """
        pick = getattr(engine, "engine_for", None)
        if pick is None:
            return engine
        return pick(requested_model) or engine

    async def _ensure_model_loaded(requested_model: str) -> None:
        """Switch model if the request asks for a different one.

        Args:
            requested_model: The model ID from the request's 'model' field.
        """
        if model_manager is None:
            return  # No model manager — single-model mode

        # "default" means use whatever is currently loaded
        if requested_model == "default":
            return

        # Already loaded? Exact final-component match (see _model_matches): handles
        # both current="/models/Qwen--Qwen2.5-1.5B-Instruct" vs requested=
        # "Qwen/Qwen2.5-1.5B-Instruct", without the endswith false-match bug.
        # On a multi-pool worker any resident pool counts as "already loaded".
        target = _engine_for(requested_model)
        current = target.current_model if hasattr(target, 'current_model') else ""
        if _model_matches(current, requested_model):
            # Resident: hold it and go — unless THIS pool's switch has already begun
            # (then wait for it below). A switch of another pool does not concern this
            # request: on a test worker a speech request otherwise waited out a 60 s text reload.
            await _hold(target)
            if id(target) not in _switching:
                return
            await _release(target)

        async with _switch_lock(target):
            # Double-check after acquiring lock
            target = _engine_for(requested_model)
            current = target.current_model if hasattr(target, 'current_model') else ""
            if _model_matches(current, requested_model):
                await _hold(target)
                return

            # Wait for requests still using this engine's current model to finish.
            async with _in_use_cond:
                await _in_use_cond.wait_for(lambda: _in_use.get(id(target), 0) == 0)
                _switching.add(id(target))
            logger.info("model switch requested: %s -> %s", current, requested_model)

            try:
                # Ensure model is available locally (download from FOC if needed)
                model_path = await model_manager.ensure_model(requested_model)
                from src.model.quant_identity import check_quant_identity
                check_quant_identity(requested_model, model_path)
                # A model of another engine kind (an image model listed for a text
                # pool) is refused here, before the pool's current model is unloaded.
                from src.model.model_kind import check_engine_fits
                kind = getattr(target, "engine_kind", None)
                if kind:
                    check_engine_fits(kind, requested_model, model_path)
                # Switch engine to new model
                await engine.switch_model(model_path)
            finally:
                _switching.discard(id(target))
            logger.info("model switch complete, now serving: %s", requested_model)
            await _hold(_engine_for(requested_model))

    @app.get("/health")
    async def health():
        status = engine.status()
        result = {
            "status": "ok" if engine.is_available() else "unavailable",
            "engine_state": status.state.value,
            "active_requests": status.active_requests,
            "loaded_model": status.loaded_model,
            # Capability advertisement, polled by the gateway. "continuation" = this
            # worker understands om_continuation (B2 stream resume); the gateway only
            # resumes interrupted streams onto workers that advertise it, so mixed old/
            # new worker fleets degrade gracefully instead of double-generating.
            # "receipt" = signs per-request inference receipts (A1); pubkey below.
            # "vision" = the LOADED model accepts image parts in chat messages
            # (OpenAI multimodal content arrays); absent for text-only models.
            # "image_gen" = the loaded model GENERATES images; the gateway routes
            # /v1/images/generations only to workers advertising it. Mutually exclusive
            # with "vision" in practice (one modality per GPU), and deliberately a
            # separate feature rather than a reuse of "vision": reading and producing
            # images are different capabilities and must not be routed interchangeably.
            # "asr" / "tts" = the worker serves /v1/audio/transcriptions and
            # /v1/audio/speech. On a multi-pool worker the flags are the UNION of its
            # pools; per-pool flags are under multi_gpu.pools.
            "features": ["continuation", "receipt"]
                        + (["vision"] if getattr(engine, "is_vlm", False) else [])
                        + (["image_gen"] if getattr(engine, "is_image_gen", False) else [])
                        + (["asr"] if getattr(engine, "is_asr", False) else [])
                        + (["tts"] if getattr(engine, "is_tts", False) else []),
            "receipt_pubkey": get_signer().pubkey_hex,
        }
        # Every resident model. A multi-pool worker has several (text, image, speech);
        # the gateway routes a request to any of them as "already loaded". loaded_model
        # above stays the primary one for gateways that predate this field.
        lm = getattr(engine, "loaded_models", None)
        result["loaded_models"] = list(lm) if lm else [status.loaded_model]
        # Request limits of a single-engine worker (a multi-pool worker reports them per
        # pool under multi_gpu.pools[].limits). Two workers serving one model can be
        # configured differently — 1 image / 4k tokens on a 10 GB card, 32 / 8k on a
        # 4-card pool — and the gateway uses these to route a request where it fits.
        if not hasattr(engine, "pools"):
            from .multipool import _engine_limits
            lim = _engine_limits(engine)
            if lim:
                result["limits"] = lim

        if hasattr(engine, "detailed_status"):
            result["multi_gpu"] = engine.detailed_status()

        return result

    @app.get("/v1/models", dependencies=[Depends(_require_auth)])
    async def list_models():
        status = engine.status()
        current = status.loaded_model or "none"

        models = [{"id": current, "object": "model",
                    "owned_by": "openmodel", "loaded": True}]

        # List registered models from FOC registry
        if model_manager and hasattr(model_manager, '_foc') and hasattr(model_manager._foc, '_registry'):
            for model_id in model_manager._foc._registry:
                if not current.endswith(model_id.replace("/", "--")):
                    models.append({"id": model_id, "object": "model",
                                   "owned_by": "openmodel", "loaded": False})

        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_auth)])
    async def chat_completions(request: ChatCompletionRequest, http_request: Request, response: Response):
        _check_max_tokens(request.max_tokens)
        # A1 signed receipts: attest the EXACT bytes this worker was asked to serve.
        _req_sha = hashlib.sha256(await http_request.body()).hexdigest()
        _rid = http_request.headers.get("x-request-id", "")
        # Auto-switch model if needed
        try:
            await _ensure_model_loaded(request.model)
        except Exception as e:
            logger.error("model switch failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Model switch failed: {e}")

        eng = _engine_for(request.model)

        if not eng.is_available():
            status = eng.status()
            state_desc = status.state.value
            if _switch_lock(eng).locked():
                state_desc = "switching model"
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": f"GPU yielded to mining ({state_desc}). Retry later.",
                        "type": "service_unavailable",
                    }
                },
                headers={"Retry-After": _retry_after_for_state(status.state)},
            )

        want_thinking = bool(request.enable_thinking or
                             (request.chat_template_kwargs or {}).get("enable_thinking"))
        try:
            prompt, images = _chat_prompt(eng, request.messages, enable_thinking=want_thinking)
        except ValueError as e:
            # A malformed / oversized / undecodable image part is the client's mistake:
            # fail clean with 400 before the engine is touched (not an unhandled 500).
            raise HTTPException(status_code=400, detail=f"invalid message content: {e}")
        if images and not getattr(eng, "is_vlm", True):
            # The loaded model has no vision encoder: vLLM would reject the
            # multi_modal_data deep inside generate() as a 500. Say so plainly —
            # the client asked a text model to look at a picture.
            raise HTTPException(
                status_code=400,
                detail=f"model {eng.current_model!r} does not accept image input")
        cap = getattr(eng, "max_images_per_prompt", 0) or 0
        if cap and len(images) > cap:
            # Same reason: vLLM enforces limit_mm_per_prompt inside generate() with a
            # bare ValueError (observed live as an unhandled 500). Refuse at the edge.
            raise HTTPException(
                status_code=400,
                detail=f"too many images: {len(images)} provided, this model accepts at most {cap} per request")
        # Template dialects differ: older Qwen3 models GENERATE "<think>…</think>"
        # themselves, newer templates pre-fill the opening tag in the prompt and
        # the generation starts inside the reasoning. Detect which one we got.
        think_pre_opened = want_thinking and prompt.rstrip().endswith(_ThinkSplitter.OPEN)
        # gpt-oss ALWAYS emits its harmony analysis channel — there is no off
        # switch in the template — so its output must be split on every request:
        # default requests keep only the final answer (the analysis is dropped),
        # thinking requests surface it as reasoning_content. Keyed on the model
        # (not a text heuristic): an ordinary answer may legitimately start with
        # the word "analysis".
        is_harmony = "gpt-oss" in (eng.current_model or "")

        def _make_splitter():
            if is_harmony:
                return _ThinkSplitter(open_tag="analysis", close_tag="assistantfinal")
            if want_thinking:
                return _ThinkSplitter(pre_opened=think_pre_opened)
            return None
        # B2 stream resume: continue EXACTLY from the text a previous worker already
        # delivered — vLLM generates the sequel of the prompt string, so appending the
        # delivered prefix verbatim (no separator) resumes mid-sentence seamlessly.
        if request.om_continuation:
            prompt += request.om_continuation

        gen_kwargs = dict(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            **_sampling_extras(request),
        )
        # VLM: hand decoded images to the engine out-of-band (never into the token
        # prompt). Empty list is a no-op for text-only engines.
        if images:
            gen_kwargs["images"] = images
        session_key = _engine_session_key(http_request, messages=request.messages)

        # --- Streaming response ---
        if request.stream:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            want_receipt = http_request.headers.get("x-om-receipt-req") == "1"

            async def sse_generator():
                gen_parts = []
                final_usage = None
                splitter = _make_splitter()
                # Reasoning withheld from a default request, kept only so that a
                # stream that never reaches an answer can hand it back at the
                # end instead of delivering nothing at all.
                withheld_reasoning = []
                sent_content = False
                try:
                    async for chunk in eng.generate_stream(prompt, session_key=session_key, **gen_kwargs):
                        if chunk.finish_reason is not None:
                            # Final chunk: send finish_reason + usage
                            data = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": chunk.finish_reason}],
                                "usage": {
                                    "prompt_tokens": chunk.prompt_tokens,
                                    "completion_tokens": chunk.completion_tokens,
                                    "total_tokens": chunk.prompt_tokens + chunk.completion_tokens,
                                    "prompt_tokens_details": {"cached_tokens": chunk.cached_tokens},
                                },
                            }
                        else:
                            delta = {"content": chunk.text_delta}
                            if splitter is not None:
                                r, c = splitter.feed(chunk.text_delta)
                                if chunk.text_delta:
                                    gen_parts.append(chunk.text_delta)
                                if not r and not c:
                                    continue  # buffering a possibly-split tag
                                delta = {}
                                if r:
                                    if want_thinking:
                                        delta["reasoning_content"] = r
                                    else:
                                        withheld_reasoning.append(r)
                                if c:
                                    delta["content"] = c
                                    sent_content = True
                                if not delta:
                                    continue  # harmony analysis withheld on a default request
                            data = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                            }
                        if chunk.finish_reason is not None:
                            if splitter is not None:
                                r, c = splitter.flush()
                                tail = {}
                                if r:
                                    if want_thinking:
                                        tail["reasoning_content"] = r
                                    else:
                                        withheld_reasoning.append(r)
                                if c:
                                    tail["content"] = c
                                    sent_content = True
                                if not sent_content and withheld_reasoning:
                                    # The budget ran out inside the reasoning
                                    # channel: no answer text ever arrived. Send
                                    # the withheld reasoning rather than closing
                                    # a stream that delivered nothing, for tokens
                                    # the caller is billed for.
                                    tail["reasoning_content"] = "".join(withheld_reasoning)
                                if tail:
                                    frame = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": request.model,
                                        "choices": [{"index": 0, "delta": tail, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(frame)}\n\n"
                            final_usage = (chunk.prompt_tokens, chunk.completion_tokens, chunk.cached_tokens)
                        elif splitter is None and chunk.text_delta:
                            gen_parts.append(chunk.text_delta)
                        yield f"data: {json.dumps(data)}\n\n"
                    # A1: attest the completed stream (delivered text + reported usage)
                    # as a dedicated event right before [DONE]; the gateway captures and
                    # strips it (never reaches the client).
                    if want_receipt and final_usage is not None:
                        text = "".join(gen_parts)
                        rcpt = build_receipt(_rid, request.model, _req_sha,
                                             hashlib.sha256(text.encode()).hexdigest(),
                                             final_usage[0], final_usage[1], final_usage[2])
                        yield f"data: {json.dumps({'om_receipt': rcpt})}\n\n"
                    yield "data: [DONE]\n\n"
                except (RuntimeError, ValueError) as e:
                    # Emit the error frame AND a terminating [DONE], matching the
                    # success path so strict SSE clients always see a clean end
                    # (audit LOW fix). ValueError is caught too in case a bad
                    # sampling value slips past Field validation.
                    error_data = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        # --- Non-streaming response ---
        try:
            result = await eng.generate(prompt, session_key=session_key, **gen_kwargs)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            # Prompt exceeds the model's context window → client error, clean 400.
            if _is_context_length_error(e):
                raise HTTPException(status_code=400, detail=f"context length exceeded: {e}")
            if _is_multimodal_limit_error(e):
                raise HTTPException(status_code=400, detail=f"too many images: {e}")
            raise

        if isinstance(result, GenerateResult):
            text = result.text
            finish_reason = result.finish_reason
            usage = Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=result.cached_tokens),
            )
        else:
            text = result
            finish_reason = "stop"
            usage = Usage()

        # A1: attach the signed receipt as a response header (whole response is
        # buffered for non-streaming, so headers are still writable here).
        rcpt = build_receipt(_rid, request.model, _req_sha,
                             hashlib.sha256(text.encode()).hexdigest(),
                             usage.prompt_tokens, usage.completion_tokens,
                             usage.prompt_tokens_details.cached_tokens)
        response.headers["X-OM-Receipt"] = base64.b64encode(
            json.dumps(rcpt, separators=(",", ":")).encode()).decode()

        # Thinking split is presentation-only: the receipt above hashed the RAW
        # text; with the flag off the text is untouched (the JSON does gain a
        # null reasoning_content field from the response model).
        reasoning_out, content_out = _split_completed(text, _make_splitter(), want_thinking)

        return CompletionResponse(
            model=request.model,
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content=content_out,
                                        reasoning_content=reasoning_out),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    @app.post("/v1/completions", dependencies=[Depends(_require_auth)])
    async def completions(request: CompletionRequest, http_request: Request, response: Response):
        _req_sha = hashlib.sha256(await http_request.body()).hexdigest()
        _rid = http_request.headers.get("x-request-id", "")
        _check_max_tokens(request.max_tokens)
        # Auto-switch model if needed
        try:
            await _ensure_model_loaded(request.model)
        except Exception as e:
            logger.error("model switch failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Model switch failed: {e}")

        eng = _engine_for(request.model)

        if not eng.is_available():
            status = eng.status()
            state_desc = status.state.value
            if _switch_lock(eng).locked():
                state_desc = "switching model"
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": f"GPU yielded to mining ({state_desc}). Retry later.",
                        "type": "service_unavailable",
                    }
                },
                headers={"Retry-After": _retry_after_for_state(status.state)},
            )

        # B2 stream resume: append the already-delivered prefix verbatim (see chat path).
        prompt = request.prompt
        if request.om_continuation:
            prompt += request.om_continuation

        try:
            result = await eng.generate(
                prompt,
                session_key=_engine_session_key(http_request, prompt=request.prompt),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                **_sampling_extras(request),
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            if _is_context_length_error(e):
                raise HTTPException(status_code=400, detail=f"context length exceeded: {e}")
            if _is_multimodal_limit_error(e):
                raise HTTPException(status_code=400, detail=f"too many images: {e}")
            raise

        if isinstance(result, GenerateResult):
            text = result.text
            finish_reason = result.finish_reason
            usage = Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=result.cached_tokens),
            )
        else:
            text = result
            finish_reason = "stop"
            usage = Usage()

        rcpt = build_receipt(_rid, request.model, _req_sha,
                             hashlib.sha256(text.encode()).hexdigest(),
                             usage.prompt_tokens, usage.completion_tokens,
                             usage.prompt_tokens_details.cached_tokens)
        response.headers["X-OM-Receipt"] = base64.b64encode(
            json.dumps(rcpt, separators=(",", ":")).encode()).decode()

        return CompletionResponse(
            object="text_completion",
            model=request.model,
            choices=[Choice(text=text, finish_reason=finish_reason)],
            usage=usage,
        )

    @app.post("/v1/images/generations", dependencies=[Depends(_require_auth)])
    async def images_generations(request: ImageGenerationRequest, http_request: Request, response: Response):
        """Render images and return them base64-encoded, OpenAI images API shape.

        Billing is per image: the response's `usage.images` states how many were
        produced, and the signed receipt attests that count plus the sha256 of the
        returned bytes. The gateway re-derives both from the response before charging.
        """
        # A1: the receipt attests the exact request bytes this worker was asked to serve.
        _req_sha = hashlib.sha256(await http_request.body()).hexdigest()
        _rid = http_request.headers.get("x-request-id", "")

        try:
            await _ensure_model_loaded(request.model)
        except Exception as e:
            logger.error("model switch failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Model switch failed: {e}")

        eng = _engine_for(request.model)

        if getattr(eng, "is_image_gen", False) is False:
            # A text or vision worker cannot generate images. Say so plainly instead of
            # letting the request reach an engine that will fail obscurely.
            raise HTTPException(
                status_code=400,
                detail=f"model {eng.current_model!r} does not generate images")

        if not eng.is_available():
            status = eng.status()
            return JSONResponse(
                status_code=503,
                content={"error": {
                    "message": f"GPU yielded to mining ({status.state.value}). Retry later.",
                    "type": "service_unavailable"}},
                headers={"Retry-After": _retry_after_for_state(status.state)},
            )

        cap = getattr(eng, "max_images_per_request", 0) or 0
        if cap and request.n > cap:
            # Reject at the edge (400) rather than let the engine raise mid-render.
            raise HTTPException(
                status_code=400,
                detail=f"too many images: {request.n} requested, at most {cap}",
            )

        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")
        max_steps = getattr(eng, "max_steps", 0) or 0
        if max_steps and request.steps and request.steps > max_steps:
            raise HTTPException(status_code=400,
                                detail=f"too many steps: {request.steps} requested, at most {max_steps}")
        if (request.response_format or "b64_json") != "b64_json":
            raise HTTPException(status_code=400, detail="only response_format \"b64_json\" is supported")
        out_fmt = (request.output_format or "png").lower()
        if out_fmt not in ("png", "jpeg", "jpg", "webp"):
            raise HTTPException(status_code=400, detail="output_format must be png, jpeg or webp")
        try:
            width, height = _parse_size(request.size)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            result = await eng.generate(
                request.prompt,
                n=request.n, steps=request.steps, guidance=request.guidance,
                height=height, width=width, seed=request.seed,
            )
        except ValueError as e:
            # Engine-side limit / bad-parameter errors are the client's mistake.
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            # A render stopped by a yield returns nothing and bills nothing; 503 tells
            # the gateway to retry elsewhere, the same answer as a yield before the start.
            if "yielded to mining" not in str(e) and "out of GPU memory" not in str(e):
                raise
            status = eng.status()
            return JSONResponse(
                status_code=503,
                content={"error": {"message": str(e), "type": "service_unavailable"}},
                headers={"Retry-After": _retry_after_for_state(status.state)},
            )

        images = getattr(result, "images", None) or []
        if not images:
            raise HTTPException(status_code=500, detail="image generation produced no output")
        if out_fmt != "png":
            # Re-encoded BEFORE hashing: the receipt and the gateway hash the bytes delivered.
            images = [_encode_image(img, out_fmt, request.output_compression) for img in images]

        # The asset digest covers the concatenated PNG bytes in order. Derived HERE, from
        # the exact bytes that go into the response, so the gateway's independent hash of
        # the same bytes must match — that agreement is the whole point of committing it.
        asset_hash = hashlib.sha256(b"".join(images)).hexdigest()

        # Receipt: asset shape (quantity + asset_hash), not the token shape. The token
        # counts are zero because an image generation produces no tokens; the render
        # TEXT hash is the asset digest, so response_sha256 identifies the pixels.
        rcpt = build_receipt(_rid, request.model, _req_sha, asset_hash,
                             0, 0, 0, quantity=len(images), asset_hash=asset_hash)
        response.headers["X-OM-Receipt"] = base64.b64encode(
            json.dumps(rcpt, separators=(",", ":")).encode()).decode()

        import base64 as _b64
        return ImageGenerationResponse(
            data=[ImageObject(b64_json=_b64.b64encode(img).decode()) for img in images],
            usage=ImageUsage(images=len(images)),
        )

    def _speech_unavailable(eng):
        status = eng.status()
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": f"GPU yielded to mining ({status.state.value}). Retry later.",
                "type": "service_unavailable"}},
            headers={"Retry-After": _retry_after_for_state(status.state)},
        )

    def _attach_receipt(response, rid, model, req_sha, asset_hash, quantity):
        # Asset shape. response_sha256 carries the asset digest, as for images, so the
        # receipt identifies the exact bytes that were billed.
        rcpt = build_receipt(rid, model, req_sha, asset_hash, 0, 0, 0,
                             quantity=quantity, asset_hash=asset_hash)
        response.headers["X-OM-Receipt"] = base64.b64encode(
            json.dumps(rcpt, separators=(",", ":")).encode()).decode()

    async def _audio_to_text(http_request: Request, response: Response, task: str):
        """Shared body of /v1/audio/transcriptions and /v1/audio/translations.

        Billing is per INPUT second, rounded up, minimum 1, from the duration the file
        DECLARES (audio_meta.audio_duration) — the number the gateway measures on the same
        bytes. The decode must not run noticeably longer than that declaration; a file
        whose header understates its length is refused rather than billed short.
        """
        # request_sha256 is over the WHOLE multipart body — the exact bytes the gateway
        # forwarded — exactly as for every other endpoint.
        _req_sha = hashlib.sha256(await http_request.body()).hexdigest()
        _rid = http_request.headers.get("x-request-id", "")
        # Parse the multipart form here rather than through FastAPI's File/Form
        # parameters: those make create_app() itself require python-multipart, so every
        # worker — text-only ones included — would need it just to start.
        try:
            form = await http_request.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"expected multipart/form-data: {e}")
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="missing audio file field \"file\"")
        model = str(form.get("model") or "default")
        language = (form.get("language") or None) if task == "transcribe" else None
        response_format = str(form.get("response_format") or "json").lower()
        if response_format not in ("json", "text", "verbose_json", "srt", "vtt"):
            raise HTTPException(status_code=400,
                                detail="response_format must be json, text, verbose_json, srt or vtt")
        prompt = str(form.get("prompt") or "") or None
        temperature = form.get("temperature")
        try:
            temperature = float(temperature) if temperature not in (None, "") else None
        except ValueError:
            raise HTTPException(status_code=400, detail="temperature must be a number")
        granularities = [str(g) for g in form.getlist("timestamp_granularities[]")] + \
                        [str(g) for g in form.getlist("timestamp_granularities")]
        want_words = "word" in granularities
        stream = str(form.get("stream") or "").lower() == "true"
        try:
            await _ensure_model_loaded(model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model switch failed: {e}")
        eng = _engine_for(model)
        if not getattr(eng, "is_asr", False):
            raise HTTPException(status_code=400,
                                detail=f"model {eng.current_model!r} does not transcribe audio")
        if not eng.is_available():
            return _speech_unavailable(eng)
        audio = await upload.read()
        if not audio:
            raise HTTPException(status_code=400, detail="empty audio file")
        try:
            declared, _fmt = audio_duration(audio)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            if hasattr(eng, "transcribe_full"):
                r = await eng.transcribe_full(audio, language, task=task, prompt=prompt,
                                              temperature=temperature, word_timestamps=want_words)
            else:
                text, dur, lang = await eng.transcribe(audio, language)
                r = {"text": text, "duration": dur, "language": lang, "segments": []}
        except Interrupted:
            # Stopped at a segment boundary for a mining yield: nothing returned, nothing
            # billed; the client retries like any other 503.
            return _speech_unavailable(eng)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            # The engine could not serve it (yielded, or the model call aborted): a
            # server-side 503 the gateway retries elsewhere, not the client's 400.
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            # An undecodable upload surfaces from the decoder as a generic error.
            raise HTTPException(status_code=400, detail=f"could not decode audio: {e}")
        if r["duration"] > declared + DURATION_SLACK_SEC + declared * DURATION_SLACK_RATIO:
            raise HTTPException(status_code=400, detail=(
                f"audio plays {r['duration']:.2f}s but its container declares {declared:.2f}s; "
                "re-encode the file"))

        seconds = billable_seconds(declared)
        _attach_receipt(response, _rid, model, _req_sha,
                        hashlib.sha256(audio).hexdigest(), seconds)
        usage = {"type": "duration", "seconds": seconds}
        text = r["text"]
        headers = dict(response.headers)
        if stream:
            # OpenAI streaming-transcription event shapes, delivered once the text is ready.
            events = [{"type": "transcript.text.delta", "delta": seg["text"]} for seg in r["segments"]]
            events.append({"type": "transcript.text.done", "text": text, "usage": usage})
            body = "".join(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n" for ev in events)
            return Response(content=body, media_type="text/event-stream", headers=headers)
        if response_format == "text":
            return Response(content=text, media_type="text/plain", headers=headers)
        if response_format in ("srt", "vtt"):
            return Response(content=subtitles(r["segments"], response_format),
                            media_type="text/plain", headers=headers)
        if response_format == "verbose_json":
            out = {"task": task, "language": r["language"], "duration": round(declared, 3),
                   "text": text, "segments": [{k: v for k, v in seg.items() if k != "words"}
                                              for seg in r["segments"]], "usage": usage}
            if want_words:
                out["words"] = [w for seg in r["segments"] for w in seg.get("words", [])]
            return out
        return {"text": text, "language": r["language"], "duration": round(declared, 3), "usage": usage}

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(_require_auth)])
    async def audio_transcriptions(http_request: Request, response: Response):
        """Transcribe uploaded audio (OpenAI audio API shape, multipart)."""
        return await _audio_to_text(http_request, response, "transcribe")

    @app.post("/v1/audio/translations", dependencies=[Depends(_require_auth)])
    async def audio_translations(http_request: Request, response: Response):
        """Translate uploaded speech into English text (OpenAI audio API shape)."""
        return await _audio_to_text(http_request, response, "translate")

    @app.post("/v1/audio/speech", dependencies=[Depends(_require_auth)])
    async def audio_speech(request: SpeechRequest, http_request: Request, response: Response):
        """Synthesize speech (OpenAI audio API shape). Returns audio/wav bytes.

        Billing is per INPUT character (Unicode code points). The receipt commits that
        count and the sha256 of the returned audio; the gateway counts the characters
        and hashes the audio it delivers, independently.
        """
        _req_sha = hashlib.sha256(await http_request.body()).hexdigest()
        _rid = http_request.headers.get("x-request-id", "")
        fmt = (request.response_format or "wav").lower()
        if fmt not in SPEECH_FORMATS:
            raise HTTPException(status_code=400,
                                detail="response_format must be wav, mp3, opus, aac, flac or pcm")
        if (request.stream_format or "audio") != "audio":
            raise HTTPException(status_code=400, detail="only stream_format \"audio\" is supported")
        try:
            await _ensure_model_loaded(request.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model switch failed: {e}")
        eng = _engine_for(request.model)
        if not getattr(eng, "is_tts", False):
            raise HTTPException(status_code=400,
                                detail=f"model {eng.current_model!r} does not synthesize speech")
        if not eng.is_available():
            return _speech_unavailable(eng)
        try:
            wav = await eng.synthesize(request.input, request.voice, request.speed)
        except Interrupted:
            return _speech_unavailable(eng)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        audio = await asyncio.to_thread(encode_speech, wav, fmt)

        # The asset is the delivered bytes, in the requested format.
        _attach_receipt(response, _rid, request.model, _req_sha,
                        hashlib.sha256(audio).hexdigest(), billable_characters(request.input))
        return Response(content=audio, media_type=SPEECH_FORMATS[fmt][2], headers=dict(response.headers))

    return app

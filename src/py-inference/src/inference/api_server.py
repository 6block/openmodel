"""OpenAI-compatible REST API server for inference with runtime model switching."""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid

import base64

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engine import EngineState, GenerateResult, InferenceEngine
from .multimodal import parse_content
from .receipt import build_receipt, get_signer

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
        rendered.append({"role": m.role, "content": parsed.rendered})

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
    return "\n".join(f"{m.role}: {_content_text(m.content)}" for m in messages), images


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


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    max_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    # GATEWAY-INTERNAL (B2 stream resume): see ChatCompletionRequest.om_continuation.
    om_continuation: str | None = None


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

    # Lock to prevent concurrent model switches
    _model_switch_lock = asyncio.Lock()

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
        current = engine.current_model if hasattr(engine, 'current_model') else ""
        if _model_matches(current, requested_model):
            return

        async with _model_switch_lock:
            # Double-check after acquiring lock
            current = engine.current_model if hasattr(engine, 'current_model') else ""
            if _model_matches(current, requested_model):
                return

            logger.info("model switch requested: %s -> %s", current, requested_model)

            # Ensure model is available locally (download from FOC if needed)
            model_path = await model_manager.ensure_model(requested_model)

            # Switch engine to new model
            await engine.switch_model(model_path)
            logger.info("model switch complete, now serving: %s", requested_model)

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
            "features": ["continuation", "receipt"]
                        + (["vision"] if getattr(engine, "is_vlm", False) else []),
            "receipt_pubkey": get_signer().pubkey_hex,
        }

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

        if not engine.is_available():
            status = engine.status()
            state_desc = status.state.value
            if _model_switch_lock.locked():
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
            prompt, images = _chat_prompt(engine, request.messages, enable_thinking=want_thinking)
        except ValueError as e:
            # A malformed / oversized / undecodable image part is the client's mistake:
            # fail clean with 400 before the engine is touched (not an unhandled 500).
            raise HTTPException(status_code=400, detail=f"invalid message content: {e}")
        if images and not getattr(engine, "is_vlm", True):
            # The loaded model has no vision encoder: vLLM would reject the
            # multi_modal_data deep inside generate() as a 500. Say so plainly —
            # the client asked a text model to look at a picture.
            raise HTTPException(
                status_code=400,
                detail=f"model {engine.current_model!r} does not accept image input")
        cap = getattr(engine, "max_images_per_prompt", 0) or 0
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
        is_harmony = "gpt-oss" in (engine.current_model or "")

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
                    async for chunk in engine.generate_stream(prompt, session_key=session_key, **gen_kwargs):
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
            result = await engine.generate(prompt, session_key=session_key, **gen_kwargs)
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

        if not engine.is_available():
            status = engine.status()
            state_desc = status.state.value
            if _model_switch_lock.locked():
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
            result = await engine.generate(
                prompt,
                session_key=_engine_session_key(http_request, prompt=request.prompt),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
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

    return app

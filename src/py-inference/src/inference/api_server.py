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
from .receipt import build_receipt, get_signer

logger = logging.getLogger(__name__)


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
            content = m.content if isinstance(m.content, str) else ""
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


class ChatMessage(BaseModel):
    role: str
    content: str


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
               api_token: str = "", max_tokens_limit: int = 0) -> FastAPI:
    """Create the FastAPI application with the given inference engine.

    Args:
        engine: The inference engine (or engine pool).
        model_manager: Optional ModelManager for runtime model switching.
            When provided, requests with a non-default 'model' field will
            trigger automatic model download and switching.
        api_token: If non-empty, /v1/* endpoints require
            'Authorization: Bearer <api_token>'. /health stays open.
        max_tokens_limit: If > 0, reject requests asking for more output tokens.
    """
    app = FastAPI(title="OpenModel Inference API")

    # Lock to prevent concurrent model switches
    _model_switch_lock = asyncio.Lock()

    def _require_auth(authorization: str = Header(default="")):
        """Bearer-token gate for /v1/* endpoints (no-op when api_token unset)."""
        if not api_token:
            return
        expected = f"Bearer {api_token}"
        # constant-time compare; reject anything not exactly "Bearer <token>"
        if not hmac.compare_digest(authorization, expected):
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
            "features": ["continuation", "receipt"],
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

        prompt = "\n".join(
            f"{msg.role}: {msg.content}" for msg in request.messages
        )
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
        session_key = _engine_session_key(http_request, messages=request.messages)

        # --- Streaming response ---
        if request.stream:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            want_receipt = http_request.headers.get("x-om-receipt-req") == "1"

            async def sse_generator():
                gen_parts = []
                final_usage = None
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
                            data = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{"index": 0, "delta": {"content": chunk.text_delta}, "finish_reason": None}],
                            }
                        if chunk.finish_reason is not None:
                            final_usage = (chunk.prompt_tokens, chunk.completion_tokens, chunk.cached_tokens)
                        elif chunk.text_delta:
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

        return CompletionResponse(
            model=request.model,
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content=text),
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

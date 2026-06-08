"""OpenAI-compatible REST API server for inference with runtime model switching."""

import asyncio
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engine import EngineState, GenerateResult, InferenceEngine

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: str | list[str] | None = None
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: str | list[str] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    text: str | None = None
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


def _retry_after_for_state(state: EngineState) -> str:
    """Return Retry-After header value based on engine state."""
    if state == EngineState.LOADING:
        return "15"   # Model is reloading, will be back soon
    return "60"       # UNLOADING/PAUSED — mining may take minutes


def create_app(engine: InferenceEngine, model_manager=None) -> FastAPI:
    """Create the FastAPI application with the given inference engine.

    Args:
        engine: The inference engine (or engine pool).
        model_manager: Optional ModelManager for runtime model switching.
            When provided, requests with a non-default 'model' field will
            trigger automatic model download and switching.
    """
    app = FastAPI(title="OpenModel Inference API")

    # Lock to prevent concurrent model switches
    _model_switch_lock = asyncio.Lock()

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

        # Already loaded?
        current = engine.current_model if hasattr(engine, 'current_model') else ""
        if requested_model == current:
            return

        # Check if it's a known model path that matches
        # (e.g., current="/home/ps/openmodel/models/Qwen--Qwen2.5-1.5B-Instruct"
        #  and requested="Qwen/Qwen2.5-1.5B-Instruct")
        if current.endswith(requested_model.replace("/", "--")):
            return

        async with _model_switch_lock:
            # Double-check after acquiring lock
            current = engine.current_model if hasattr(engine, 'current_model') else ""
            if requested_model == current or current.endswith(requested_model.replace("/", "--")):
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
        }

        if hasattr(engine, "detailed_status"):
            result["multi_gpu"] = engine.detailed_status()

        return result

    @app.get("/v1/models")
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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
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

        gen_kwargs = dict(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

        # --- Streaming response ---
        if request.stream:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())

            async def sse_generator():
                try:
                    async for chunk in engine.generate_stream(prompt, **gen_kwargs):
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
                        yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"
                except RuntimeError as e:
                    error_data = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        # --- Non-streaming response ---
        try:
            result = await engine.generate(prompt, **gen_kwargs)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        if isinstance(result, GenerateResult):
            text = result.text
            finish_reason = result.finish_reason
            usage = Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            )
        else:
            text = result
            finish_reason = "stop"
            usage = Usage()

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

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
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

        try:
            result = await engine.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        if isinstance(result, GenerateResult):
            text = result.text
            finish_reason = result.finish_reason
            usage = Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            )
        else:
            text = result
            finish_reason = "stop"
            usage = Usage()

        return CompletionResponse(
            object="text_completion",
            model=request.model,
            choices=[Choice(text=text, finish_reason=finish_reason)],
            usage=usage,
        )

    return app

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_llm_wrapper
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    StreamEstimationRequest,
)
from app.services.llm_service import build_system_prompt
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()

router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def create_estimation(
    request: EstimationRequest,
    prompt_version: str = Query("v1", description="Prompt template version (v1, v2)"),
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EstimationResponse:
    """Receive structured project requirements and return a software project estimation."""
    try:
        system_prompt, user_prompt = render_estimation_prompt(request, version=prompt_version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("prompt_rendering_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to render prompt: {exc}")

    try:
        result = wrapper.complete(
            system_prompt=system_prompt,
            user_message=user_prompt,
        )
    except Exception as exc:
        log.error("estimation_endpoint_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    return EstimationResponse(
        text=result["estimation"],
        prompt_version=prompt_version,
    )


@router.post("/estimate/stream")
async def create_estimation_stream(
    request: StreamEstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EventSourceResponse:
    """Stream a software estimation token by token via Server-Sent Events."""
    system_prompt = build_system_prompt()

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        chunks = wrapper.complete_stream(
            system_prompt=system_prompt,
            user_message=request.transcription,
            model_override=request.model,
            max_tokens=request.max_tokens,
        )

        def _next_chunk() -> str | None:
            try:
                return next(chunks)
            except StopIteration:
                return None
            except Exception as exc:  # noqa: BLE001 — surface as SSE error event
                log.error("estimate_stream_failed", error=str(exc), error_type=type(exc).__name__)
                raise

        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())

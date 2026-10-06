"""Router for multi-turn conversational session endpoints.

POST /sessions                          — Initialize a new empty session.
POST /sessions/{session_id}/estimate    — Submit a turn with optional attachments.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_llm_wrapper
from app.prompts import render_session_system_prompt
from app.schemas.session import SessionCreateResponse, SessionEstimateResponse
from app.services.attachment_extractor import extract_attachment_text
from app.services.llm_wrapper import LLMWrapper
from app.services.metadata_extractor import update_project_metadata
from app.sessions import create_session, get_session

log = structlog.get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])

PROMPT_VERSION = "v1"


@router.post("", response_model=SessionCreateResponse)
def init_session() -> SessionCreateResponse:
    """Initialize a new conversational estimation session."""
    session = create_session()
    log.info("session_created", session_id=session.session_id)
    return SessionCreateResponse(session_id=session.session_id)


@router.post("/{session_id}/estimate", response_model=SessionEstimateResponse)
async def estimate_session_turn(
    session_id: str,
    transcript: str = Form(..., description="Transcript or user text for this turn."),
    attachments: list[UploadFile] = File(
        default=[], description="Optional document attachments (.pdf, .docx, .txt)."
    ),
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> SessionEstimateResponse:
    """Submit a conversational turn with optional attachments and receive an estimate."""
    session = get_session(session_id)
    if session is None:
        log.warning("session_not_found", session_id=session_id)
        raise HTTPException(status_code=404, detail="Session not found")

    # Extract text from any attached files (Camino B)
    attachment_blocks: list[str] = []
    for attachment in attachments:
        extracted = await extract_attachment_text(attachment)
        attachment_blocks.append(extracted)

    full_user_turn = transcript
    if attachment_blocks:
        full_user_turn = f"{transcript}\n\n" + "\n\n".join(attachment_blocks)

    log.info(
        "session_turn_received",
        session_id=session_id,
        transcript_chars=len(transcript),
        attachments_count=len(attachments),
        has_metadata=session.metadata.has_facts(),
    )

    # Render dynamic system prompt containing current project metadata
    system_prompt = render_session_system_prompt(
        project_metadata=session.metadata,
        version=PROMPT_VERSION,
    )

    # Buffer user turn and construct OpenAI/LiteLLM message array
    session.history.add_user_message(full_user_turn)
    messages = session.history.to_messages_list(system_prompt)

    try:
        result = wrapper.complete_messages(messages=messages)
    except Exception as exc:
        log.error(
            "session_turn_llm_failed",
            session_id=session_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc

    estimation_text = result["estimation"]

    # Close the turn in history (enforcing sliding window)
    session.history.add_assistant_message(estimation_text)

    # Extract new facts from interaction and update metadata
    session.metadata = update_project_metadata(
        current=session.metadata,
        user_text=full_user_turn,
        assistant_text=estimation_text,
    )
    session.touch()

    log.info(
        "session_turn_completed",
        session_id=session_id,
        output_chars=len(estimation_text),
        team_size=session.metadata.assumed_team_size,
        technologies=session.metadata.mentioned_technologies,
    )

    return SessionEstimateResponse(
        session_id=session.session_id,
        text=estimation_text,
        prompt_version=PROMPT_VERSION,
        project_metadata=session.metadata,
    )

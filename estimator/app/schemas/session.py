"""Response and request models for conversational sessions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.sessions import ProjectMetadata


class SessionCreateResponse(BaseModel):
    """Payload returned upon session creation."""

    session_id: str = Field(description="Unique identifier for the created session.")


class SessionEstimateResponse(BaseModel):
    """Payload returned by multi-turn estimation endpoint."""

    session_id: str = Field(description="Identifier of the session.")
    text: str = Field(description="Rendered estimation text from the LLM.")
    prompt_version: str = Field(description="Version of prompt template used.")
    project_metadata: ProjectMetadata = Field(
        description="Structured project memory facts preserved across turns."
    )

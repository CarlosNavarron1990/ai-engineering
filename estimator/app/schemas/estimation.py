from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Session 4: Typed Contract
# ---------------------------------------------------------------------------


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class ReferenceProject(BaseModel):
    """Optional reference project for few-shot in-context learning."""

    name: str = Field(..., description="Name or title of the reference project")
    description: str = Field(..., description="Summary or scope of the reference project")
    duration_weeks: int | None = Field(default=None, ge=1, description="Estimated duration in weeks")
    cost_eur: float | None = Field(default=None, ge=0.0, description="Estimated cost in EUR")


class EstimationRequest(BaseModel):
    """Structured request for project estimation from typed parameters."""

    description: str = Field(..., min_length=20, max_length=2000, description="Project requirements description")
    project_type: ProjectType = Field(..., description="Type of software project")
    detail_level: DetailLevel = Field(..., description="Depth and granularity of estimation")
    output_format: OutputFormat = Field(..., description="Format and structure of the output")
    reference_projects: list[ReferenceProject] | None = Field(
        default=None,
        description="Optional list of similar reference projects for context",
    )


class EstimationResponse(BaseModel):
    """Response containing generated estimation text and prompt metadata."""

    text: str = Field(..., description="Generated software estimation text")
    prompt_version: str = Field(..., description="Version of the prompt template used")


# ---------------------------------------------------------------------------
# Streaming and Legacy Schemas
# ---------------------------------------------------------------------------

PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]


class StreamEstimationRequest(BaseModel):
    """Streaming endpoint request."""

    transcription: str = Field(..., min_length=50, description="Meeting transcription text")
    model: str | None = Field(default=None, description="Override the default model")
    max_tokens: int = Field(default=4000, gt=0, le=16000)


class TokenUsage(BaseModel):
    """Token consumption details from the LLM call(s)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0


class StructureCheck(BaseModel):
    """Level-1 structural evaluation of the generated estimation."""

    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_row_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float
    issues: list[str]

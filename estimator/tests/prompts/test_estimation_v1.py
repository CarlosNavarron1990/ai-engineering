import pytest
from jinja2 import UndefinedError

from app.prompts.loader import get_jinja_environment, render_estimation_prompt
from app.schemas import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
    ReferenceProject,
)


def _base_request(**kwargs) -> EstimationRequest:
    defaults = {
        "description": "We need an end-to-end B2B platform with team workspaces, role permissions and Stripe billing.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    defaults.update(kwargs)
    return EstimationRequest(**defaults)


def test_render_includes_description_literally_inside_project_description_block():
    """Verify description is placed verbatim inside the <project_description> block."""
    raw_description = "Special token #XYZ-999: Build a custom real-time fleet tracking portal."
    req = _base_request(description=raw_description)

    system_prompt, user_prompt = render_estimation_prompt(req, version="v1")

    assert "<project_description>" in user_prompt
    assert "</project_description>" in user_prompt
    # Must contain the verbatim text inside the block
    inner_content = user_prompt.split("<project_description>")[1].split("</project_description>")[0]
    assert raw_description in inner_content


def test_output_format_phases_table_vs_narrative():
    """Verify phases_table includes format keywords ('phases_table', 'confidence_pct')

    and narrative format strictly omits them.
    """
    req_phases = _base_request(output_format=OutputFormat.PHASES_TABLE)
    system_phases, _ = render_estimation_prompt(req_phases, version="v1")
    assert "phases_table" in system_phases
    assert "confidence_pct" in system_phases

    req_narrative = _base_request(output_format=OutputFormat.NARRATIVE)
    system_narrative, _ = render_estimation_prompt(req_narrative, version="v1")
    assert "phases_table" not in system_narrative
    assert "confidence_pct" not in system_narrative
    assert "narrative" in system_narrative.lower()


def test_detail_level_detailed_includes_assumptions_instruction_and_summary_omits_it():
    """Verify detailed level requires listing assumptions per phase ('listar asunciones por fase'

    or 'assumptions per phase'), while summary level does not contain this instruction.
    """
    req_detailed = _base_request(detail_level=DetailLevel.DETAILED)
    system_detailed, _ = render_estimation_prompt(req_detailed, version="v1")
    assert "asunciones por fase" in system_detailed or "assumptions per phase" in system_detailed.lower()

    req_summary = _base_request(detail_level=DetailLevel.SUMMARY)
    system_summary, _ = render_estimation_prompt(req_summary, version="v1")
    assert "asunciones por fase" not in system_summary
    assert "assumptions per phase" not in system_summary.lower()


def test_reference_projects_for_loop_renders_when_present():
    """Verify optional reference projects are rendered via loop when provided and omitted when None."""
    ref1 = ReferenceProject(
        name="Project Falcon",
        description="Previous enterprise SaaS billing integration",
        duration_weeks=6,
        cost_eur=18000.0,
    )
    ref2 = ReferenceProject(
        name="Project Eagle",
        description="Mobile fleet tracking MVP",
        duration_weeks=8,
        cost_eur=24000.0,
    )

    req_with_refs = _base_request(reference_projects=[ref1, ref2])
    _, user_with_refs = render_estimation_prompt(req_with_refs, version="v1")
    assert "<reference_projects>" in user_with_refs
    assert "Project Falcon" in user_with_refs
    assert "Project Eagle" in user_with_refs
    assert "18000.0 EUR" in user_with_refs

    req_without_refs = _base_request(reference_projects=None)
    _, user_without_refs = render_estimation_prompt(req_without_refs, version="v1")
    assert "<reference_projects>" not in user_without_refs


def test_strict_undefined_raises_on_missing_variable():
    """Verify that Jinja2 StrictUndefined catches missing template variables."""
    env = get_jinja_environment("v1")
    template = env.from_string("Hello {{ missing_variable_name }}")
    with pytest.raises(UndefinedError):
        template.render()


def test_v2_prompt_version_renders_agile_template():
    """Verify prompt versioning allows rendering v2 with agile instructions."""
    req = _base_request(project_type=ProjectType.MOBILE_APP, output_format=OutputFormat.PHASES_TABLE)
    system_v2, user_v2 = render_estimation_prompt(req, version="v2")

    assert "Principal Engineering Consultant" in system_v2
    assert "Agile" in system_v2 or "Sprint" in system_v2
    assert "<project_description>" in user_v2

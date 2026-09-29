from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest

log = structlog.get_logger()

PROMPTS_ROOT = Path(__file__).resolve().parent / "estimation"


@lru_cache(maxsize=16)
def get_jinja_environment(version: str) -> Environment:
    """Return a configured Jinja2 Environment for the specified prompt version.

    Uses StrictUndefined, trim_blocks=True, and lstrip_blocks=True as required
    by the template architecture contract.
    """
    version_dir = PROMPTS_ROOT / version
    if not version_dir.is_dir():
        available = [p.name for p in PROMPTS_ROOT.iterdir() if p.is_dir()]
        raise ValueError(
            f"Prompt version '{version}' not found at {version_dir}. "
            f"Available versions: {available}"
        )

    loader = FileSystemLoader([str(version_dir), str(PROMPTS_ROOT)])
    return Environment(
        loader=loader,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_estimation_prompt(
    request: EstimationRequest | dict[str, Any],
    version: str = "v1",
) -> tuple[str, str]:
    """Render the versioned Jinja2 prompt templates for project estimation.

    Parameters
    ----------
    request:
        The validated EstimationRequest or a compatible dictionary.
    version:
        The prompt template version directory to load (e.g. "v1", "v2").

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt) ready for model dispatch.
    """
    env = get_jinja_environment(version)

    if isinstance(request, EstimationRequest):
        context = {
            "description": request.description,
            "project_type": request.project_type.value,
            "detail_level": request.detail_level.value,
            "output_format": request.output_format.value,
            "reference_projects": request.reference_projects,
            "request": request,
        }
    elif isinstance(request, dict):
        context = dict(request)
        # Ensure enums or values are string-compatible for template conditionals
        for k in ("project_type", "detail_level", "output_format"):
            if k in context and hasattr(context[k], "value"):
                context[k] = context[k].value
    else:
        context = {
            "description": getattr(request, "description", ""),
            "project_type": getattr(getattr(request, "project_type", None), "value", str(getattr(request, "project_type", ""))),
            "detail_level": getattr(getattr(request, "detail_level", None), "value", str(getattr(request, "detail_level", ""))),
            "output_format": getattr(getattr(request, "output_format", None), "value", str(getattr(request, "output_format", ""))),
            "reference_projects": getattr(request, "reference_projects", None),
            "request": request,
        }

    template_system = env.get_template("system.j2")
    template_user = env.get_template("user.j2")

    system_prompt = template_system.render(**context)
    user_prompt = template_user.render(**context)

    # Calculate content fingerprint for auditability and debugging
    prompt_payload = f"{system_prompt}\n---USER---\n{user_prompt}"
    content_hash = hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest()[:16]

    log.info(
        "prompt_rendered",
        prompt_version=version,
        content_hash=content_hash,
        system_chars=len(system_prompt),
        user_chars=len(user_prompt),
    )

    return system_prompt, user_prompt

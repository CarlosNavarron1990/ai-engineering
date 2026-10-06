"""Project metadata extraction service.

Extracts structured project facts (name, team size, technologies, scope) from
conversation turns using fast, rule-based heuristics and regex patterns.
Avoids adding secondary LLM API roundtrip latency or token cost during each turn.
"""

from __future__ import annotations

import re

from app.sessions import ProjectMetadata

# Known technologies catalog for case-insensitive matching
TECH_CATALOG = [
    "Python", "FastAPI", "Django", "Flask", "TypeScript", "JavaScript",
    "React", "Vue", "Next.js", "Node.js", "Angular", "Svelte",
    "PostgreSQL", "Postgres", "MySQL", "MongoDB", "Redis", "SQLite",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "GraphQL", "REST",
    "Tailwind", "CSS", "HTML", "Swift", "Kotlin", "Flutter", "React Native",
    "Kafka", "RabbitMQ", "Celery", "Elasticsearch", "OpenAI", "Anthropic",
    "PyTorch", "TensorFlow", "Pandas", "Spark",
]


def _extract_project_name(text: str) -> str | None:
    patterns = [
        r'(?:project|app|system|service|platform)\s*(?:named|called|is|:)\s*["\']?([A-Za-z0-9_\-\s]{3,30})["\']?(?:[\.\,\n]|$)',
        r'(?:estimating|estimate for)\s*["\']?([A-Za-z0-9_\-\s]{3,30})["\']?(?:[\.\,\n]|$)',
        r'#\s*Project\s*:\s*([A-Za-z0-9_\-\s]{3,30})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Exclude generic words
            if candidate.lower() not in {"an mvp", "a platform", "a service", "new system"}:
                return candidate
    return None


def _extract_team_size(text: str) -> int | None:
    patterns = [
        r'(?:team(?:\s*size)?\s*(?:of|is|:)?\s*)(\d+)\s*(?:engineers|developers|devs|people|members)?',
        r'(\d+)\s*(?:engineers|developers|devs|people)\s*(?:team)?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                size = int(match.group(1))
                if 1 <= size <= 100:
                    return size
            except ValueError:
                continue
    return None


def _extract_technologies(text: str) -> list[str]:
    found: list[str] = []
    text_lower = text.lower()
    for tech in TECH_CATALOG:
        # Match as word boundary
        pattern = rf'\b{re.escape(tech.lower())}\b'
        if re.search(pattern, text_lower):
            found.append(tech)
    return found


def _extract_scope(text: str) -> str | None:
    patterns = [
        r'(?:scope|agreed scope|goal)\s*(?:is|:)\s*([^\n\.\;]{10,120})',
        r'(?:building|build|develop)\s*([^\n\.\;]{10,120})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scope = match.group(1).strip()
            if len(scope) >= 10:
                return scope
    return None


def update_project_metadata(
    current: ProjectMetadata,
    user_text: str,
    assistant_text: str,
) -> ProjectMetadata:
    """Extract facts from user turn and assistant response and merge with current metadata."""
    combined_text = f"{user_text}\n{assistant_text}"

    # Extract new facts
    detected_name = _extract_project_name(combined_text)
    detected_team = _extract_team_size(combined_text)
    detected_tech = _extract_technologies(combined_text)
    detected_scope = _extract_scope(combined_text)

    # Merge project name (retain existing if not newly found)
    project_name = detected_name or current.project_name

    # Merge team size
    assumed_team_size = detected_team if detected_team is not None else current.assumed_team_size

    # Merge technologies maintaining uniqueness and original casing
    existing_tech_set = {t.lower(): t for t in current.mentioned_technologies}
    for tech in detected_tech:
        if tech.lower() not in existing_tech_set:
            existing_tech_set[tech.lower()] = tech
    merged_tech = list(existing_tech_set.values())

    # Merge agreed scope
    agreed_scope = detected_scope or current.agreed_scope

    return ProjectMetadata(
        project_name=project_name,
        assumed_team_size=assumed_team_size,
        mentioned_technologies=merged_tech,
        agreed_scope=agreed_scope,
    )

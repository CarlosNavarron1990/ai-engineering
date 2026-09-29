from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_llm_wrapper
from app.main import app


class _FakeWrapper:
    """Fake LLMWrapper capturing call inputs and returning canned responses."""

    def __init__(self, estimation_text: str = "## Project Summary\n\nTotal estimation: 240 hours."):
        self.estimation_text = estimation_text
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None = None,
        max_tokens: int = 4000,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model_override": model_override,
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            }
        )
        return {
            "estimation": self.estimation_text,
            "model": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 150, "output_tokens": 75, "total_tokens": 225},
            "latency_ms": 25,
            "cost_usd": 0.00015,
            "cache_hit": False,
        }


@pytest.fixture
def fake_wrapper() -> Iterator[_FakeWrapper]:
    wrapper = _FakeWrapper()
    app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    try:
        yield wrapper
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


VALID_PAYLOAD = {
    "description": "We need a SaaS dashboard with multi-tenant auth, roles, and automated billing.",
    "project_type": "web_saas",
    "detail_level": "detailed",
    "output_format": "phases_table",
}


def test_post_estimate_success(client: TestClient, fake_wrapper: _FakeWrapper) -> None:
    response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt_version"] == "v1"
    assert data["text"] == fake_wrapper.estimation_text

    # Verify wrapper received separate system and user prompts
    assert len(fake_wrapper.calls) == 1
    call = fake_wrapper.calls[0]
    assert "phases_table" in call["system_prompt"]
    assert "<project_description>" in call["user_message"]
    assert VALID_PAYLOAD["description"] in call["user_message"]


def test_post_estimate_with_query_param_version_v2(
    client: TestClient, fake_wrapper: _FakeWrapper
) -> None:
    response = client.post("/api/v1/estimate?prompt_version=v2", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt_version"] == "v2"
    assert len(fake_wrapper.calls) == 1
    assert "Principal Engineering Consultant" in fake_wrapper.calls[0]["system_prompt"]


def test_post_estimate_invalid_version_returns_400(
    client: TestClient, fake_wrapper: _FakeWrapper
) -> None:
    response = client.post("/api/v1/estimate?prompt_version=v999", json=VALID_PAYLOAD)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()
    assert len(fake_wrapper.calls) == 0


def test_post_estimate_description_too_short_fails_validation(
    client: TestClient, fake_wrapper: _FakeWrapper
) -> None:
    bad_payload = {**VALID_PAYLOAD, "description": "Too short"}
    response = client.post("/api/v1/estimate", json=bad_payload)
    assert response.status_code == 422
    assert len(fake_wrapper.calls) == 0


def test_post_estimate_works_on_both_root_and_apiv1_routes(
    client: TestClient, fake_wrapper: _FakeWrapper
) -> None:
    res_root = client.post("/estimate", json=VALID_PAYLOAD)
    assert res_root.status_code == 200

    res_api = client.post("/api/v1/estimate", json=VALID_PAYLOAD)
    assert res_api.status_code == 200
    assert res_root.json() == res_api.json()


def test_post_estimate_with_reference_projects(
    client: TestClient, fake_wrapper: _FakeWrapper
) -> None:
    payload = {
        **VALID_PAYLOAD,
        "reference_projects": [
            {
                "name": "E-Commerce Microservices",
                "description": "Refactoring monolithic store into event-driven services",
                "duration_weeks": 10,
                "cost_eur": 35000.0,
            }
        ],
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    call = fake_wrapper.calls[0]
    assert "<reference_projects>" in call["user_message"]
    assert "E-Commerce Microservices" in call["user_message"]
    assert "35000.0 EUR" in call["user_message"]

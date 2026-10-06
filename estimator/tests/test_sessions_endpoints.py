"""Integration tests for multi-turn sessions endpoints."""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_llm_wrapper
from app.main import app
from app.sessions import clear_sessions


class FakeSessionLLMWrapper:
    """Mock wrapper that records all message payloads passed in calls."""

    def __init__(self, response_text: str = "Mocked multi-turn estimation.") -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def complete_messages(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "estimation": self.response_text,
            "model": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
            "latency_ms": 10,
            "cost_usd": 0.0001,
            "cache_hit": False,
        }


@pytest.fixture
def fake_llm() -> FakeSessionLLMWrapper:
    wrapper = FakeSessionLLMWrapper()
    app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    yield wrapper
    app.dependency_overrides.pop(get_llm_wrapper, None)


@pytest.fixture(autouse=True)
def clean_sessions_state():
    clear_sessions()
    yield
    clear_sessions()


def test_create_session(client: TestClient) -> None:
    response = client.post("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 10


def test_estimate_nonexistent_session_returns_404(
    client: TestClient, fake_llm: FakeSessionLLMWrapper
) -> None:
    response = client.post(
        "/sessions/nonexistent-id/estimate",
        data={"transcript": "Some project description text that is long enough."},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_multi_turn_persists_and_updates_metadata(
    client: TestClient, fake_llm: FakeSessionLLMWrapper
) -> None:
    # 1. Create session
    create_resp = client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    # 2. Turn 1: introduce project and team size
    turn1_data = {
        "transcript": 'Project: "InventoryPro". Team of 3 engineers. We will use FastAPI and React.'
    }
    resp1 = client.post(f"/sessions/{session_id}/estimate", data=turn1_data)
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["session_id"] == session_id
    assert body1["project_metadata"]["project_name"] == "InventoryPro"
    assert body1["project_metadata"]["assumed_team_size"] == 3
    assert "FastAPI" in body1["project_metadata"]["mentioned_technologies"]

    # Check that LLM received 1 turn
    assert len(fake_llm.calls) == 1
    call1_messages = fake_llm.calls[0]["messages"]
    assert call1_messages[0]["role"] == "system"
    assert call1_messages[1]["role"] == "user"

    # 3. Turn 2: add more requirements
    turn2_data = {
        "transcript": "We also need PostgreSQL for relational data and Redis for caching."
    }
    resp2 = client.post(f"/sessions/{session_id}/estimate", data=turn2_data)
    assert resp2.status_code == 200
    body2 = resp2.json()

    # Verify metadata accumulated
    assert body2["project_metadata"]["project_name"] == "InventoryPro"
    assert "PostgreSQL" in body2["project_metadata"]["mentioned_technologies"]
    assert "Redis" in body2["project_metadata"]["mentioned_technologies"]

    # Verify history sent to LLM included previous user and assistant turns
    assert len(fake_llm.calls) == 2
    call2_messages = fake_llm.calls[1]["messages"]
    # Messages should be: System, Turn 1 User, Turn 1 Assistant, Turn 2 User
    assert len(call2_messages) == 4
    assert call2_messages[0]["role"] == "system"
    # System prompt should have injected <project_metadata>
    assert "<project_metadata>" in call2_messages[0]["content"]
    assert "InventoryPro" in call2_messages[0]["content"]
    assert call2_messages[1]["role"] == "user"
    assert call2_messages[2]["role"] == "assistant"
    assert call2_messages[3]["role"] == "user"


def test_estimate_with_attachment(
    client: TestClient, fake_llm: FakeSessionLLMWrapper
) -> None:
    create_resp = client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    sample_doc = io.BytesIO(b"Architecture Specification:\nMicroservice architecture using Kafka.")
    files = [
        ("attachments", ("spec.txt", sample_doc, "text/plain"))
    ]
    data = {"transcript": "Please estimate based on the attached architecture."}

    resp = client.post(f"/sessions/{session_id}/estimate", data=data, files=files)
    assert resp.status_code == 200

    # Verify LLM received the extracted attachment content
    assert len(fake_llm.calls) == 1
    user_message = fake_llm.calls[0]["messages"][1]["content"]
    assert "--- attachment: spec.txt ---" in user_message
    assert "Microservice architecture using Kafka" in user_message


def test_sliding_window_caps_messages_sent_to_llm(
    client: TestClient, fake_llm: FakeSessionLLMWrapper
) -> None:
    create_resp = client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    # Send 8 turns
    for turn_idx in range(1, 9):
        client.post(
            f"/sessions/{session_id}/estimate",
            data={"transcript": f"Turn {turn_idx} details and refinements."},
        )

    assert len(fake_llm.calls) == 8
    last_call_messages = fake_llm.calls[-1]["messages"]

    # System prompt (1) + 6 pairs of history (12) + current turn user message (1) = 13 messages max
    # Max turns = 6 pairs
    assert len(last_call_messages) <= 13
    assert last_call_messages[0]["role"] == "system"
    # Turn 1 and 2 must have been dropped from the conversation window
    all_contents = " ".join(m["content"] for m in last_call_messages)
    assert "Turn 1 details" not in all_contents
    assert "Turn 8 details" in all_contents

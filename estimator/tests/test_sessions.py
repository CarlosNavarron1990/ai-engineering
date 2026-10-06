"""Unit tests for in-memory session management, sliding window, and metadata extraction."""

from __future__ import annotations

import pytest

from app.services.metadata_extractor import update_project_metadata
from app.sessions import (
    ConversationHistory,
    ProjectMetadata,
    clear_sessions,
    create_session,
    get_session,
)


@pytest.fixture(autouse=True)
def clean_sessions_state():
    """Ensure in-memory sessions are cleared before and after each test."""
    clear_sessions()
    yield
    clear_sessions()


def test_create_and_retrieve_session() -> None:
    session = create_session(max_turns=6)
    assert session.session_id is not None
    assert len(session.session_id) > 10

    retrieved = get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id
    assert retrieved.history.max_turns == 6
    assert isinstance(retrieved.metadata, ProjectMetadata)


def test_get_nonexistent_session_returns_none() -> None:
    assert get_session("non-existent-uuid") is None


def test_conversation_history_sliding_window_normal() -> None:
    history = ConversationHistory(max_turns=3)

    history.add_user_message("Turn 1 user")
    history.add_assistant_message("Turn 1 assistant")

    history.add_user_message("Turn 2 user")
    history.add_assistant_message("Turn 2 assistant")

    assert len(history.turns) == 2
    messages = history.to_messages_list(system_prompt="system prompt")
    assert len(messages) == 5  # 1 system + 2 * 2 turns
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Turn 1 user"
    assert messages[2]["content"] == "Turn 1 assistant"
    assert messages[3]["content"] == "Turn 2 user"
    assert messages[4]["content"] == "Turn 2 assistant"


def test_conversation_history_sliding_window_caps_at_max_turns() -> None:
    history = ConversationHistory(max_turns=6)

    # Add 8 consecutive turns (user + assistant pairs)
    for i in range(1, 9):
        history.add_user_message(f"Turn {i} user")
        history.add_assistant_message(f"Turn {i} assistant")

    # Strictly capped at max_turns = 6
    assert len(history.turns) == 6

    # Verify oldest turns (1 and 2) were dropped and turns 3..8 remain
    assert history.turns[0] == ("Turn 3 user", "Turn 3 assistant")
    assert history.turns[-1] == ("Turn 8 user", "Turn 8 assistant")

    messages = history.to_messages_list(system_prompt="Base System")
    assert len(messages) == 1 + (6 * 2)  # 13 messages total
    assert messages[0] == {"role": "system", "content": "Base System"}
    assert messages[1]["content"] == "Turn 3 user"


def test_project_metadata_tech_merge() -> None:
    meta = ProjectMetadata(mentioned_technologies=["FastAPI", "React"])
    updated = update_project_metadata(
        current=meta,
        user_text="Let's also use PostgreSQL and Docker.",
        assistant_text="Acknowledged. We will add Postgres and Docker to the infrastructure.",
    )

    expected_subset = {"FastAPI", "React", "PostgreSQL", "Docker"}
    actual_set = set(updated.mentioned_technologies)
    assert expected_subset.issubset(actual_set)


def test_metadata_extractor_heuristics() -> None:
    meta = ProjectMetadata()
    user_msg = (
        'Estimating project: "FleetTracker". We have a team of 4 developers. '
        "The scope is to build an MVP logistics dashboard in Python and Vue."
    )
    assistant_msg = "Here is the estimation for FleetTracker with 4 developers..."

    updated = update_project_metadata(meta, user_msg, assistant_msg)

    assert updated.project_name == "FleetTracker"
    assert updated.assumed_team_size == 4
    assert "Python" in updated.mentioned_technologies
    assert "Vue" in updated.mentioned_technologies
    assert updated.has_facts() is True

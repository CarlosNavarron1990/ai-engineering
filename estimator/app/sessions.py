"""Session state management and in-memory conversation history.

Why In-Memory Volatility is Accepted:
-------------------------------------
Sessions are maintained in process memory (a standard Python dictionary) rather
than an external database or Redis. This design is deliberate for Session 5:
1. The goal is to isolate conversational memory mechanics (sliding window,
   project metadata accumulation) without introducing persistence migration
   overhead.
2. An in-process dictionary guarantees instant sub-millisecond retrieval and
   zero external infrastructure dependencies during development.
3. Persistent session storage (PostgreSQL/Redis) is slated for later architectural
   iterations once memory schemas and extraction heuristics stabilize.
"""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field


class ProjectMetadata(BaseModel):
    """Structured project truths preserved outside the sliding history window."""

    project_name: str | None = Field(
        default=None,
        description="Identified or declared name of the software project.",
    )
    assumed_team_size: int | None = Field(
        default=None,
        description="Assumed or agreed engineering team size.",
    )
    mentioned_technologies: list[str] = Field(
        default_factory=list,
        description="Cumulative list of programming languages, frameworks, or tools.",
    )
    agreed_scope: str | None = Field(
        default=None,
        description="High-level scope summary agreed upon across conversation turns.",
    )

    def has_facts(self) -> bool:
        """Return True if at least one metadata fact has been discovered."""
        return bool(
            self.project_name
            or self.assumed_team_size is not None
            or self.mentioned_technologies
            or self.agreed_scope
        )


class ConversationHistory:
    """Sliding-window conversational turn manager.

    A turn consists of a user message and an assistant message pair.
    When turn count exceeds max_turns, the oldest pair is dropped,
    while the system prompt remains invariant when generating message lists.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        # Store as pairs of (user_content, assistant_content)
        self.turns: list[tuple[str, str]] = []
        self._pending_user_message: str | None = None

    def add_user_message(self, content: str) -> None:
        """Buffer a user message waiting for assistant completion."""
        self._pending_user_message = content

    def add_assistant_message(self, content: str) -> None:
        """Complete the turn with the assistant response and enforce sliding window."""
        if self._pending_user_message is not None:
            self.turns.append((self._pending_user_message, content))
            self._pending_user_message = None

            # Enforce sliding window: retain only the last max_turns pairs
            if len(self.turns) > self.max_turns:
                self.turns = self.turns[-self.max_turns:]

    def to_messages_list(self, system_prompt: str) -> list[dict[str, str]]:
        """Render the complete OpenAI/LiteLLM message payload.

        The system prompt is always injected as the first message.
        Followed by active history turns, plus any pending user message,
        ensuring the total number of user turns never exceeds max_turns.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        history_turns = self.turns
        if self._pending_user_message is not None and len(history_turns) >= self.max_turns:
            history_turns = history_turns[-(self.max_turns - 1):]

        for user_msg, assistant_msg in history_turns:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        if self._pending_user_message is not None:
            messages.append({"role": "user", "content": self._pending_user_message})

        return messages


class Session:
    """In-memory session holding conversational state and project memory."""

    def __init__(self, session_id: str, max_turns: int = 6) -> None:
        self.session_id = session_id
        self.history = ConversationHistory(max_turns=max_turns)
        self.metadata = ProjectMetadata()
        self.created_at = time.time()
        self.updated_at = time.time()

    def touch(self) -> None:
        """Update last modified timestamp."""
        self.updated_at = time.time()


# Global in-memory storage dictionary
_SESSIONS: dict[str, Session] = {}


def create_session(max_turns: int = 6) -> Session:
    """Create a new session with a UUID v4 identifier."""
    session_id = str(uuid.uuid4())
    session = Session(session_id=session_id, max_turns=max_turns)
    _SESSIONS[session_id] = session
    return session


def get_session(session_id: str) -> Session | None:
    """Retrieve an existing session by ID or return None."""
    return _SESSIONS.get(session_id)


def clear_sessions() -> None:
    """Purge all sessions from memory (used for test isolation)."""
    _SESSIONS.clear()

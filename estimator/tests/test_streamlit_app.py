from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_streamlit_app_renders():
    """Test that streamlit_app.py renders the sidebar, CAG examples, and chat components."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    assert len(at.sidebar) > 0


def test_streamlit_app_submit_message():
    """Test that submitting a message via chat input triggers stream and updates state."""

    def fake_stream(messages, metrics_collector=None):
        if metrics_collector is not None:
            metrics_collector["provider"] = "mock"
            metrics_collector["model"] = "mock-model"
            metrics_collector["duration"] = 0.5
            metrics_collector["usage"] = {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
            }
        yield "## Estimación de Prueba\n"
        yield "- Horas: 100"

    with patch("app.services.llm_service.stream_estimation", side_effect=fake_stream):
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception

        # Simulate user entering chat input
        at.chat_input[0].set_value("Necesitamos una app móvil para reservas").run(timeout=10)
        assert not at.exception

        # Check that session state messages were populated
        assert len(at.session_state["messages"]) >= 2
        assert at.session_state["messages"][0]["role"] == "user"
        assert at.session_state["messages"][1]["role"] == "assistant"
        assert "Estimación de Prueba" in at.session_state["messages"][1]["content"]

from unittest.mock import MagicMock, patch

from app.services.llm_service import stream_estimation


def test_stream_estimation_openai():
    """Verify that stream_estimation yields chunks and populates metrics for OpenAI."""
    # Mock OpenAI client and streaming chunks
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock()]
    mock_chunk_1.choices[0].delta.content = "## Estimación Proyecto\n\n"
    mock_chunk_1.usage = None
    mock_chunk_1.model = "gpt-4o-mini"

    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock()]
    mock_chunk_2.choices[0].delta.content = "| Task | Hours | Cost (EUR) |\n"
    mock_chunk_2.usage = None
    mock_chunk_2.model = "gpt-4o-mini"

    mock_chunk_3 = MagicMock()
    mock_chunk_3.choices = []
    mock_chunk_3.usage = MagicMock()
    mock_chunk_3.usage.prompt_tokens = 120
    mock_chunk_3.usage.completion_tokens = 45
    mock_chunk_3.usage.total_tokens = 165
    mock_chunk_3.model = "gpt-4o-mini"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [
        mock_chunk_1,
        mock_chunk_2,
        mock_chunk_3,
    ]

    metrics = {}
    with patch("openai.OpenAI", return_value=mock_client):
        chunks = list(stream_estimation("Meeting transcript", metrics_collector=metrics))

    assert chunks == ["## Estimación Proyecto\n\n", "| Task | Hours | Cost (EUR) |\n"]
    assert metrics["provider"] == "openai"
    assert metrics["model"] == "gpt-4o-mini"
    assert metrics["usage"]["input_tokens"] == 120
    assert metrics["usage"]["output_tokens"] == 45
    assert metrics["usage"]["total_tokens"] == 165
    assert "duration" in metrics


def test_stream_estimation_anthropic():
    """Verify that stream_estimation yields chunks and populates metrics for Anthropic."""
    mock_final_msg = MagicMock()
    mock_final_msg.model = "claude-3-5-sonnet-20241022"
    mock_final_msg.usage.input_tokens = 200
    mock_final_msg.usage.output_tokens = 80

    mock_stream = MagicMock()
    mock_stream.text_stream = ["## Claude Estimation\n\n", "Total: 10,000 EUR"]
    mock_stream.get_final_message.return_value = mock_final_msg
    mock_stream.__enter__.return_value = mock_stream

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream

    metrics = {}
    with patch("app.services.llm_service.get_settings") as mock_settings:
        mock_settings.return_value.LLM_PROVIDER = "anthropic"
        mock_settings.return_value.ANTHROPIC_API_KEY = "test-key"
        mock_settings.return_value.LLM_MODEL = "claude-3-5-sonnet-20241022"

        with patch("anthropic.Anthropic", return_value=mock_client):
            chunks = list(stream_estimation("Meeting transcript", metrics_collector=metrics))

    assert chunks == ["## Claude Estimation\n\n", "Total: 10,000 EUR"]
    assert metrics["provider"] == "anthropic"
    assert metrics["model"] == "claude-3-5-sonnet-20241022"
    assert metrics["usage"]["input_tokens"] == 200
    assert metrics["usage"]["output_tokens"] == 80
    assert metrics["usage"]["total_tokens"] == 280
    assert "duration" in metrics

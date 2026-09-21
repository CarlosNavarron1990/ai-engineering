import time
from collections.abc import Generator

import structlog

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples_for_prompt

log = structlog.get_logger()

MAX_TOKENS = 4000


class LLMServiceError(Exception):
    """Raised when the LLM provider call fails."""


def build_system_prompt() -> str:
    """Construct the system prompt with role definition and reference examples."""
    examples_text = format_examples_for_prompt(ESTIMATION_EXAMPLES)
    return (
        "You are a senior software consultant with 15+ years of experience in project "
        "estimation. Your task is to produce a detailed software project estimation based "
        "on a meeting transcription provided by the user.\n\n"
        "Below are reference estimations from previous projects. Use them as a guide for "
        "structure, level of detail, and realistic pricing. Adapt the content to match the "
        "specific project described in the transcription.\n\n"
        "Your output MUST follow this exact format:\n"
        "- Project title as an H2 heading\n"
        "- A task breakdown table with columns: Task, Hours, Cost (EUR)\n"
        "- Total hours\n"
        "- Total cost in EUR\n"
        "- Recommended team composition\n"
        "- Estimated duration in weeks\n\n"
        "Use a developer rate of approximately 62.50 EUR/hour (500 EUR/day) and a designer "
        "rate of approximately 50 EUR/hour (400 EUR/day). Provide realistic, well-justified "
        "numbers.\n\n"
        f"{examples_text}"
    )


def generate_estimation(transcription: str) -> dict:
    """Generate a software estimation from a meeting transcription using the configured LLM."""
    settings = get_settings()
    system_prompt = build_system_prompt()

    log.info("generating_estimation", provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL)

    try:
        if settings.LLM_PROVIDER == "openai":
            return _call_openai(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription},
                ],
            )
        else:
            return _call_anthropic(
                system=system_prompt,
                user_message=transcription,
            )
    except LLMServiceError:
        raise
    except Exception as exc:
        log.error("llm_call_failed", error=str(exc), provider=settings.LLM_PROVIDER)
        raise LLMServiceError(f"LLM call failed: {exc}") from exc


def _call_openai(messages: list[dict]) -> dict:
    """Send a chat completion request to the OpenAI API."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )

    usage = response.usage
    log.info(
        "llm_response_received",
        provider="openai",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
    )

    return {
        "estimation": response.choices[0].message.content,
        "model": response.model,
        "provider": "openai",
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


def _call_anthropic(system: str, user_message: str) -> dict:
    """Send a message request to the Anthropic API."""
    from anthropic import Anthropic

    settings = get_settings()
    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    log.info(
        "llm_response_received",
        provider="anthropic",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    return {
        "estimation": response.content[0].text,
        "model": response.model,
        "provider": "anthropic",
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
    }


def stream_estimation(
    messages: list[dict] | str,
    metrics_collector: dict | None = None,
) -> Generator[str, None, None]:
    """Stream a software estimation from a meeting transcription or conversation messages.

    Optionally collects execution metrics (provider, model, duration, token usage) in
    the provided metrics_collector dictionary.
    """
    settings = get_settings()
    system_prompt = build_system_prompt()

    if isinstance(messages, str):
        normalized_messages = [{"role": "user", "content": messages}]
    else:
        normalized_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]

    log.info("streaming_estimation", provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL)

    try:
        if settings.LLM_PROVIDER == "openai":
            yield from _stream_openai(
                system_prompt=system_prompt,
                messages=normalized_messages,
                metrics_collector=metrics_collector,
            )
        else:
            yield from _stream_anthropic(
                system_prompt=system_prompt,
                messages=normalized_messages,
                metrics_collector=metrics_collector,
            )
    except LLMServiceError:
        raise
    except Exception as exc:
        log.error("llm_stream_failed", error=str(exc), provider=settings.LLM_PROVIDER)
        raise LLMServiceError(f"LLM streaming failed: {exc}") from exc


def _stream_openai(
    system_prompt: str,
    messages: list[dict],
    metrics_collector: dict | None = None,
) -> Generator[str, None, None]:
    """Stream chat completion from the OpenAI API and capture usage stats."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=full_messages,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )

    accumulated_content: list[str] = []
    usage_info: dict | None = None
    model_name = settings.LLM_MODEL

    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                accumulated_content.append(delta.content)
                yield delta.content
        if getattr(chunk, "model", None):
            model_name = chunk.model
        if getattr(chunk, "usage", None) and chunk.usage:
            u = chunk.usage
            usage_info = {
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }

    duration = time.perf_counter() - start_time
    if metrics_collector is not None:
        metrics_collector["provider"] = "openai"
        metrics_collector["model"] = model_name
        metrics_collector["duration"] = round(duration, 2)
        if usage_info:
            metrics_collector["usage"] = usage_info
        else:
            full_text = "".join(accumulated_content)
            in_chars = len(system_prompt) + sum(len(m.get("content", "")) for m in messages)
            out_chars = len(full_text)
            metrics_collector["usage"] = {
                "input_tokens": max(1, in_chars // 4),
                "output_tokens": max(1, out_chars // 4),
                "total_tokens": max(1, (in_chars + out_chars) // 4),
            }


def _stream_anthropic(
    system_prompt: str,
    messages: list[dict],
    metrics_collector: dict | None = None,
) -> Generator[str, None, None]:
    """Stream message response from the Anthropic API and capture usage stats."""
    from anthropic import Anthropic

    settings = get_settings()
    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    start_time = time.perf_counter()
    accumulated_content: list[str] = []
    usage_info: dict | None = None
    model_name = settings.LLM_MODEL

    with client.messages.stream(
        model=settings.LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            accumulated_content.append(text)
            yield text

        try:
            final_message = stream.get_final_message()
            if final_message:
                model_name = final_message.model
                if final_message.usage:
                    usage_info = {
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                        "total_tokens": final_message.usage.input_tokens
                        + final_message.usage.output_tokens,
                    }
        except (AttributeError, RuntimeError, Exception) as exc:  # noqa: BLE001
            log.warning("could_not_get_final_anthropic_usage", error=str(exc))

    duration = time.perf_counter() - start_time
    if metrics_collector is not None:
        metrics_collector["provider"] = "anthropic"
        metrics_collector["model"] = model_name
        metrics_collector["duration"] = round(duration, 2)
        if usage_info:
            metrics_collector["usage"] = usage_info
        else:
            full_text = "".join(accumulated_content)
            in_chars = len(system_prompt) + sum(len(m.get("content", "")) for m in messages)
            out_chars = len(full_text)
            metrics_collector["usage"] = {
                "input_tokens": max(1, in_chars // 4),
                "output_tokens": max(1, out_chars // 4),
                "total_tokens": max(1, (in_chars + out_chars) // 4),
            }

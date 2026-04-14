"""LLM wrapper with lazy client creation and offline-friendly imports."""

from __future__ import annotations

from functools import lru_cache

from git_explainer import config

Message = dict[str, str]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None  # type: ignore[assignment]


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM cannot be used."""


@lru_cache(maxsize=1)
def _get_client():
    if OpenAI is None:
        raise LLMUnavailableError("openai package is not installed")
    if not config.has_groq_api_key():
        raise LLMUnavailableError("GROQ_API_KEY is not configured")
    return OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
        max_retries=3,
        timeout=60.0,
    )


def is_available() -> bool:
    return OpenAI is not None and config.has_groq_api_key()


def chat(
    user_content: str,
    *,
    system_prompt: str = "",
    history: list[Message] | None = None,
    model: str = config.GROQ_MODEL,
    max_tokens: int = config.GROQ_MAX_TOKENS,
    temperature: float = 0.3,
) -> str:
    """Send a message to the configured model and return the reply."""
    messages: list[Message] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_content})

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return (response.choices[0].message.content or "").strip()

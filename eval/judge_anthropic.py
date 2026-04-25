"""Anthropic-Claude judge adapter for the evaluation harness.

The eval harness needs an LLM-as-judge that is independent from the
agent's synthesizer (Groq llama-3.3-70b). Using a Claude model from a
different provider removes the same-family circular dependency in the
honesty rubric and gives us a real headline number.

This adapter is eval-only — it is not used by the agent at runtime.
The agent's LLM client (`git_explainer.llm`) is unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import anthropic
except ImportError:  # pragma: no cover — exercised only when the SDK is absent
    anthropic = None  # type: ignore[assignment]

# Cheapest current Claude family member. The user explicitly chose this for
# cost reasons — judge calls send ~3K input + ~150 output tokens per case,
# so 29 cases costs roughly $0.10 on Haiku 4.5 ($1/$5 per million in/out).
_JUDGE_MODEL = "claude-haiku-4-5"

# Strict JSON schema. Using output_config.format with a json_schema means
# the API guarantees a valid JSON object back — this eliminates the
# "judge returned prose" parse-failure mode the Groq path had to retry on.
_JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rating": {
            "type": "string",
            "enum": ["accurate", "partially accurate", "hallucinated"],
        },
        "reasoning": {"type": "string"},
        "contradictions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["rating", "reasoning", "contradictions"],
    "additionalProperties": False,
}


def _resolve_api_key() -> str:
    """Read the Anthropic key from `ANTHROPIC_API_KEY` (the SDK's standard
    var) or `ANTHROPIC_KEY` (an alternate name a user may use)."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def is_available() -> bool:
    """Return True iff the Anthropic SDK is importable and a key is set."""
    return anthropic is not None and bool(_resolve_api_key())


def chat(
    user_content: str,
    *,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> str:
    """Call Claude Haiku 4.5 with a strict JSON schema and return the JSON text.

    Returns the raw JSON string so it can be fed straight into the harness's
    existing `_parse_llm_judge_response` without changes. The schema
    constraint means the response is always a valid JSON object with the
    expected keys, so retries-for-parse-failure are no longer needed.
    """
    if anthropic is None:
        raise RuntimeError(
            "anthropic package is not installed. Run: pip install anthropic"
        )
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError("Neither ANTHROPIC_API_KEY nor ANTHROPIC_KEY is set")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=_JUDGE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt or anthropic.NOT_GIVEN,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": _JUDGE_OUTPUT_SCHEMA,
            }
        },
        messages=[{"role": "user", "content": user_content}],
    )

    text = next(
        (block.text for block in response.content if block.type == "text"),
        "",
    )
    if not text:
        raise RuntimeError("Anthropic judge returned no text content")

    # Sanity-parse so a malformed payload surfaces here instead of as a
    # downstream parse failure with no context.
    json.loads(text)
    return text


def model_id() -> str:
    """Return the Claude model ID this adapter uses (for run logs / receipts)."""
    return _JUDGE_MODEL

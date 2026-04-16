"""Prompt builders for the Git explainer agent."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are a Git history explainer.
Ground every claim in provided evidence only.
Prefer precise, cautious language.
If the evidence does not establish intent, say so explicitly.
Every section must include bracketed citations like [commit:abc1234], [pr:#42], or [issue:#101].
Return valid JSON with keys: what_changed, why, tradeoffs, limitations, summary."""


def build_synthesis_prompt(
    query: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    """Build the prompt passed to the synthesis model."""
    return (
        "Explain why the selected code exists and how it evolved.\n"
        "If the query includes a natural-language question, explain why the resolved code span answers it.\n\n"
        f"Query:\n{json.dumps(query, indent=2, sort_keys=True)}\n\n"
        f"Evidence:\n{json.dumps(evidence, indent=2, sort_keys=True)}\n\n"
        "Write concise sections for what_changed, why, tradeoffs, limitations, and summary."
    )

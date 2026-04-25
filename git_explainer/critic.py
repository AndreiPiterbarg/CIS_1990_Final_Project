"""Critic LLM: grade the synthesized explanation against the evidence.

The Critic runs *after* the synthesis LLM has produced a draft. It asks
a second, independently-housed model (Anthropic Claude Haiku 4.5 by
default -- a different family from the Groq llama synthesis model) two
questions:

1. Does every claim in the explanation trace to the gathered evidence?
2. Is there documented evidence that was *not* used but should have
   been (e.g. a fetched PR whose body the explanation never cites)?

The critic returns one of two verdicts:

* ``"ok"`` -- accept the draft as-is.
* ``"needs_more_evidence"`` -- re-invoke the Planner with the listed
  ``focus_hints`` and re-synthesize once.

A second LLM family is deliberate: scoring a Groq synthesis with a Groq
critic produces correlated errors. Using Claude here gives us genuine
provider diversity, which is the same reason the eval harness's judge
lives in :mod:`eval.judge_anthropic`. This module shares the SDK loading
pattern with that file but is used at runtime, not just in eval.

If the Anthropic SDK is missing, no key is set, or the call fails, the
critic returns ``verdict="skipped"`` and the orchestrator accepts the
synthesis draft without re-planning. The agent always produces a
result.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from git_explainer import config

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only when SDK is absent
    anthropic = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CriticReport:
    """Structured outcome of one critic round."""

    verdict: str  # "ok" | "needs_more_evidence" | "skipped"
    issues: list[str] = field(default_factory=list)
    focus_hints: list[str] = field(default_factory=list)
    reasoning: str = ""
    # ``available`` is False when the critic could not run at all (no
    # SDK, no key, hard error). Distinguished from a successful "ok"
    # verdict so eval logs can tell "agent had no critic" from "critic
    # ran and approved the draft".
    available: bool = True
    error: str | None = None
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "issues": list(self.issues),
            "focus_hints": list(self.focus_hints),
            "reasoning": self.reasoning,
            "available": self.available,
            "error": self.error,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Schema + prompt
# ---------------------------------------------------------------------------


_CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["ok", "needs_more_evidence"],
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific claims in the explanation that are not supported "
                "by the evidence, or contradicted by it. Empty when the "
                "draft is faithful."
            ),
        },
        "focus_hints": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Concrete tool calls or evidence the planner should "
                "fetch on a retry. Each hint is a short imperative like "
                "'fetch issue #42 referenced in PR body' or 'get diff "
                "for commit abc1234'. Empty when verdict is 'ok'."
            ),
        },
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "issues", "focus_hints", "reasoning"],
    "additionalProperties": False,
}


_SYSTEM_PROMPT = """You are the critic in a multi-stage Git history explainer agent.

You receive a draft explanation (five sections: what_changed, why,
tradeoffs, limitations, summary) and the evidence the explanation was
synthesized from (commits, pull requests, issues, diffs, file contexts).

Your job is to decide whether the draft is faithful to the evidence:

1. Every concrete claim must trace to something in the evidence.
   Vague language ("this commit improved performance") is fine if the
   evidence does not establish a specific number; specific claims need
   evidence to back them.
2. Bracketed citations like [commit:abc1234], [pr:#42], or
   [issue:#101] must reference IDs that actually appear in the
   evidence. A citation to a SHA or PR number that was never fetched
   is a hallucinated citation.
3. If the evidence has documented signals the draft did not use --
   for example, an issue body that explains the bug, a PR review
   comment that captures a tradeoff -- and using that signal would
   meaningfully improve the explanation, list it as a focus hint.

Return verdict "ok" iff the draft is acceptable. Return
"needs_more_evidence" iff additional tool calls would materially
improve faithfulness. Be conservative: prefer "ok" when issues are
minor or when the evidence simply does not support a stronger claim.
Re-running the pipeline costs latency.
"""


def _build_user_prompt(
    *,
    query_dict: dict[str, Any],
    explanation: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    return (
        "Query:\n"
        f"{json.dumps(query_dict, indent=2, sort_keys=True, default=str)}\n\n"
        "Draft explanation:\n"
        f"{json.dumps(explanation, indent=2, sort_keys=True, default=str)}\n\n"
        "Evidence available to synthesis:\n"
        f"{json.dumps(_evidence_for_critic(evidence), indent=2, sort_keys=True, default=str)}\n\n"
        "Return the critique JSON now."
    )


def _evidence_for_critic(evidence: dict[str, Any]) -> dict[str, Any]:
    """Trim long-form bodies before sending to the critic.

    The critic does not need the full PR/issue body to verify that the
    draft references the right artifacts. Sending titles, IDs, and
    truncated bodies keeps the critic prompt within Haiku's context
    budget and keeps cost predictable.
    """
    return {
        "commits": [
            {
                "sha": c.get("sha"),
                "full_sha": c.get("full_sha"),
                "message": (c.get("message") or "")[:300],
                "date": c.get("date"),
            }
            for c in evidence.get("commits", [])
        ],
        "pull_requests": [
            {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "state": pr.get("state"),
                "body_excerpt": (pr.get("body") or "")[:600],
                "review_comment_count": len(pr.get("review_comments") or []),
            }
            for pr in evidence.get("pull_requests", [])
        ],
        "issues": [
            {
                "number": iss.get("number"),
                "title": iss.get("title"),
                "state": iss.get("state"),
                "body_excerpt": (iss.get("body") or "")[:600],
                "comment_count": len(iss.get("comments") or []),
            }
            for iss in evidence.get("issues", [])
        ],
        "diffs": [
            {
                "commit_sha": d.get("commit_sha"),
                "hunk_count": len(d.get("hunks", [])),
            }
            for d in evidence.get("diffs", [])
        ],
        "file_contexts": [
            {
                "commit_sha": fc.get("commit_sha"),
                "file_path": fc.get("file_path"),
                "char_count": len(fc.get("content") or ""),
            }
            for fc in evidence.get("file_contexts", [])
        ],
    }


# ---------------------------------------------------------------------------
# Anthropic client wrapper
# ---------------------------------------------------------------------------


def _resolve_api_key() -> str:
    """Read the Anthropic API key from the same env vars the eval judge uses."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def is_available() -> bool:
    """Return True iff the Anthropic SDK is importable and a key is set."""
    return anthropic is not None and bool(_resolve_api_key())


def _call_anthropic_critic(prompt: str) -> str:
    """Send the critic prompt to Claude and return the raw JSON reply.

    Uses the same structured-output mechanism as the eval judge so the
    response is guaranteed to parse. We never retry here -- the schema
    constraint means a parse failure can only come from a hard upstream
    error, which the caller surfaces as ``verdict="skipped"``.
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
        model=config.CRITIC_MODEL,
        max_tokens=config.CRITIC_MAX_TOKENS,
        temperature=0.0,
        system=_SYSTEM_PROMPT,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": _CRITIC_SCHEMA,
            }
        },
        messages=[{"role": "user", "content": prompt}],
    )

    text = next(
        (block.text for block in response.content if block.type == "text"),
        "",
    )
    if not text:
        raise RuntimeError("Anthropic critic returned no text content")
    # Sanity-parse to surface malformed output here rather than downstream.
    json.loads(text)
    return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def critique(
    *,
    query_dict: dict[str, Any],
    explanation: dict[str, Any],
    evidence: dict[str, Any],
    chat_fn=None,
    is_available_fn=None,
) -> CriticReport:
    """Run one critic pass over a synthesized explanation.

    Args:
        query_dict: Dict-shaped, validated query (from
            :func:`ExplainerQuery.to_dict`).
        explanation: The five-section dict from synthesis. Bare
            ``ExplanationSections`` is fine.
        evidence: The evidence dict that was fed to synthesis (after
            condensation).
        chat_fn / is_available_fn: Injected for tests. Default to the
            module's Anthropic SDK wrapper.

    Returns a :class:`CriticReport`. Always returns -- never raises.
    """
    chat_fn = chat_fn or _call_anthropic_critic
    is_available_fn = is_available_fn or is_available

    if not is_available_fn():
        return CriticReport(
            verdict="skipped",
            available=False,
            error="critic_llm_unavailable",
            model=config.CRITIC_MODEL,
        )

    prompt = _build_user_prompt(
        query_dict=query_dict,
        explanation=explanation,
        evidence=evidence,
    )
    try:
        raw = chat_fn(prompt)
        payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 -- transient API/parse failure
        return CriticReport(
            verdict="skipped",
            available=True,
            error=f"critic_call_failed: {exc}",
            model=config.CRITIC_MODEL,
        )

    if not isinstance(payload, dict):
        return CriticReport(
            verdict="skipped",
            available=True,
            error="critic_returned_non_object",
            model=config.CRITIC_MODEL,
        )

    verdict = payload.get("verdict")
    if verdict not in ("ok", "needs_more_evidence"):
        return CriticReport(
            verdict="skipped",
            available=True,
            error=f"critic_returned_invalid_verdict: {verdict!r}",
            model=config.CRITIC_MODEL,
        )

    issues = payload.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    focus_hints = payload.get("focus_hints") or []
    if not isinstance(focus_hints, list):
        focus_hints = []

    return CriticReport(
        verdict=verdict,
        issues=[str(i) for i in issues],
        focus_hints=[str(h) for h in focus_hints],
        reasoning=str(payload.get("reasoning") or "").strip(),
        available=True,
        model=config.CRITIC_MODEL,
    )


__all__ = [
    "CriticReport",
    "critique",
    "is_available",
]

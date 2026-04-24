"""Pre-summarize large evidence payloads before synthesis.

Long PR threads and issue discussions can easily blow past the synthesis
model's context window. This module implements a two-tier condensation
strategy:

* Tier 1 (preferred): ask the configured LLM to summarize long free-form
  fields, preserving citation anchors (commit SHAs, PR/issue numbers,
  file paths) and stated intent.
* Tier 2 (fallback): deterministic head/tail truncation with a visible
  elision marker.

Condensation is intentionally narrow: we only touch PR bodies, PR review
comments, issue bodies, and issue comments. Citation-relevant metadata
(SHAs, numbers, titles, labels, URLs, file contexts, diffs) is preserved
verbatim.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from git_explainer import config, llm

_PRE_SUMMARIZED_PREFIX = "[pre-summarized]\n"
_TRUNCATED_PREFIX = "[truncated]\n"
_HEAD_CHARS = 800
_TAIL_CHARS = 400


@dataclass(slots=True)
class CondensationReport:
    """Summary of what the condenser did (or decided not to do)."""

    original_size: int = 0
    condensed_size: int = 0
    fields_condensed: list[str] = field(default_factory=list)
    method_used: str = "none"  # "none" | "llm" | "heuristic" | "mixed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_size": self.original_size,
            "condensed_size": self.condensed_size,
            "fields_condensed": list(self.fields_condensed),
            "method_used": self.method_used,
        }


def _measure(evidence: dict[str, Any]) -> int:
    """Return the serialized character count of the evidence dict."""
    return len(json.dumps(evidence, sort_keys=True, default=str))


def _summarize_with_llm(text: str, kind: str) -> str | None:
    """Ask the LLM to summarize a field. Returns None on any failure."""
    target = config.EVIDENCE_SUMMARY_TARGET_CHARS
    # Target words roughly equal target_chars / 5 (avg word length).
    target_words = max(80, target // 5)
    prompt = (
        f"Summarize the following {kind} content in roughly {target_words} words.\n"
        "Preserve any mentions of commit SHAs (7-40 hex chars), PR/issue numbers "
        "(#123), file paths, technical trade-offs, and stated intent. Drop "
        "pleasantries, greetings, CI noise, and repeated quoting. Return plain "
        "text only -- no JSON, no markdown fences.\n\n"
        f"Content:\n{text}"
    )
    try:
        reply = llm.chat(
            prompt,
            system_prompt=(
                "You are a concise technical summarizer. Preserve citation "
                "anchors (SHAs, PR/issue numbers, file paths) verbatim."
            ),
            temperature=0.0,
            max_tokens=400,
        )
    except Exception:  # pragma: no cover - any LLM error falls back to heuristic
        return None
    reply = (reply or "").strip()
    if not reply:
        return None
    return reply


def _truncate(text: str) -> str:
    """Head/tail truncation with an explicit elision marker."""
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text
    elided = len(text) - _HEAD_CHARS - _TAIL_CHARS
    head = text[:_HEAD_CHARS]
    tail = text[-_TAIL_CHARS:]
    return f"{head}\n\n[... content truncated: {elided} chars elided ...]\n\n{tail}"


def _condense_field(
    text: str,
    kind: str,
    *,
    use_llm: bool,
) -> tuple[str, str]:
    """Condense a single long text field.

    Returns ``(new_text, method)`` where ``method`` is ``"llm"`` or
    ``"heuristic"``. Falls back to heuristic truncation if the LLM is not
    available or the call fails.
    """
    if use_llm:
        summary = _summarize_with_llm(text, kind)
        if summary:
            return _PRE_SUMMARIZED_PREFIX + summary, "llm"
    return _TRUNCATED_PREFIX + _truncate(text), "heuristic"


def _iter_targets(evidence: dict[str, Any]) -> list[tuple[str, int, Any, str, str]]:
    """Enumerate fields that are candidates for condensation.

    Returns a list of tuples ``(label, length, container, key, kind)`` where
    ``container[key]`` is the string to condense. ``label`` is a
    human-readable identifier used in the report.
    """
    targets: list[tuple[str, int, Any, str, str]] = []

    for pr in evidence.get("pull_requests", []) or []:
        body = pr.get("body") or ""
        if isinstance(body, str) and len(body) >= config.EVIDENCE_FIELD_MAX_CHARS:
            targets.append(
                (f"pr#{pr.get('number', '?')}.body", len(body), pr, "body", "pull request description")
            )
        for idx, comment in enumerate(pr.get("review_comments", []) or []):
            if not isinstance(comment, dict):
                continue
            cbody = comment.get("body") or ""
            if isinstance(cbody, str) and len(cbody) >= config.EVIDENCE_FIELD_MAX_CHARS:
                targets.append(
                    (
                        f"pr#{pr.get('number', '?')}.review_comments[{idx}].body",
                        len(cbody),
                        comment,
                        "body",
                        "pull request review comment",
                    )
                )

    for issue in evidence.get("issues", []) or []:
        body = issue.get("body") or ""
        if isinstance(body, str) and len(body) >= config.EVIDENCE_FIELD_MAX_CHARS:
            targets.append(
                (f"issue#{issue.get('number', '?')}.body", len(body), issue, "body", "issue description")
            )
        for idx, comment in enumerate(issue.get("comments", []) or []):
            if not isinstance(comment, dict):
                continue
            cbody = comment.get("body") or ""
            if isinstance(cbody, str) and len(cbody) >= config.EVIDENCE_FIELD_MAX_CHARS:
                targets.append(
                    (
                        f"issue#{issue.get('number', '?')}.comments[{idx}].body",
                        len(cbody),
                        comment,
                        "body",
                        "issue comment",
                    )
                )

    # Longest first -- biggest wins.
    targets.sort(key=lambda t: t[1], reverse=True)
    return targets


def condense_evidence(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], CondensationReport]:
    """Condense long free-form evidence fields so the payload fits the budget.

    The input dict is never mutated. If the serialized evidence is already
    under :data:`config.EVIDENCE_CHAR_BUDGET` the input is returned unchanged
    (same reference is fine -- callers should treat it as read-only) with a
    report whose ``method_used`` is ``"none"``.
    """
    original_size = _measure(evidence)
    report = CondensationReport(
        original_size=original_size,
        condensed_size=original_size,
        fields_condensed=[],
        method_used="none",
    )

    if original_size <= config.EVIDENCE_CHAR_BUDGET:
        return evidence, report

    condensed = copy.deepcopy(evidence)
    targets = _iter_targets(condensed)
    if not targets:
        # Nothing we know how to shrink; return the copy with unchanged report.
        report.condensed_size = _measure(condensed)
        return condensed, report

    use_llm = llm.is_available()
    methods_used: set[str] = set()

    for label, _length, container, key, kind in targets:
        if _measure(condensed) <= config.EVIDENCE_CHAR_BUDGET:
            break
        original_text = container.get(key, "") or ""
        if not isinstance(original_text, str) or not original_text:
            continue
        new_text, method = _condense_field(original_text, kind, use_llm=use_llm)
        container[key] = new_text
        report.fields_condensed.append(label)
        methods_used.add(method)

    report.condensed_size = _measure(condensed)
    if not methods_used:
        report.method_used = "none"
    elif len(methods_used) == 1:
        report.method_used = next(iter(methods_used))
    else:
        report.method_used = "mixed"

    return condensed, report


__all__ = [
    "CondensationReport",
    "condense_evidence",
]

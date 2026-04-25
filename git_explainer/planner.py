"""Planner LLM: pick the next deterministic tool to run.

This module replaces the orchestrator's hard-coded "for each commit:
fetch its PRs, then comments, then linked issues, then diff" sequence
with an iterative LLM-driven loop. On each turn the Planner sees the
user query, the evidence collected so far, and the audit trail of past
tool calls, and produces a JSON object describing the next action:

    {"action": "call_tool", "tool": "<name>", "arguments": {...}, "reasoning": "..."}

or

    {"action": "done", "reasoning": "..."}

Compared to native tool-use APIs, the JSON-out approach has three
advantages for this codebase:

1. It works on any chat-completions endpoint (Groq's tool-use schema
   is compatible but not universally enabled per model variant), so
   we don't tie the agent to a specific API mode.
2. Mocking the LLM in tests is just ``mock_chat.side_effect = [...]``
   with hand-written JSON strings -- no SDK-specific tool-call object
   construction.
3. Validation failures fall through cleanly: a bad tool name or a
   missing argument is *the same* failure mode as malformed JSON, so
   we have one retry/fallback path instead of three.

If the Planner errors out (transient LLM failure, repeated invalid
actions, or iteration cap exhausted), the orchestrator falls back to
the deterministic ``_collect_evidence`` path -- the agent always
produces a result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from git_explainer import config, llm
from git_explainer.tool_registry import (
    TOOL_SPECS,
    ToolCallContext,
    ToolCallRecord,
    ToolDispatchError,
    _empty_evidence,
    _summarize_result,
    dispatch_tool,
    merge_tool_result,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PlannerResult:
    """Outcome of one full planner-driven evidence-collection run."""

    evidence: dict[str, Any]
    tool_calls: list[ToolCallRecord]
    iterations_used: int
    halted_reason: str
    # ``available`` is False when the Planner LLM was not reachable at
    # all (no key, no SDK, or hard error before any iteration). The
    # orchestrator uses this to decide whether to fall back to the
    # fixed-sequence ``_collect_evidence`` path.
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations_used": self.iterations_used,
            "halted_reason": self.halted_reason,
            "available": self.available,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
        }


class _InvalidPlannerResponse(ValueError):
    """Raised when the Planner LLM's reply does not match our schema."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are the planning brain of a Git history explainer agent.

You decide which deterministic git/GitHub tools to invoke (and in what
order) so a downstream synthesis model has enough evidence to write a
faithful explanation of why a piece of code exists.

Rules:
1. Reply with a SINGLE JSON object. No prose, no markdown fences.
2. Each turn, set "action" to either "call_tool" or "done".
3. When "action" is "call_tool", set "tool" to one of the listed names
   and "arguments" to an object whose keys exactly match that tool's
   input_schema. Do not invent arguments.
4. When "action" is "done", you are confirming that enough evidence has
   been gathered for synthesis -- pick "done" once you have at least
   the relevant commits and any documented PRs/issues.
5. Always include a one-sentence "reasoning" string.
6. Do not repeat a tool call with identical arguments that has already
   succeeded; check the call history first.
7. Prefer cheap LOCAL tools (trace_line_history, get_diff,
   extract_issue_refs) before GITHUB API calls. Only fetch PR/issue
   comments when the body alone is not enough to document intent.
8. Respect the iteration budget. If you cannot produce more useful
   evidence, choose "done" rather than burning iterations.
"""


def _format_tools(tools: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ],
        indent=2,
    )


def _summarize_evidence_for_prompt(evidence: dict[str, Any]) -> dict[str, Any]:
    """Strip evidence down to a compact view the Planner can reason over.

    We do NOT send full PR/issue bodies or diff hunks back to the
    Planner each turn -- that re-bills the entire context window per
    iteration. Instead we send identifiers and short titles, which are
    enough for the Planner to decide what to fetch next.
    """
    return {
        "commits": [
            {
                "sha": c.get("sha"),
                "message": (c.get("message") or "")[:160],
                "date": c.get("date"),
            }
            for c in evidence.get("commits", [])
        ],
        "pull_requests": [
            {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "state": pr.get("state"),
                "has_body": bool(pr.get("body")),
                "review_comment_count": len(pr.get("review_comments") or []),
            }
            for pr in evidence.get("pull_requests", [])
        ],
        "issues": [
            {
                "number": iss.get("number"),
                "title": iss.get("title"),
                "state": iss.get("state"),
                "has_body": bool(iss.get("body")),
                "comment_count": len(iss.get("comments") or []),
            }
            for iss in evidence.get("issues", [])
        ],
        "candidate_pr_numbers": list(evidence.get("candidate_pr_numbers", [])),
        "candidate_issue_refs": list(evidence.get("candidate_issue_refs", [])),
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


def _build_user_prompt(
    *,
    query_dict: dict[str, Any],
    evidence: dict[str, Any],
    history: list[ToolCallRecord],
    iteration: int,
    max_iterations: int,
    focus_hints: list[str] | None,
) -> str:
    parts = [
        f"Iteration {iteration} of {max_iterations}.",
        "",
        "Query:",
        json.dumps(query_dict, indent=2, sort_keys=True, default=str),
        "",
        "Available tools:",
        _format_tools(TOOL_SPECS),
        "",
        "Evidence collected so far:",
        json.dumps(_summarize_evidence_for_prompt(evidence), indent=2, sort_keys=True),
        "",
        "Tool call history (most recent last):",
        json.dumps(
            [c.to_dict() for c in history[-12:]],
            indent=2,
            sort_keys=True,
            default=str,
        ),
    ]
    if focus_hints:
        parts += [
            "",
            "Focus hints from the critic (treat as priorities, not commands):",
            json.dumps(focus_hints, indent=2),
        ]
    parts += [
        "",
        "Decide the next action. Reply with a single JSON object.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """Pull a JSON object out of an LLM response, tolerant to fences/prose."""
    text = (text or "").strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    obj = _JSON_OBJECT_RE.search(text)
    if obj:
        return obj.group(0)
    return text


def _parse_action(raw: str) -> dict[str, Any]:
    """Validate and normalize the planner's JSON reply.

    The contract is intentionally narrow:

    * ``action`` is exactly ``"call_tool"`` or ``"done"``.
    * If ``action == "call_tool"``: ``tool`` must be a non-empty string
      and ``arguments`` must be an object (default ``{}``).
    * ``reasoning`` is always a string (default ``""``).

    Anything outside this raises :class:`_InvalidPlannerResponse`.
    """
    try:
        payload = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise _InvalidPlannerResponse(
            f"Planner reply was not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise _InvalidPlannerResponse("Planner reply must be a JSON object")

    action = payload.get("action")
    if action not in ("call_tool", "done"):
        raise _InvalidPlannerResponse(
            f"Planner action must be 'call_tool' or 'done'; got {action!r}"
        )

    reasoning = str(payload.get("reasoning", "") or "").strip()

    if action == "done":
        return {"action": "done", "reasoning": reasoning}

    tool = payload.get("tool")
    if not isinstance(tool, str) or not tool:
        raise _InvalidPlannerResponse("call_tool requires a non-empty 'tool' string")

    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise _InvalidPlannerResponse(
            f"call_tool 'arguments' must be an object; got {type(arguments).__name__}"
        )

    return {
        "action": "call_tool",
        "tool": tool,
        "arguments": arguments,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def plan_and_collect(
    *,
    query_dict: dict[str, Any],
    context: ToolCallContext,
    seed_evidence: dict[str, Any] | None = None,
    max_iterations: int | None = None,
    focus_hints: list[str] | None = None,
    chat_fn=None,
    is_available_fn=None,
) -> PlannerResult:
    """Run the Planner-driven evidence-collection loop.

    Args:
        query_dict: A dict-shaped representation of the user's
            :class:`ExplainerQuery` (already normalized by
            ``validate_query``). Sent to the Planner verbatim.
        context: The dispatcher side-channel (repo path, owner, repo,
            memory cache).
        seed_evidence: Optional evidence dict to start from. Useful when
            the orchestrator has already run ``trace_line_history`` and
            wants the Planner to continue from there. If ``None`` the
            Planner starts with an empty evidence dict and is expected
            to call a history-tracing tool itself on the first turn.
        max_iterations: Cap on Planner LLM round trips. Defaults to
            ``config.PLANNER_MAX_ITERATIONS``.
        focus_hints: Optional priority list from a previous critic
            round (e.g. ``["fetch issue #42", "get diff for sha abc"]``).
            Folded into the prompt as guidance.
        chat_fn / is_available_fn: Injected for testability. If omitted
            we use ``git_explainer.llm.chat`` and ``llm.is_available``.

    Returns a :class:`PlannerResult`. Always returns -- never raises.
    """
    if max_iterations is None:
        max_iterations = config.PLANNER_MAX_ITERATIONS
    chat_fn = chat_fn or llm.chat
    is_available_fn = is_available_fn or llm.is_available

    evidence: dict[str, Any] = (
        seed_evidence if seed_evidence is not None else _empty_evidence()
    )
    # Ensure the Planner-required scratch lists exist even when the
    # caller seeded with the orchestrator's evidence shape.
    evidence.setdefault("candidate_pr_numbers", [])
    evidence.setdefault("candidate_issue_refs", [])

    history: list[ToolCallRecord] = []

    if not is_available_fn():
        return PlannerResult(
            evidence=evidence,
            tool_calls=history,
            iterations_used=0,
            halted_reason="llm_unavailable",
            available=False,
        )

    consecutive_invalid = 0
    halted_reason = "max_iterations"

    for iteration in range(1, max_iterations + 1):
        prompt = _build_user_prompt(
            query_dict=query_dict,
            evidence=evidence,
            history=history,
            iteration=iteration,
            max_iterations=max_iterations,
            focus_hints=focus_hints,
        )
        try:
            raw = chat_fn(
                prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=config.PLANNER_MAX_TOKENS,
                model=config.PLANNER_MODEL,
            )
        except Exception as exc:  # noqa: BLE001 -- transient LLM failure
            history.append(
                ToolCallRecord(
                    iteration=iteration,
                    tool="<planner>",
                    arguments={},
                    status="error",
                    error=f"planner_llm_error: {exc}",
                )
            )
            halted_reason = "llm_error"
            break

        try:
            action = _parse_action(raw)
        except _InvalidPlannerResponse as exc:
            history.append(
                ToolCallRecord(
                    iteration=iteration,
                    tool="<planner>",
                    arguments={"raw_reply": (raw or "")[:500]},
                    status="error",
                    error=str(exc),
                )
            )
            consecutive_invalid += 1
            if consecutive_invalid >= 2:
                halted_reason = "invalid_action"
                break
            continue
        consecutive_invalid = 0

        if action["action"] == "done":
            history.append(
                ToolCallRecord(
                    iteration=iteration,
                    tool="<done>",
                    arguments={},
                    status="ok",
                    reasoning=action["reasoning"],
                )
            )
            halted_reason = "done"
            break

        tool_name = action["tool"]
        arguments = action["arguments"]
        reasoning = action["reasoning"]

        try:
            result = dispatch_tool(tool_name, arguments, context)
        except ToolDispatchError as exc:
            history.append(
                ToolCallRecord(
                    iteration=iteration,
                    tool=tool_name,
                    arguments=arguments,
                    status="error",
                    error=str(exc),
                    reasoning=reasoning,
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001 -- backend failure
            # Network blip, git error, etc. Treat as a recoverable
            # iteration failure: record it and let the Planner pick a
            # different action next turn.
            history.append(
                ToolCallRecord(
                    iteration=iteration,
                    tool=tool_name,
                    arguments=arguments,
                    status="error",
                    error=f"backend_error: {exc}",
                    reasoning=reasoning,
                )
            )
            continue

        merge_tool_result(evidence, tool_name, arguments, result)
        history.append(
            ToolCallRecord(
                iteration=iteration,
                tool=tool_name,
                arguments=arguments,
                status="ok",
                result_summary=_summarize_result(tool_name, result),
                reasoning=reasoning,
            )
        )

    return PlannerResult(
        evidence=evidence,
        tool_calls=history,
        iterations_used=len([c for c in history if c.tool != "<done>"]),
        halted_reason=halted_reason,
        available=True,
    )


__all__ = [
    "PlannerResult",
    "plan_and_collect",
]

"""Tests for the Planner LLM driver."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from git_explainer.planner import (
    PlannerResult,
    _extract_json_object,
    _parse_action,
    plan_and_collect,
)
from git_explainer.tool_registry import ToolCallContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_mock() -> MagicMock:
    """Build a memory mock whose getters return None by default.

    Without this, ``MagicMock().get_pr(...)`` returns a fresh MagicMock,
    which the dispatcher treats as a cache hit (truthy != None) and
    returns to the planner -- causing downstream merge logic to choke.
    """
    memory = MagicMock()
    memory.get_pr.return_value = None
    memory.get_pr_comments.return_value = None
    memory.get_issue.return_value = None
    memory.get_issue_comments.return_value = None
    memory.get_commit_prs.return_value = None
    memory.get_diff.return_value = None
    memory.get_context.return_value = None
    return memory


def _ctx(memory=None) -> ToolCallContext:
    return ToolCallContext(
        repo_path="/tmp/repo",
        owner="octocat",
        repo_name="hello",
        memory=memory or _make_memory_mock(),
    )


def _make_query() -> dict:
    return {
        "repo_path": "/tmp/repo",
        "file_path": "src/app.py",
        "start_line": 10,
        "end_line": 12,
        "owner": "octocat",
        "repo_name": "hello",
    }


# ---------------------------------------------------------------------------
# JSON extraction + action parsing
# ---------------------------------------------------------------------------


def test_extract_json_strips_markdown_fences():
    text = "```json\n{\"action\": \"done\"}\n```"
    assert _extract_json_object(text) == '{"action": "done"}'


def test_extract_json_handles_bare_object():
    assert _extract_json_object('{"action": "done"}') == '{"action": "done"}'


def test_extract_json_finds_object_in_prose():
    text = "Sure, here is my reply: {\"action\": \"done\"} thanks!"
    assert _extract_json_object(text) == '{"action": "done"}'


def test_parse_action_done():
    action = _parse_action('{"action": "done", "reasoning": "enough"}')
    assert action == {"action": "done", "reasoning": "enough"}


def test_parse_action_call_tool():
    action = _parse_action(
        json.dumps({
            "action": "call_tool",
            "tool": "fetch_pr",
            "arguments": {"pr_number": 42},
            "reasoning": "need PR body",
        })
    )
    assert action["action"] == "call_tool"
    assert action["tool"] == "fetch_pr"
    assert action["arguments"] == {"pr_number": 42}
    assert action["reasoning"] == "need PR body"


def test_parse_action_rejects_unknown_action():
    import pytest

    from git_explainer.planner import _InvalidPlannerResponse

    with pytest.raises(_InvalidPlannerResponse):
        _parse_action('{"action": "spin"}')


def test_parse_action_rejects_call_tool_without_tool_name():
    import pytest

    from git_explainer.planner import _InvalidPlannerResponse

    with pytest.raises(_InvalidPlannerResponse):
        _parse_action('{"action": "call_tool"}')


def test_parse_action_rejects_non_object_arguments():
    import pytest

    from git_explainer.planner import _InvalidPlannerResponse

    with pytest.raises(_InvalidPlannerResponse):
        _parse_action(
            '{"action": "call_tool", "tool": "fetch_pr", "arguments": [1,2]}'
        )


def test_parse_action_rejects_invalid_json():
    import pytest

    from git_explainer.planner import _InvalidPlannerResponse

    with pytest.raises(_InvalidPlannerResponse):
        _parse_action("not json at all")


# ---------------------------------------------------------------------------
# Driver behavior
# ---------------------------------------------------------------------------


def test_plan_and_collect_when_llm_unavailable_returns_empty_and_marks_unavailable():
    result = plan_and_collect(
        query_dict=_make_query(),
        context=_ctx(),
        chat_fn=lambda *a, **k: "",
        is_available_fn=lambda: False,
    )
    assert isinstance(result, PlannerResult)
    assert result.available is False
    assert result.halted_reason == "llm_unavailable"
    assert result.tool_calls == []


def test_plan_and_collect_runs_one_tool_then_done(monkeypatch):
    """Happy path: planner picks fetch_pr, then says done."""
    memory = MagicMock()
    memory.get_pr.return_value = None  # force backend call

    fake_pr = {
        "number": 42,
        "title": "Improve parser",
        "body": "fixes #7",
        "state": "merged",
    }
    monkeypatch.setattr(
        "git_explainer.tool_registry.fetch_pr",
        MagicMock(return_value=fake_pr),
    )

    replies = iter(
        [
            json.dumps({
                "action": "call_tool",
                "tool": "fetch_pr",
                "arguments": {"pr_number": 42},
                "reasoning": "need PR body",
            }),
            json.dumps({"action": "done", "reasoning": "have enough"}),
        ]
    )

    def fake_chat(*args, **kwargs):
        return next(replies)

    result = plan_and_collect(
        query_dict=_make_query(),
        context=_ctx(memory=memory),
        chat_fn=fake_chat,
        is_available_fn=lambda: True,
    )
    assert result.halted_reason == "done"
    assert result.evidence["pull_requests"][0]["number"] == 42
    # 1 successful tool call + 1 <done> marker
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool == "fetch_pr"
    assert result.tool_calls[0].status == "ok"
    assert result.tool_calls[1].tool == "<done>"


def test_plan_and_collect_records_dispatch_error_and_continues():
    """An invalid argument is recorded, but the loop keeps going."""
    replies = iter(
        [
            # First: missing required argument -> dispatch error
            json.dumps({
                "action": "call_tool",
                "tool": "fetch_pr",
                "arguments": {},
                "reasoning": "oops",
            }),
            # Second: clean done
            json.dumps({"action": "done", "reasoning": "give up"}),
        ]
    )

    result = plan_and_collect(
        query_dict=_make_query(),
        context=_ctx(),
        chat_fn=lambda *a, **k: next(replies),
        is_available_fn=lambda: True,
    )
    assert result.halted_reason == "done"
    error_calls = [c for c in result.tool_calls if c.status == "error"]
    assert len(error_calls) == 1
    assert "missing required arguments" in error_calls[0].error


def test_plan_and_collect_two_consecutive_invalid_json_halts():
    """If the LLM returns garbage twice in a row, the planner halts."""
    result = plan_and_collect(
        query_dict=_make_query(),
        context=_ctx(),
        chat_fn=lambda *a, **k: "absolute nonsense",
        is_available_fn=lambda: True,
    )
    assert result.halted_reason == "invalid_action"
    assert all(c.status == "error" for c in result.tool_calls)


def test_plan_and_collect_llm_error_halts_with_reason():
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    result = plan_and_collect(
        query_dict=_make_query(),
        context=_ctx(),
        chat_fn=boom,
        is_available_fn=lambda: True,
    )
    assert result.halted_reason == "llm_error"
    assert result.tool_calls[-1].error.startswith("planner_llm_error")


def test_plan_and_collect_respects_max_iterations():
    """Loop stops at max_iterations even if planner never says done."""
    # Always return a done-arguments mismatch so we keep looping with errors.
    forever = json.dumps({
        "action": "call_tool",
        "tool": "fetch_pr",
        "arguments": {"pr_number": 1},
        "reasoning": "again",
    })

    # Make backend always return None so nothing gets cached and the
    # planner keeps issuing the same call.
    import git_explainer.tool_registry as reg

    original = reg.fetch_pr
    reg.fetch_pr = MagicMock(return_value={"number": 1, "title": "t", "body": "", "state": "open"})
    try:
        result = plan_and_collect(
            query_dict=_make_query(),
            context=_ctx(),
            chat_fn=lambda *a, **k: forever,
            is_available_fn=lambda: True,
            max_iterations=3,
        )
    finally:
        reg.fetch_pr = original

    # After 3 iterations the loop must stop.
    assert result.halted_reason == "max_iterations"
    assert len([c for c in result.tool_calls if c.tool == "fetch_pr"]) == 3


def test_plan_and_collect_seed_evidence_is_preserved():
    seed = {
        "commits": [{"sha": "abc1234", "full_sha": "abc1234abc", "message": "x"}],
        "pull_requests": [],
        "issues": [],
        "file_contexts": [],
        "diffs": [],
    }
    result = plan_and_collect(
        query_dict=_make_query(),
        context=_ctx(),
        seed_evidence=seed,
        chat_fn=lambda *a, **k: '{"action": "done", "reasoning": "ok"}',
        is_available_fn=lambda: True,
    )
    assert result.evidence["commits"] == seed["commits"]


def test_plan_and_collect_passes_focus_hints_into_prompt():
    """Focus hints from the critic must reach the planner prompt."""
    captured: dict = {}

    def capture(prompt, **kwargs):
        captured["prompt"] = prompt
        return '{"action": "done", "reasoning": "noted"}'

    plan_and_collect(
        query_dict=_make_query(),
        context=_ctx(),
        chat_fn=capture,
        is_available_fn=lambda: True,
        focus_hints=["fetch issue #99 referenced in PR body"],
    )
    assert "Focus hints" in captured["prompt"]
    assert "issue #99" in captured["prompt"]

"""Tests for the Critic LLM."""

from __future__ import annotations

import json

from git_explainer.critic import CriticReport, critique


def _query() -> dict:
    return {
        "repo_path": "/tmp/repo",
        "file_path": "src/app.py",
        "start_line": 10,
        "end_line": 12,
    }


def _explanation() -> dict:
    return {
        "what_changed": "Parser changed [commit:abc1234].",
        "why": "Improved parser [pr:#42].",
        "tradeoffs": "Minor [commit:abc1234].",
        "limitations": "Limited to retrieved evidence [commit:abc1234].",
        "summary": "Summary [commit:abc1234].",
    }


def _evidence() -> dict:
    return {
        "commits": [
            {"sha": "abc1234", "full_sha": "abc1234abc", "message": "Fix parser", "date": "2024-01-01"},
        ],
        "pull_requests": [
            {"number": 42, "title": "Improve parser", "body": "fixes #7", "state": "merged"},
        ],
        "issues": [],
        "diffs": [],
        "file_contexts": [],
    }


def test_critique_when_critic_unavailable_returns_skipped():
    report = critique(
        query_dict=_query(),
        explanation=_explanation(),
        evidence=_evidence(),
        chat_fn=lambda prompt: "should not be called",
        is_available_fn=lambda: False,
    )
    assert isinstance(report, CriticReport)
    assert report.verdict == "skipped"
    assert report.available is False
    assert report.error == "critic_llm_unavailable"


def test_critique_ok_verdict_passes_through():
    payload = {
        "verdict": "ok",
        "issues": [],
        "focus_hints": [],
        "reasoning": "draft is faithful",
    }
    report = critique(
        query_dict=_query(),
        explanation=_explanation(),
        evidence=_evidence(),
        chat_fn=lambda prompt: json.dumps(payload),
        is_available_fn=lambda: True,
    )
    assert report.verdict == "ok"
    assert report.issues == []
    assert report.focus_hints == []
    assert report.available is True
    assert report.error is None


def test_critique_needs_more_evidence_returns_focus_hints():
    payload = {
        "verdict": "needs_more_evidence",
        "issues": ["claim about X is unsupported"],
        "focus_hints": ["fetch issue #99 referenced in PR body"],
        "reasoning": "PR mentions issue #99 which was never fetched",
    }
    report = critique(
        query_dict=_query(),
        explanation=_explanation(),
        evidence=_evidence(),
        chat_fn=lambda prompt: json.dumps(payload),
        is_available_fn=lambda: True,
    )
    assert report.verdict == "needs_more_evidence"
    assert report.focus_hints == ["fetch issue #99 referenced in PR body"]
    assert report.issues == ["claim about X is unsupported"]


def test_critique_handles_chat_fn_exception_as_skipped():
    def boom(prompt):
        raise RuntimeError("anthropic 503")

    report = critique(
        query_dict=_query(),
        explanation=_explanation(),
        evidence=_evidence(),
        chat_fn=boom,
        is_available_fn=lambda: True,
    )
    assert report.verdict == "skipped"
    assert report.available is True
    assert "critic_call_failed" in (report.error or "")


def test_critique_handles_invalid_verdict_as_skipped():
    payload = {
        "verdict": "maybe",  # not in our allowed enum
        "issues": [],
        "focus_hints": [],
        "reasoning": "",
    }
    report = critique(
        query_dict=_query(),
        explanation=_explanation(),
        evidence=_evidence(),
        chat_fn=lambda prompt: json.dumps(payload),
        is_available_fn=lambda: True,
    )
    assert report.verdict == "skipped"
    assert "critic_returned_invalid_verdict" in (report.error or "")


def test_critique_handles_non_json_reply_as_skipped():
    report = critique(
        query_dict=_query(),
        explanation=_explanation(),
        evidence=_evidence(),
        chat_fn=lambda prompt: "not JSON at all",
        is_available_fn=lambda: True,
    )
    assert report.verdict == "skipped"
    assert "critic_call_failed" in (report.error or "")


def test_critique_coerces_non_list_issues_and_hints_to_empty():
    """A malformed payload that *technically* parses but has wrong types
    must not crash. We treat list-typed fields as empty when the LLM
    returns something else."""
    payload = {
        "verdict": "ok",
        "issues": "string instead of list",
        "focus_hints": None,
        "reasoning": "ok",
    }
    report = critique(
        query_dict=_query(),
        explanation=_explanation(),
        evidence=_evidence(),
        chat_fn=lambda prompt: json.dumps(payload),
        is_available_fn=lambda: True,
    )
    assert report.verdict == "ok"
    assert report.issues == []
    assert report.focus_hints == []


def test_critique_truncates_long_evidence_in_prompt():
    """The critic prompt must trim PR/issue bodies so cost is bounded."""
    captured: dict = {}

    def capture(prompt):
        captured["prompt"] = prompt
        return json.dumps({
            "verdict": "ok",
            "issues": [],
            "focus_hints": [],
            "reasoning": "",
        })

    huge = "X" * 50_000
    evidence = _evidence()
    evidence["pull_requests"][0]["body"] = huge

    critique(
        query_dict=_query(),
        explanation=_explanation(),
        evidence=evidence,
        chat_fn=capture,
        is_available_fn=lambda: True,
    )

    # The full 50k body must NOT appear verbatim. Only the first 600
    # chars (per ``_evidence_for_critic``) should be sent.
    assert huge not in captured["prompt"]
    assert "X" * 600 in captured["prompt"]


def test_critic_report_to_dict_round_trip():
    report = CriticReport(
        verdict="needs_more_evidence",
        issues=["a"],
        focus_hints=["b"],
        reasoning="c",
        available=True,
        error=None,
        model="claude-haiku-4-5",
    )
    d = report.to_dict()
    assert d["verdict"] == "needs_more_evidence"
    assert d["issues"] == ["a"]
    assert d["focus_hints"] == ["b"]
    assert d["model"] == "claude-haiku-4-5"

"""Tests for the evidence condenser."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from git_explainer import config
from git_explainer.evidence_condenser import (
    CondensationReport,
    condense_evidence,
)


def _make_small_evidence() -> dict:
    return {
        "commits": [{"sha": "abc1234", "full_sha": "abc123456789", "message": "fix"}],
        "pull_requests": [
            {
                "number": 42,
                "title": "Improve parser",
                "body": "Short body referencing #7",
                "state": "merged",
                "review_comments": [{"user": "r", "body": "LGTM"}],
            }
        ],
        "issues": [
            {
                "number": 7,
                "title": "Parser bug",
                "body": "short",
                "state": "open",
                "comments": [],
            }
        ],
        "file_contexts": [],
        "diffs": [],
    }


def _make_large_evidence() -> dict:
    ev = _make_small_evidence()
    # Body with embedded citation anchors the condenser should preserve when
    # using the LLM path (heuristic truncation may drop middle-content).
    long_body = (
        "Intro mentioning commit abc1234 and issue #7 and file src/app.py. "
        + ("filler line " * 1500)
        + " Final note mentioning pr #42 and commit def5678."
    )
    ev["pull_requests"][0]["body"] = long_body
    ev["pull_requests"][0]["review_comments"].append(
        {"user": "alice", "body": "x " * 2000}
    )
    return ev


def test_under_threshold_returns_unchanged_and_reports_none():
    ev = _make_small_evidence()
    # Force a very generous budget so the payload is well under it.
    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 10 ** 9):
        out, report = condense_evidence(ev)
    assert out is ev  # same reference; no copy needed
    assert isinstance(report, CondensationReport)
    assert report.method_used == "none"
    assert report.fields_condensed == []
    assert report.original_size == report.condensed_size


def test_heuristic_truncation_when_llm_unavailable():
    ev = _make_large_evidence()
    original_body = ev["pull_requests"][0]["body"]

    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 1000), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 500), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=False):
        out, report = condense_evidence(ev)

    new_body = out["pull_requests"][0]["body"]
    assert new_body.startswith("[truncated]\n")
    assert "content truncated" in new_body
    assert len(new_body) < len(original_body)
    assert report.method_used == "heuristic"
    assert any("body" in f for f in report.fields_condensed)
    assert report.condensed_size < report.original_size
    # Original dict must not have been mutated.
    assert ev["pull_requests"][0]["body"] == original_body


def test_llm_summarization_when_available():
    ev = _make_large_evidence()
    original_body = ev["pull_requests"][0]["body"]
    # The mocked LLM return preserves citation anchors as the prompt requests.
    mocked_summary = (
        "Short summary: touches commit abc1234, issue #7, and src/app.py; "
        "references pr #42 and commit def5678."
    )

    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 1000), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 500), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=True), \
         patch("git_explainer.evidence_condenser.llm.chat", return_value=mocked_summary):
        out, report = condense_evidence(ev)

    new_body = out["pull_requests"][0]["body"]
    assert new_body.startswith("[pre-summarized]\n")
    assert mocked_summary in new_body
    # Citation anchors preserved via the mocked summary.
    assert "abc1234" in new_body
    assert "#7" in new_body
    assert "src/app.py" in new_body
    assert "#42" in new_body
    assert report.method_used in {"llm", "mixed"}
    assert any("body" in f for f in report.fields_condensed)
    assert ev["pull_requests"][0]["body"] == original_body


def test_llm_failure_falls_back_to_heuristic():
    ev = _make_large_evidence()

    def _boom(*a, **kw):
        raise RuntimeError("LLM down")

    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 1000), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 500), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=True), \
         patch("git_explainer.evidence_condenser.llm.chat", side_effect=_boom):
        out, report = condense_evidence(ev)

    new_body = out["pull_requests"][0]["body"]
    assert new_body.startswith("[truncated]\n")
    assert report.method_used == "heuristic"


def test_long_pr_body_condensed_short_pr_body_untouched():
    ev = _make_small_evidence()
    # Add a SECOND PR whose body is huge. The first PR's body is small and
    # should not be touched.
    ev["pull_requests"].append(
        {
            "number": 99,
            "title": "Big PR",
            "body": "x " * 5000,
            "state": "merged",
            "review_comments": [],
        }
    )
    small_body = ev["pull_requests"][0]["body"]

    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 1000), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 3000), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=False):
        out, report = condense_evidence(ev)

    assert out["pull_requests"][0]["body"] == small_body
    assert out["pull_requests"][1]["body"].startswith("[truncated]\n")
    assert any("pr#99.body" in f for f in report.fields_condensed)
    assert not any("pr#42.body" in f for f in report.fields_condensed)


def test_report_fields_populated():
    ev = _make_large_evidence()
    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 1000), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 500), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=False):
        out, report = condense_evidence(ev)

    d = report.to_dict()
    assert set(d.keys()) == {
        "original_size",
        "condensed_size",
        "fields_condensed",
        "method_used",
    }
    assert d["original_size"] > 0
    assert d["condensed_size"] > 0
    assert d["condensed_size"] <= d["original_size"]
    assert d["method_used"] in {"none", "llm", "heuristic", "mixed"}
    assert isinstance(d["fields_condensed"], list)


def test_llm_path_preserves_citation_anchors_via_prompt():
    """The LLM is invoked with text containing anchors; the mocked summary
    keeps them -- mirrors the real contract ("preserve commit SHAs, PR/issue
    numbers, file paths")."""
    ev = _make_large_evidence()
    captured = {}

    def _fake_chat(user_content, **kw):
        captured["prompt"] = user_content
        return "Summary: commit abc1234 touches src/app.py and links issue #7."

    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 1000), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 500), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=True), \
         patch("git_explainer.evidence_condenser.llm.chat", side_effect=_fake_chat):
        out, _ = condense_evidence(ev)

    # The prompt that went to the LLM told it to preserve SHAs/numbers/paths.
    assert "commit SHA" in captured["prompt"] or "commit SHAs" in captured["prompt"]
    assert "PR/issue numbers" in captured["prompt"]
    assert "file paths" in captured["prompt"]
    # And the returned anchors survive into the condensed evidence.
    new_body = out["pull_requests"][0]["body"]
    assert "abc1234" in new_body
    assert "src/app.py" in new_body
    assert "#7" in new_body


def test_does_not_mutate_input_evidence():
    ev = _make_large_evidence()
    snapshot = json.dumps(ev, sort_keys=True)
    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 1000), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 500), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=False):
        condense_evidence(ev)
    assert json.dumps(ev, sort_keys=True) == snapshot


def test_issue_body_and_comments_condensed():
    ev = _make_small_evidence()
    ev["issues"][0]["body"] = "i " * 3000
    ev["issues"][0]["comments"].append({"user": "u", "body": "c " * 3000})

    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 500), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 500), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=False):
        out, report = condense_evidence(ev)

    assert out["issues"][0]["body"].startswith("[truncated]\n")
    # At least the issue body should be in the report.
    assert any("issue#7.body" in f for f in report.fields_condensed)

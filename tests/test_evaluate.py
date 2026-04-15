"""Tests for eval.evaluate — benchmark loading, scoring, and filtering."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.evaluate import BenchmarkCase, CaseScore, filter_cases, load_benchmark, score_case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_CASE_DICT = {
    "id": "case-1",
    "description": "Test case 1",
    "repo_url": "https://github.com/octocat/hello",
    "owner": "octocat",
    "repo_name": "hello",
    "file_path": "src/app.py",
    "start_line": 10,
    "end_line": 20,
    "max_commits": 5,
    "use_llm": False,
    "expected": {},
    "tags": [],
}


def _make_case(**overrides) -> BenchmarkCase:
    """Build a BenchmarkCase with sensible defaults, overriding any field."""
    data = {**MINIMAL_CASE_DICT, **overrides}
    return BenchmarkCase.from_dict(data)


def _make_result(**overrides) -> dict:
    """Build a minimal ExplanationResult dict, overriding any key."""
    base = {
        "query": {},
        "explanation": {
            "what_changed": "",
            "why": "",
            "tradeoffs": "",
            "limitations": "",
            "summary": "",
        },
        "commits": [],
        "pull_requests": [],
        "issues": [],
        "file_contexts": [],
        "cache_stats": {"hits": 0, "misses": 0, "writes": 0},
        "used_fallback": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# load_benchmark
# ---------------------------------------------------------------------------


class TestLoadBenchmark:
    def test_parses_valid_json(self, tmp_path):
        cases_data = [
            {**MINIMAL_CASE_DICT, "id": "case-1"},
            {**MINIMAL_CASE_DICT, "id": "case-2", "start_line": 30, "end_line": 40},
        ]
        bench_file = tmp_path / "benchmark.json"
        bench_file.write_text(json.dumps(cases_data), encoding="utf-8")

        result = load_benchmark(bench_file)

        assert len(result) == 2
        assert all(isinstance(c, BenchmarkCase) for c in result)
        assert result[0].id == "case-1"
        assert result[1].id == "case-2"
        assert result[1].start_line == 30
        assert result[1].end_line == 40

    def test_empty_file(self, tmp_path):
        bench_file = tmp_path / "benchmark.json"
        bench_file.write_text("[]", encoding="utf-8")

        with pytest.raises(SystemExit):
            load_benchmark(bench_file)

    def test_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.json"

        with pytest.raises(SystemExit):
            load_benchmark(missing)


# ---------------------------------------------------------------------------
# score_case
# ---------------------------------------------------------------------------


class TestScoreCase:
    def test_all_checks_pass(self):
        case = _make_case(
            expected={
                "min_commits": 1,
                "commit_message_contains": ["fix"],
                "pr_numbers": [42],
                "issue_numbers": [7],
                "explanation_contains": ["src/app.py"],
                "used_fallback": True,
            },
        )
        result = _make_result(
            commits=[
                {
                    "sha": "abc1234",
                    "full_sha": "abc1234" * 5,
                    "author": "Alice",
                    "date": "2024-06-15",
                    "message": "Fix bug in parser",
                }
            ],
            pull_requests=[{"number": 42, "title": "Fix parser"}],
            issues=[{"number": 7, "title": "Parser crash"}],
            explanation={
                "what_changed": "Changed src/app.py to fix parser.",
                "why": "Bug report.",
                "tradeoffs": "",
                "limitations": "",
                "summary": "Fixed the parser.",
            },
            used_fallback=True,
        )

        score = score_case(case, result, elapsed=1.5)

        assert score.passed is True
        assert score.case_id == "case-1"
        assert all(score.checks.values())
        assert score.checks["min_commits"] is True
        assert score.checks["commit_message_contains"] is True
        assert score.checks["pr_numbers"] is True
        assert score.checks["issue_numbers"] is True
        assert score.checks["explanation_contains"] is True
        assert score.checks["used_fallback"] is True
        assert score.error is None

    def test_partial_failure(self):
        case = _make_case(
            expected={
                "min_commits": 1,
                "commit_message_contains": ["fix"],
                "pr_numbers": [42],
                "issue_numbers": [7],
                "explanation_contains": ["src/app.py"],
                "used_fallback": True,
            },
        )
        result = _make_result(
            commits=[
                {
                    "sha": "abc1234",
                    "full_sha": "abc1234" * 5,
                    "author": "Alice",
                    "date": "2024-06-15",
                    "message": "Fix bug in parser",
                }
            ],
            pull_requests=[],  # missing PR 42
            issues=[{"number": 7, "title": "Parser crash"}],
            explanation={
                "what_changed": "Changed src/app.py to fix parser.",
                "why": "Bug report.",
                "tradeoffs": "",
                "limitations": "",
                "summary": "Fixed the parser.",
            },
            used_fallback=True,
        )

        score = score_case(case, result, elapsed=2.0)

        assert score.passed is False
        assert score.checks["pr_numbers"] is False
        assert score.checks["min_commits"] is True
        assert score.checks["commit_message_contains"] is True
        assert score.checks["issue_numbers"] is True
        assert score.checks["explanation_contains"] is True
        assert score.checks["used_fallback"] is True

    def test_commit_message_case_insensitive(self):
        case = _make_case(
            expected={"commit_message_contains": ["REFACTOR"]},
        )
        result = _make_result(
            commits=[
                {
                    "sha": "def5678",
                    "full_sha": "def5678" * 5,
                    "author": "Bob",
                    "date": "2024-06-01",
                    "message": "refactor parser",
                }
            ],
        )

        score = score_case(case, result, elapsed=0.5)

        assert score.checks["commit_message_contains"] is True
        assert score.passed is True

    def test_no_expected_fields(self):
        case = _make_case(expected={})
        result = _make_result()

        score = score_case(case, result, elapsed=0.1)

        assert score.passed is True
        assert score.checks == {}

    def test_explanation_contains_searches_all_sections(self):
        case = _make_case(
            expected={"explanation_contains": ["tradeoff detail"]},
        )
        result = _make_result(
            explanation={
                "what_changed": "Changed X.",
                "why": "Because Y.",
                "tradeoffs": "There is a tradeoff detail here.",
                "limitations": "None.",
                "summary": "Summary.",
            },
        )

        score = score_case(case, result, elapsed=0.3)

        assert score.checks["explanation_contains"] is True
        assert score.passed is True


# ---------------------------------------------------------------------------
# filter_cases
# ---------------------------------------------------------------------------


class TestFilterCases:
    def test_filter_by_tags(self):
        cases = [
            _make_case(id="c1", tags=["fallback"]),
            _make_case(id="c2", tags=["llm", "multi-pr"]),
            _make_case(id="c3", tags=["fallback"]),
        ]

        filtered = filter_cases(cases, tags=["fallback"])

        assert len(filtered) == 2
        assert {c.id for c in filtered} == {"c1", "c3"}

    def test_filter_by_ids(self):
        cases = [
            _make_case(id="case-1"),
            _make_case(id="case-2"),
            _make_case(id="case-3"),
        ]

        filtered = filter_cases(cases, ids=["case-1", "case-3"])

        assert len(filtered) == 2
        assert {c.id for c in filtered} == {"case-1", "case-3"}

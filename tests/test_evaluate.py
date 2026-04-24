"""Tests for eval.evaluate — benchmark loading, scoring, and filtering."""

from __future__ import annotations

import json

import pytest

from eval.evaluate import (
    BenchmarkCase,
    CaseScore,
    filter_cases,
    load_benchmark,
    save_results,
    score_case,
    summarize_scores,
)


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
    "question": None,
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
        "diffs": [],
        "cache_stats": {"hits": 0, "misses": 0, "writes": 0},
        "used_fallback": False,
        "resolved_target": None,
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

    def test_parses_question_mode_case(self, tmp_path):
        cases_data = [
            {
                **MINIMAL_CASE_DICT,
                "id": "question-case",
                "file_path": None,
                "start_line": None,
                "end_line": None,
                "question": "Why is requests used for issue lookups?",
            }
        ]
        bench_file = tmp_path / "benchmark.json"
        bench_file.write_text(json.dumps(cases_data), encoding="utf-8")

        result = load_benchmark(bench_file)

        assert len(result) == 1
        assert result[0].question == "Why is requests used for issue lookups?"
        assert result[0].file_path is None
        assert result[0].start_line is None
        assert result[0].end_line is None

    def test_parses_suite_object_with_cases(self, tmp_path):
        bench_file = tmp_path / "benchmark.json"
        bench_file.write_text(
            json.dumps(
                {
                    "name": "suite",
                    "cases": [{**MINIMAL_CASE_DICT, "id": "case-embedded"}],
                }
            ),
            encoding="utf-8",
        )

        result = load_benchmark(bench_file)

        assert len(result) == 1
        assert result[0].id == "case-embedded"


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

    def test_empty_commit_message_targets_are_skipped(self):
        case = _make_case(expected={"commit_message_contains": []})
        result = _make_result()

        score = score_case(case, result, elapsed=0.1)

        assert "commit_message_contains" not in score.checks
        assert score.passed is True

    def test_empty_pr_numbers_requires_no_pull_requests(self):
        case = _make_case(expected={"pr_numbers": []})
        result = _make_result(pull_requests=[{"number": 42, "title": "Unexpected PR"}])

        score = score_case(case, result, elapsed=0.2)

        assert score.checks["pr_numbers"] is False
        assert score.passed is False

    def test_empty_issue_numbers_requires_no_issues(self):
        case = _make_case(expected={"issue_numbers": []})
        result = _make_result(issues=[{"number": 7, "title": "Unexpected issue"}])

        score = score_case(case, result, elapsed=0.2)

        assert score.checks["issue_numbers"] is False
        assert score.passed is False

    def test_explicit_no_pull_requests_overrides_pr_numbers(self):
        case = _make_case(
            expected={
                "expects_no_pull_requests": True,
                "pr_numbers": [42],
            }
        )
        result = _make_result(pull_requests=[])

        score = score_case(case, result, elapsed=0.2)

        assert "pr_numbers" not in score.checks
        assert score.checks["expects_no_pull_requests"] is True
        assert score.passed is True

    def test_explicit_no_issues_overrides_issue_numbers(self):
        case = _make_case(
            expected={
                "expects_no_issues": True,
                "issue_numbers": [7],
            }
        )
        result = _make_result(issues=[])

        score = score_case(case, result, elapsed=0.2)

        assert "issue_numbers" not in score.checks
        assert score.checks["expects_no_issues"] is True
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

    def test_resolved_target_checks_pass(self):
        case = _make_case(
            file_path=None,
            start_line=None,
            end_line=None,
            question="Why is requests used for issue lookups?",
            expected={
                "resolved_file_path": "src/app.py",
                "resolved_matched_terms": ["requests", "issue"],
                "resolved_preview_contains": ["import requests"],
            },
        )
        result = _make_result(
            resolved_target={
                "file_path": "src/app.py",
                "start_line": 1,
                "end_line": 6,
                "score": 12.5,
                "matched_terms": ["requests", "issue"],
                "preview": "import requests\n\ndef fetch_issue():\n    pass",
            }
        )

        score = score_case(case, result, elapsed=0.2)

        assert score.passed is True
        assert score.checks["resolved_file_path"] is True
        assert score.checks["resolved_matched_terms"] is True
        assert score.checks["resolved_preview_contains"] is True

    def test_metrics_include_retrieval_citation_and_faithfulness(self):
        case = _make_case(
            expected={
                "commit_message_contains": ["fix"],
                "pr_numbers": [42],
                "issue_numbers": [7],
            }
        )
        result = _make_result(
            commits=[
                {
                    "sha": "abc1234",
                    "full_sha": "abc1234" + ("0" * 33),
                    "author": "Alice",
                    "date": "2024-06-15",
                    "message": "Fix bug in parser",
                }
            ],
            pull_requests=[{"number": 42, "title": "Fix parser"}],
            issues=[{"number": 7, "title": "Parser crash"}],
            explanation={
                "what_changed": "Changed src/app.py [commit:abc1234].",
                "why": "Bug report [issue:#7].",
                "tradeoffs": "Small behavior change [pr:#42].",
                "limitations": "Only benchmarked on one path [commit:abc1234].",
                "summary": "Parser behavior is now fixed [pr:#42].",
            },
        )

        score = score_case(case, result, elapsed=0.4)

        assert score.metrics["retrieval_target_count"] == 3
        assert score.metrics["retrieval_matched_count"] == 3
        assert score.metrics["retrieval_accuracy"] == 1.0
        assert score.metrics["citation_coverage"] == 1.0
        assert score.metrics["citation_validity"] == 1.0
        assert score.metrics["faithfulness_rubric"]["overall"] == 5.0

    def test_metrics_report_partial_citation_coverage(self):
        case = _make_case(expected={})
        result = _make_result(
            commits=[
                {
                    "sha": "abc1234",
                    "full_sha": "abc1234" + ("0" * 33),
                    "author": "Alice",
                    "date": "2024-06-15",
                    "message": "Refactor parser",
                }
            ],
            explanation={
                "what_changed": "Changed src/app.py [commit:abc1234]. Another uncited sentence.",
                "why": "Because we needed the cleanup [commit:abc1234].",
                "tradeoffs": "Slightly more code [commit:abc1234].",
                "limitations": "Only covers one parser path [commit:abc1234].",
                "summary": "Parser code is cleaner [commit:abc1234].",
            },
        )

        score = score_case(case, result, elapsed=0.6)

        assert score.metrics["citable_sentence_count"] == 6
        assert score.metrics["cited_sentence_count"] == 5
        assert score.metrics["citation_coverage"] == pytest.approx(0.833, abs=0.001)
        assert score.metrics["citation_validity"] == 1.0


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


class TestSummaries:
    def test_summarize_scores_aggregates_metrics(self):
        cases = [_make_case(id="c1"), _make_case(id="c2")]
        scores = [
            CaseScore(
                case_id="c1",
                passed=True,
                checks={"min_commits": True},
                elapsed_seconds=1.0,
                metrics={
                    "retrieval_matched_count": 2,
                    "retrieval_target_count": 2,
                    "cited_sentence_count": 5,
                    "citable_sentence_count": 5,
                    "valid_citation_count": 5,
                    "citation_count": 5,
                    "faithfulness_rubric": {"overall": 5.0},
                },
            ),
            CaseScore(
                case_id="c2",
                passed=False,
                checks={"min_commits": False},
                elapsed_seconds=3.0,
                metrics={
                    "retrieval_matched_count": 1,
                    "retrieval_target_count": 2,
                    "cited_sentence_count": 3,
                    "citable_sentence_count": 5,
                    "valid_citation_count": 3,
                    "citation_count": 4,
                    "faithfulness_rubric": {"overall": 3.0},
                },
            ),
        ]

        summary = summarize_scores(cases, scores, total_elapsed=4.0)

        assert summary["benchmark"] == {"case_count": 2, "repo_count": 1}
        assert summary["counts"] == {
            "passed": 1,
            "failed": 1,
            "errors": 0,
            "skipped": 0,
            "total": 2,
        }
        assert summary["pass_rate"] == 0.5
        assert summary["retrieval"]["accuracy"] == 0.75
        assert summary["citation"]["coverage"] == 0.8
        assert summary["citation"]["validity"] == pytest.approx(0.889, abs=0.001)
        assert summary["faithfulness_rubric"]["average"] == 4.0
        assert summary["latency"]["average_seconds"] == 2.0
        assert summary["latency"]["p50_seconds"] == 2.0
        assert summary["latency"]["p95_seconds"] == pytest.approx(2.9, abs=0.001)

    def test_skipped_scores_excluded_from_pass_rate_and_aggregates(self):
        cases = [_make_case(id="c1"), _make_case(id="c2", use_llm=True)]
        scores = [
            CaseScore(
                case_id="c1",
                passed=True,
                checks={"min_commits": True},
                elapsed_seconds=1.0,
                metrics={
                    "retrieval_matched_count": 2,
                    "retrieval_target_count": 2,
                },
            ),
            CaseScore(
                case_id="c2",
                passed=False,
                checks={},
                elapsed_seconds=0.0,
                skipped=True,
                skip_reason="use_llm=true case skipped under --no-llm",
            ),
        ]

        summary = summarize_scores(cases, scores, total_elapsed=1.0)

        assert summary["counts"] == {
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "skipped": 1,
            "total": 2,
        }
        assert summary["pass_rate"] == 1.0
        assert summary["retrieval"]["matched_count"] == 2
        assert summary["retrieval"]["target_count"] == 2
        assert summary["latency"]["average_seconds"] == 1.0

    def test_save_results_writes_summary_and_cases(self, tmp_path):
        results_file = tmp_path / "results.json"
        summary = {"benchmark": {"case_count": 1, "repo_count": 1}}
        scores = [
            CaseScore(
                case_id="case-1",
                passed=True,
                checks={"min_commits": True},
                elapsed_seconds=0.2,
                metrics={"retrieval_accuracy": 1.0},
            )
        ]

        save_results(scores, summary, results_file)

        payload = json.loads(results_file.read_text(encoding="utf-8"))

        assert payload["summary"] == summary
        assert payload["cases"][0]["case_id"] == "case-1"
        assert payload["cases"][0]["metrics"]["retrieval_accuracy"] == 1.0

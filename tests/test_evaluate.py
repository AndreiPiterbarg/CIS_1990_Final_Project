"""Tests for eval.evaluate — benchmark loading, scoring, and filtering."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from eval.evaluate import (
    BenchmarkCase,
    CaseScore,
    _compute_llm_judge_faithfulness,
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
# expected_commit_shas scoring
# ---------------------------------------------------------------------------


class TestExpectedCommitShas:
    """Tests covering the expected_commit_shas ground-truth matcher."""

    _FULL_SHA_A = "a" * 40
    _FULL_SHA_B = "b" * 40

    def _commit(self, full_sha: str, message: str = "") -> dict:
        return {
            "sha": full_sha[:7],
            "full_sha": full_sha,
            "author": "Dev",
            "date": "2024-01-01",
            "message": message or "Commit",
        }

    def test_full_sha_match(self):
        case = _make_case(
            expected={"expected_commit_shas": [self._FULL_SHA_A]},
        )
        result = _make_result(commits=[self._commit(self._FULL_SHA_A, "Fix thing")])

        score = score_case(case, result, elapsed=0.1)

        assert score.checks["expected_commit_shas"] is True
        assert score.metrics["commit_sha_total"] == 1
        assert score.metrics["commit_sha_matches"] == 1
        assert score.passed is True

    def test_prefix_match_short_sha_in_result(self):
        # Simulate a result that only exposes a 7-char short SHA.
        short = self._FULL_SHA_A[:7]
        case = _make_case(
            expected={"expected_commit_shas": [self._FULL_SHA_A]},
        )
        result = _make_result(
            commits=[{"sha": short, "full_sha": "", "author": "Dev",
                      "date": "2024-01-01", "message": "Fix"}]
        )

        score = score_case(case, result, elapsed=0.1)

        assert score.checks["expected_commit_shas"] is True
        assert score.metrics["commit_sha_matches"] == 1

    def test_miss_when_expected_sha_not_in_commits(self):
        case = _make_case(
            expected={"expected_commit_shas": [self._FULL_SHA_A]},
        )
        result = _make_result(commits=[self._commit(self._FULL_SHA_B, "Other")])

        score = score_case(case, result, elapsed=0.1)

        assert score.checks["expected_commit_shas"] is False
        assert score.metrics["commit_sha_matches"] == 0
        assert score.metrics["commit_sha_total"] == 1
        assert score.passed is False

    def test_multiple_expected_shas_must_all_match(self):
        case = _make_case(
            expected={
                "expected_commit_shas": [self._FULL_SHA_A, self._FULL_SHA_B]
            },
        )
        # Only one of the two expected SHAs is present — should fail.
        result = _make_result(commits=[self._commit(self._FULL_SHA_A, "One")])

        score = score_case(case, result, elapsed=0.1)

        assert score.checks["expected_commit_shas"] is False
        assert score.metrics["commit_sha_matches"] == 1
        assert score.metrics["commit_sha_total"] == 2
        # Retrieval accuracy should reflect the partial match (1/2).
        assert score.metrics["retrieval_accuracy"] == 0.5

    def test_empty_expected_shas_skips_check(self):
        case = _make_case(expected={"expected_commit_shas": []})
        result = _make_result()

        score = score_case(case, result, elapsed=0.1)

        assert "expected_commit_shas" not in score.checks
        # No SHA targets contributed to retrieval.
        assert score.metrics["commit_sha_total"] == 0

    def test_sha_matches_contribute_to_retrieval_accuracy(self):
        case = _make_case(
            expected={
                "commit_message_contains": ["fix"],
                "expected_commit_shas": [self._FULL_SHA_A],
            },
        )
        result = _make_result(
            commits=[self._commit(self._FULL_SHA_A, "Fix bug")],
        )

        score = score_case(case, result, elapsed=0.1)

        # One message target + one SHA target, both hit.
        assert score.metrics["retrieval_target_count"] == 2
        assert score.metrics["retrieval_matched_count"] == 2
        assert score.metrics["retrieval_accuracy"] == 1.0

    def test_benchmark_parses_expected_commit_shas_field(self, tmp_path):
        cases_data = [
            {
                **MINIMAL_CASE_DICT,
                "id": "with-shas",
                "expected": {
                    "min_commits": 1,
                    "expected_commit_shas": [self._FULL_SHA_A],
                },
            }
        ]
        bench_file = tmp_path / "benchmark.json"
        bench_file.write_text(json.dumps(cases_data), encoding="utf-8")

        parsed = load_benchmark(bench_file)

        assert parsed[0].expected["expected_commit_shas"] == [self._FULL_SHA_A]

    def test_summary_aggregates_commit_sha_counts(self):
        cases = [_make_case(id="c1"), _make_case(id="c2")]
        scores = [
            CaseScore(
                case_id="c1",
                passed=True,
                checks={"expected_commit_shas": True},
                elapsed_seconds=0.1,
                metrics={
                    "retrieval_matched_count": 1,
                    "retrieval_target_count": 1,
                    "commit_sha_matches": 1,
                    "commit_sha_total": 1,
                },
            ),
            CaseScore(
                case_id="c2",
                passed=False,
                checks={"expected_commit_shas": False},
                elapsed_seconds=0.2,
                metrics={
                    "retrieval_matched_count": 0,
                    "retrieval_target_count": 1,
                    "commit_sha_matches": 0,
                    "commit_sha_total": 1,
                },
            ),
        ]

        summary = summarize_scores(cases, scores, total_elapsed=0.3)

        assert summary["retrieval"]["commit_sha_matches"] == 1
        assert summary["retrieval"]["commit_sha_total"] == 2
        assert summary["retrieval"]["commit_sha_accuracy"] == 0.5


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


# ---------------------------------------------------------------------------
# LLM-as-judge faithfulness
# ---------------------------------------------------------------------------


def _judge_result() -> dict:
    """Build a result suitable for the LLM judge: has commits, PRs, and explanation."""
    return {
        "query": {},
        "explanation": {
            "what_changed": "Added a new parser module [commit:abc1234].",
            "why": "PR #42 motivated the change [pr:#42].",
            "tradeoffs": "Slightly more code [commit:abc1234].",
            "limitations": "Only tested on happy path [commit:abc1234].",
            "summary": "Parser now exists [commit:abc1234].",
        },
        "commits": [
            {
                "sha": "abc1234",
                "full_sha": "abc1234" + ("0" * 33),
                "author": "Alice",
                "date": "2024-06-15",
                "message": "Add parser module",
            }
        ],
        "pull_requests": [{"number": 42, "title": "Add parser", "body": "Adds a parser."}],
        "issues": [],
        "file_contexts": [],
        "diffs": [],
        "cache_stats": {"hits": 0, "misses": 0, "writes": 0},
        "used_fallback": False,
        "resolved_target": None,
    }


class TestLLMJudgeFaithfulness:
    def test_accurate_rating_parsed(self):
        case = _make_case()
        response = json.dumps(
            {
                "rating": "accurate",
                "reasoning": "The explanation aligns with the PR and commit message.",
                "contradictions": [],
            }
        )
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", return_value=response) as mock_chat,
        ):
            judgment = _compute_llm_judge_faithfulness(_judge_result(), case)

        assert judgment["rating"] == "accurate"
        assert judgment["passes"] is True
        assert judgment["contradictions"] == []
        assert judgment["raw_response"] is None
        assert mock_chat.call_count == 1

    def test_partially_accurate_rating_parsed(self):
        case = _make_case()
        response = json.dumps(
            {
                "rating": "partially accurate",
                "reasoning": "Missing key detail about motivation.",
                "contradictions": ["no mention of PR body's performance note"],
            }
        )
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", return_value=response),
        ):
            judgment = _compute_llm_judge_faithfulness(_judge_result(), case)

        assert judgment["rating"] == "partially accurate"
        assert judgment["passes"] is True
        assert judgment["contradictions"] == ["no mention of PR body's performance note"]

    def test_hallucinated_rating_parsed(self):
        case = _make_case()
        response = json.dumps(
            {
                "rating": "hallucinated",
                "reasoning": "Cites an issue that does not appear in evidence.",
                "contradictions": ["issue #99 not in evidence"],
            }
        )
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", return_value=response),
        ):
            judgment = _compute_llm_judge_faithfulness(_judge_result(), case)

        assert judgment["rating"] == "hallucinated"
        assert judgment["passes"] is False
        assert judgment["contradictions"] == ["issue #99 not in evidence"]

    def test_skipped_when_llm_unavailable(self):
        case = _make_case()
        with (
            patch("eval.evaluate.llm.is_available", return_value=False),
            patch("eval.evaluate.llm.chat") as mock_chat,
        ):
            judgment = _compute_llm_judge_faithfulness(_judge_result(), case)

        assert judgment["rating"] == "skipped"
        assert judgment["passes"] is False
        assert judgment["reasoning"] == "llm unavailable"
        mock_chat.assert_not_called()

    def test_retries_once_on_parse_failure(self):
        case = _make_case()
        good_json = json.dumps(
            {
                "rating": "accurate",
                "reasoning": "Grounded in the evidence.",
                "contradictions": [],
            }
        )
        responses = ["Sure! Here's my rating: the explanation is fine.", good_json]
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", side_effect=responses) as mock_chat,
        ):
            judgment = _compute_llm_judge_faithfulness(_judge_result(), case)

        assert judgment["rating"] == "accurate"
        assert judgment["passes"] is True
        assert judgment["raw_response"] is None
        assert mock_chat.call_count == 2

    def test_double_parse_failure_returns_unscored(self):
        case = _make_case()
        responses = [
            "I cannot rate this because I'm not sure.",
            "Still prose, no JSON here.",
        ]
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", side_effect=responses) as mock_chat,
        ):
            judgment = _compute_llm_judge_faithfulness(_judge_result(), case)

        assert judgment["rating"] == "unscored"
        assert judgment["passes"] is False
        assert judgment["raw_response"] == "Still prose, no JSON here."
        assert mock_chat.call_count == 2

    def test_handles_markdown_fenced_json(self):
        case = _make_case()
        fenced = (
            "```json\n"
            + json.dumps(
                {
                    "rating": "accurate",
                    "reasoning": "Grounded.",
                    "contradictions": [],
                }
            )
            + "\n```"
        )
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", return_value=fenced),
        ):
            judgment = _compute_llm_judge_faithfulness(_judge_result(), case)

        assert judgment["rating"] == "accurate"
        assert judgment["passes"] is True

    def test_unknown_rating_treated_as_parse_failure(self):
        case = _make_case()
        bad_rating = json.dumps(
            {"rating": "mostly right", "reasoning": "...", "contradictions": []}
        )
        good_json = json.dumps(
            {"rating": "partially accurate", "reasoning": "ok", "contradictions": []}
        )
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", side_effect=[bad_rating, good_json]),
        ):
            judgment = _compute_llm_judge_faithfulness(_judge_result(), case)

        assert judgment["rating"] == "partially accurate"

    def test_score_case_includes_judge_when_enabled(self):
        case = _make_case()
        response = json.dumps(
            {
                "rating": "accurate",
                "reasoning": "Matches the evidence.",
                "contradictions": [],
            }
        )
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", return_value=response),
        ):
            score = score_case(case, _judge_result(), elapsed=0.1, use_llm_judge=True)

        assert "llm_judge" in score.metrics
        assert score.metrics["llm_judge"]["rating"] == "accurate"

    def test_score_case_omits_judge_when_disabled(self):
        case = _make_case()
        with patch("eval.evaluate.llm.chat") as mock_chat:
            score = score_case(case, _judge_result(), elapsed=0.1, use_llm_judge=False)

        assert "llm_judge" not in score.metrics
        mock_chat.assert_not_called()

    def test_per_case_min_rating_check_enforced(self):
        case = _make_case(expected={"llm_judge_min_rating": "accurate"})
        response = json.dumps(
            {
                "rating": "partially accurate",
                "reasoning": "Missing detail.",
                "contradictions": [],
            }
        )
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", return_value=response),
        ):
            score = score_case(case, _judge_result(), elapsed=0.1, use_llm_judge=True)

        assert score.checks["llm_judge_min_rating"] is False
        assert score.passed is False

    def test_per_case_min_rating_partial_accepts_partial(self):
        case = _make_case(expected={"llm_judge_min_rating": "partially accurate"})
        response = json.dumps(
            {
                "rating": "partially accurate",
                "reasoning": "Missing detail.",
                "contradictions": [],
            }
        )
        with (
            patch("eval.evaluate.llm.is_available", return_value=True),
            patch("eval.evaluate.llm.chat", return_value=response),
        ):
            score = score_case(case, _judge_result(), elapsed=0.1, use_llm_judge=True)

        assert score.checks["llm_judge_min_rating"] is True


class TestSummarizeLLMJudge:
    def test_aggregates_llm_judge_counts_and_pass_rate(self):
        cases = [_make_case(id=f"c{i}") for i in range(1, 6)]
        scores = [
            CaseScore(
                case_id="c1",
                passed=True,
                checks={},
                elapsed_seconds=0.1,
                metrics={"llm_judge": {"rating": "accurate", "passes": True}},
            ),
            CaseScore(
                case_id="c2",
                passed=True,
                checks={},
                elapsed_seconds=0.1,
                metrics={"llm_judge": {"rating": "partially accurate", "passes": True}},
            ),
            CaseScore(
                case_id="c3",
                passed=False,
                checks={},
                elapsed_seconds=0.1,
                metrics={"llm_judge": {"rating": "hallucinated", "passes": False}},
            ),
            CaseScore(
                case_id="c4",
                passed=False,
                checks={},
                elapsed_seconds=0.1,
                metrics={"llm_judge": {"rating": "unscored", "passes": False}},
            ),
            CaseScore(
                case_id="c5",
                passed=False,
                checks={},
                elapsed_seconds=0.1,
                metrics={"llm_judge": {"rating": "skipped", "passes": False}},
            ),
        ]

        summary = summarize_scores(cases, scores, total_elapsed=0.5)

        judge = summary["llm_judge"]
        assert judge is not None
        assert judge["accurate_count"] == 1
        assert judge["partially_accurate_count"] == 1
        assert judge["hallucinated_count"] == 1
        assert judge["unscored_count"] == 1
        assert judge["skipped_count"] == 1
        assert judge["scored_count"] == 3
        # 2 passing / 3 scored (excludes skipped + unscored)
        assert judge["pass_rate"] == pytest.approx(0.667, abs=0.001)

    def test_summary_omits_llm_judge_when_absent(self):
        cases = [_make_case(id="c1")]
        scores = [
            CaseScore(
                case_id="c1",
                passed=True,
                checks={},
                elapsed_seconds=0.1,
                metrics={},
            ),
        ]

        summary = summarize_scores(cases, scores, total_elapsed=0.1)

        assert summary["llm_judge"] is None

"""Tests for the top-level Git explainer orchestration flow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from git_explainer import config
from git_explainer.guardrails import ExplainerQuery
from git_explainer.orchestrator import GitExplainerAgent
from git_explainer.tools.question_resolver import ResolvedCodeSpan


def _make_query(tmp_path: Path) -> ExplainerQuery:
    repo = tmp_path / "repo"
    repo.mkdir()
    return ExplainerQuery(
        repo_path=str(repo),
        file_path="src/app.py",
        start_line=10,
        end_line=12,
        owner="octocat",
        repo_name="hello",
        max_commits=5,
    )


@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.read_file_at_revision")
@patch("git_explainer.orchestrator.fetch_issue_comments")
@patch("git_explainer.orchestrator.fetch_issue")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_agent_collects_evidence_and_uses_fallback(
    mock_validate,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_fetch_issue,
    mock_fetch_issue_comments,
    mock_read_file,
    mock_get_diff,
    tmp_path,
):
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "Alice", "date": "2024-06-01", "message": "Fix bug #7"},
        {"sha": "def5678", "full_sha": "def567890123", "author": "Bob", "date": "2024-05-20", "message": "Refactor parser"},
    ]
    mock_find_prs.side_effect = [[42], [42]]
    mock_fetch_pr.return_value = {"number": 42, "title": "Improve parser", "body": "Fixes #7", "state": "merged"}
    mock_fetch_pr_comments.return_value = [{"user": "reviewer", "body": "Looks good"}]
    mock_fetch_issue.return_value = {"number": 7, "title": "Parser bug", "body": "Broken edge case", "state": "open"}
    mock_fetch_issue_comments.return_value = [{"user": "maintainer", "body": "Please patch soon"}]
    mock_read_file.return_value = "surrounding context"
    mock_get_diff.return_value = {
        "files": [{"hunks": [{"header": "@@ -1,3 +1,4 @@", "lines": [
            {"type": "add", "content": "new line", "old_line": None, "new_line": 2},
        ]}]}],
    }

    result = GitExplainerAgent(use_llm=False).explain(query)

    assert result["used_fallback"] is True
    assert result["resolved_target"] is None
    assert len(result["pull_requests"]) == 1
    assert len(result["issues"]) == 1
    assert len(result["file_contexts"]) == 2
    assert len(result["diffs"]) >= 1
    assert "[commit:abc1234]" in result["explanation"]["summary"]
    assert "[pr:#42]" in result["explanation"]["why"]
    assert "[issue:#7]" in result["explanation"]["why"]


@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.read_file_at_revision")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_agent_reuses_cache_across_runs(
    mock_validate,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_read_file,
    mock_get_diff,
    tmp_path,
):
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "Alice", "date": "2024-06-01", "message": "Fix parser"},
    ]
    mock_find_prs.return_value = [42]
    mock_fetch_pr.return_value = {"number": 42, "title": "Improve parser", "body": "", "state": "merged"}
    mock_fetch_pr_comments.return_value = []
    mock_read_file.return_value = "context"
    mock_get_diff.return_value = {"files": []}

    agent = GitExplainerAgent(use_llm=False)
    agent.explain(query)
    agent.explain(query)

    assert mock_find_prs.call_count == 1
    assert mock_fetch_pr.call_count == 1
    assert mock_fetch_pr_comments.call_count == 1
    assert mock_get_diff.call_count == 1


@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.resolve_question_to_code")
@patch("git_explainer.orchestrator.validate_query")
def test_agent_resolves_question_before_tracing_history(
    mock_validate,
    mock_resolve_question,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_get_diff,
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    question_query = ExplainerQuery(
        repo_path=str(repo),
        question="Why is requests used for GitHub issue lookups?",
        owner="octocat",
        repo_name="hello",
        max_commits=5,
    )
    resolved_query = ExplainerQuery(
        repo_path=str(repo),
        file_path="src/app.py",
        start_line=8,
        end_line=12,
        question=question_query.question,
        owner="octocat",
        repo_name="hello",
        max_commits=5,
    )
    mock_validate.side_effect = [question_query, resolved_query]
    mock_resolve_question.return_value = ResolvedCodeSpan(
        file_path="src/app.py",
        start_line=8,
        end_line=12,
        score=22.0,
        matched_terms=["requests", "issue"],
        preview="import requests\n\ndef fetch_issue():\n    pass",
    )
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "Alice", "date": "2024-06-01", "message": "feat: add GitHub issue lookup"},
    ]
    mock_find_prs.return_value = []
    mock_fetch_pr.return_value = None
    mock_fetch_pr_comments.return_value = []
    mock_get_diff.return_value = {"files": []}

    result = GitExplainerAgent(use_llm=False).explain(question_query)

    mock_resolve_question.assert_called_once_with(
        str(repo),
        "Why is requests used for GitHub issue lookups?",
        file_path_hint=None,
    )
    mock_trace.assert_called_once_with(str(repo), "src/app.py", 8, 12, max_count=5)
    assert result["resolved_target"] == {
        "file_path": "src/app.py",
        "start_line": 8,
        "end_line": 12,
        "score": 22.0,
        "matched_terms": ["requests", "issue"],
        "preview": "import requests\n\ndef fetch_issue():\n    pass",
    }
    assert result["query"]["question"] == "Why is requests used for GitHub issue lookups?"


@patch("git_explainer.orchestrator.chat")
@patch("git_explainer.orchestrator.is_available")
@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.read_file_at_revision")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_agent_rejects_uncited_llm_output_and_uses_fallback(
    mock_validate,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_read_file,
    mock_get_diff,
    mock_is_available,
    mock_chat,
    tmp_path,
):
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "Alice", "date": "2024-06-01", "message": "Fix parser"},
    ]
    mock_find_prs.return_value = [42]
    mock_fetch_pr.return_value = {"number": 42, "title": "Improve parser", "body": "", "state": "merged"}
    mock_fetch_pr_comments.return_value = []
    mock_read_file.return_value = "context"
    mock_get_diff.return_value = {"files": []}
    mock_is_available.return_value = True
    mock_chat.side_effect = [
        json.dumps({
            "what_changed": "A parser helper changed.",
            "why": "It improved the parser [pr:#42].",
            "tradeoffs": "Trade-offs were minor [commit:abc1234].",
            "limitations": "This only covers the available evidence [commit:abc1234].",
            "summary": "The parser changed [commit:abc1234].",
        }),
        json.dumps({
            "what_changed": "A parser helper changed.",
            "why": "It improved the parser [pr:#42].",
            "tradeoffs": "Trade-offs were minor [commit:abc1234].",
            "limitations": "This only covers the available evidence [commit:abc1234].",
            "summary": "The parser changed [commit:abc1234].",
        }),
    ]

    result = GitExplainerAgent(use_llm=True).explain(query)

    assert result["used_fallback"] is True
    assert mock_chat.call_count == 2
    assert "[commit:abc1234]" in result["explanation"]["what_changed"]
    assert "[pr:#42]" in result["explanation"]["why"]


@patch("git_explainer.orchestrator.chat")
@patch("git_explainer.orchestrator.is_available")
@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.read_file_at_revision")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_orchestrator_condenses_long_evidence_before_synthesis(
    mock_validate,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_read_file,
    mock_get_diff,
    mock_is_available,
    mock_chat,
    tmp_path,
):
    """Long PR body must be pre-summarized in the synthesis prompt but
    returned un-condensed in the ExplanationResult."""
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "Alice", "date": "2024-06-01", "message": "Fix parser"},
    ]
    mock_find_prs.return_value = [42]

    # A genuinely huge PR body so condensation triggers even with the default
    # budget (30k). Cap it modestly to keep tests fast.
    long_pr_body = "start of body. " + ("filler filler filler " * 3000) + " end."
    mock_fetch_pr.return_value = {
        "number": 42,
        "title": "Improve parser",
        "body": long_pr_body,
        "state": "merged",
    }
    mock_fetch_pr_comments.return_value = []
    mock_read_file.return_value = "ctx"
    mock_get_diff.return_value = {"files": []}

    # Synthesis LLM is available and returns a valid cited JSON.
    mock_is_available.return_value = True
    mock_chat.return_value = json.dumps({
        "what_changed": "Parser changed [commit:abc1234].",
        "why": "See [pr:#42].",
        "tradeoffs": "Minor [commit:abc1234].",
        "limitations": "Limited to retrieved evidence [commit:abc1234].",
        "summary": "Summary [commit:abc1234].",
    })

    # Run with a tiny budget so condensation definitely triggers without
    # needing a 30k body. The condenser itself uses the LLM path when the
    # synthesis LLM is available, so we also stub its chat call via the
    # evidence_condenser module.
    with patch.object(config, "EVIDENCE_CHAR_BUDGET", 2000), \
         patch.object(config, "EVIDENCE_FIELD_MAX_CHARS", 1000), \
         patch("git_explainer.evidence_condenser.llm.is_available", return_value=False):
        result = GitExplainerAgent(use_llm=True).explain(query)

    # The ExplanationResult contains the ORIGINAL PR body.
    assert result["pull_requests"][0]["body"] == long_pr_body

    # The condensation report was populated.
    assert result["condensation"] is not None
    assert result["condensation"]["method_used"] == "heuristic"
    assert result["condensation"]["original_size"] > result["condensation"]["condensed_size"]
    assert any("pr#42.body" in f for f in result["condensation"]["fields_condensed"])

    # The synthesis prompt that went to the model saw the CONDENSED body.
    assert mock_chat.call_count >= 1
    synthesis_prompt = mock_chat.call_args_list[0].args[0]
    assert "[truncated]" in synthesis_prompt
    # And the full, massive body should NOT appear verbatim in the prompt.
    assert long_pr_body not in synthesis_prompt

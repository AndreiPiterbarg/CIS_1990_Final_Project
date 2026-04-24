"""Tests for guardrails and query normalization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from git_explainer.guardrails import (
    ExplainerQuery,
    normalize_file_path,
    should_fetch_file_context,
    validate_query,
)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    return repo


def test_normalize_absolute_path_inside_repo(tmp_path):
    repo = _make_repo(tmp_path)
    absolute = repo / "src" / "app.py"
    assert normalize_file_path(str(repo), str(absolute)) == "src/app.py"


def test_normalize_rejects_path_outside_repo(tmp_path):
    repo = _make_repo(tmp_path)
    outside = tmp_path / "elsewhere.py"
    outside.write_text("print('x')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the repository"):
        normalize_file_path(str(repo), str(outside))


@patch("git_explainer.guardrails.run_git")
def test_validate_query_infers_owner_and_repo(mock_git, tmp_path):
    repo = _make_repo(tmp_path)
    mock_git.return_value = "git@github.com:octocat/hello.git\n"

    result = validate_query(
        ExplainerQuery(
            repo_path=str(repo),
            file_path="./src/app.py",
            start_line=2,
            end_line=4,
            enforce_public_repo=False,
        )
    )

    assert result.file_path == "src/app.py"
    assert result.owner == "octocat"
    assert result.repo_name == "hello"


def test_validate_query_rejects_large_ranges(tmp_path):
    repo = _make_repo(tmp_path)

    with pytest.raises(ValueError, match="maximum is 200"):
        validate_query(
            ExplainerQuery(
                repo_path=str(repo),
                file_path="src/app.py",
                start_line=1,
                end_line=205,
            )
        )


def test_validate_query_accepts_question_mode(tmp_path):
    repo = _make_repo(tmp_path)

    result = validate_query(
        ExplainerQuery(
            repo_path=str(repo),
            question="Why does this code fetch surrounding file context?",
            enforce_public_repo=False,
        )
    )

    assert result.file_path is None
    assert result.start_line is None
    assert result.end_line is None
    assert result.question == "Why does this code fetch surrounding file context?"


def test_validate_query_accepts_question_with_file_hint(tmp_path):
    repo = _make_repo(tmp_path)

    result = validate_query(
        ExplainerQuery(
            repo_path=str(repo),
            file_path="./src/app.py",
            question="Why does this file exist?",
            enforce_public_repo=False,
        )
    )

    assert result.file_path == "src/app.py"
    assert result.question == "Why does this file exist?"


def test_validate_query_rejects_question_with_line_numbers(tmp_path):
    repo = _make_repo(tmp_path)

    with pytest.raises(ValueError, match="question mode does not accept"):
        validate_query(
            ExplainerQuery(
                repo_path=str(repo),
                file_path="src/app.py",
                start_line=1,
                end_line=2,
                question="Why is this here?",
            )
        )


def test_should_fetch_file_context_for_generic_message():
    assert should_fetch_file_context("Fix bug")
    assert not should_fetch_file_context(
        "Add structured PR evidence gathering",
        pr_body="This PR explains the motivation in detail and links the rollout plan.",
    )


def test_enforce_public_repo_defaults_to_true():
    """The dataclass default must be True so the public-repo guardrail is
    on by default. Regression guard for the default flip."""
    query = ExplainerQuery(repo_path="/tmp/fake")
    assert query.enforce_public_repo is True


@patch("git_explainer.guardrails.ensure_public_github_repo")
def test_validate_query_calls_ensure_public_github_repo_by_default(
    mock_ensure, tmp_path
):
    """With the new default, a line-range query with owner/repo set should
    invoke the public-repo guardrail."""
    repo = _make_repo(tmp_path)
    mock_ensure.return_value = {"private": False}

    validate_query(
        ExplainerQuery(
            repo_path=str(repo),
            file_path="src/app.py",
            start_line=1,
            end_line=2,
            owner="octocat",
            repo_name="hello",
        )
    )

    mock_ensure.assert_called_once_with("octocat", "hello")


@patch("git_explainer.guardrails.ensure_public_github_repo")
def test_validate_query_skips_public_check_when_opted_out(mock_ensure, tmp_path):
    """Passing ``enforce_public_repo=False`` must skip the guardrail."""
    repo = _make_repo(tmp_path)

    validate_query(
        ExplainerQuery(
            repo_path=str(repo),
            file_path="src/app.py",
            start_line=1,
            end_line=2,
            owner="octocat",
            repo_name="hello",
            enforce_public_repo=False,
        )
    )

    mock_ensure.assert_not_called()

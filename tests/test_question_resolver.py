"""Tests for resolving natural-language questions to code spans."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from git_explainer.tools.question_resolver import resolve_question_to_code


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "src").mkdir()
    return repo


@patch("git_explainer.tools.question_resolver.run_git")
def test_resolve_question_prefers_matching_python_span(mock_git, tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "src" / "github_issue_lookup.py").write_text(
        "\n".join([
            "import requests",
            "",
            "def fetch_issue(url: str) -> dict:",
            "    response = requests.get(url)",
            "    return response.json()",
            "",
        ]),
        encoding="utf-8",
    )
    (repo / "src" / "other.py").write_text(
        "def unrelated():\n    return 'ok'\n",
        encoding="utf-8",
    )
    mock_git.return_value = "src/github_issue_lookup.py\nsrc/other.py\n"

    result = resolve_question_to_code(
        str(repo),
        "Why is requests used for GitHub issue lookups?",
    )

    assert result.file_path == "src/github_issue_lookup.py"
    assert "requests" in result.matched_terms
    assert "issue" in result.matched_terms
    assert "import requests" in result.preview
    assert "fetch_issue" in result.preview


@patch("git_explainer.tools.question_resolver.run_git")
def test_resolve_question_uses_file_hint_as_safe_fallback(mock_git, tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "src" / "app.py").write_text(
        "def hello():\n    return 'hi'\n",
        encoding="utf-8",
    )
    mock_git.return_value = "src/app.py\n"

    result = resolve_question_to_code(
        str(repo),
        "banana widget migration details",
        file_path_hint="src/app.py",
    )

    assert result.file_path == "src/app.py"
    assert result.start_line == 1
    assert result.end_line == 2
    assert result.score == 0.0


@patch("git_explainer.tools.question_resolver.run_git")
def test_resolve_question_prefers_source_files_over_tests(mock_git, tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "src" / "github_issue_lookup.py").write_text(
        "\n".join([
            "import requests",
            "",
            "def fetch_issue(url: str) -> dict:",
            "    response = requests.get(url)",
            "    return response.json()",
            "",
        ]),
        encoding="utf-8",
    )
    (repo / "tests" / "test_orchestrator.py").write_text(
        "\n".join([
            "def test_question_prompt():",
            '    question = "Why is requests used for GitHub issue lookups?"',
            '    assert "GitHub issue lookups" in question',
            "",
        ]),
        encoding="utf-8",
    )
    mock_git.return_value = "src/github_issue_lookup.py\ntests/test_orchestrator.py\n"

    result = resolve_question_to_code(
        str(repo),
        "Why is requests used for GitHub issue lookups?",
    )

    assert result.file_path == "src/github_issue_lookup.py"


@patch("git_explainer.tools.question_resolver.run_git")
@patch("git_explainer.tools.question_resolver.read_file_at_revision")
def test_resolve_question_prefers_head_content_for_tracked_files(
    mock_read_file,
    mock_git,
    tmp_path,
):
    repo = _make_repo(tmp_path)
    mock_git.return_value = "src/github_issue_lookup.py\n"

    def fake_read(_repo_path, file_path, *, revision=None, start_line=None, end_line=None):
        assert start_line is None
        assert end_line is None
        if file_path != "src/github_issue_lookup.py":
            return None
        if revision == "HEAD":
            return "\n".join([
                "import requests",
                "",
                "def fetch_issue(url: str) -> dict:",
                "    response = requests.get(url)",
                "    return response.json()",
                "",
            ])
        return "unrelated placeholder text"

    mock_read_file.side_effect = fake_read

    result = resolve_question_to_code(
        str(repo),
        "Why is requests used for GitHub issue lookups?",
    )

    assert result.file_path == "src/github_issue_lookup.py"
    assert "fetch_issue" in result.preview
    mock_read_file.assert_any_call(str(repo), "src/github_issue_lookup.py", revision="HEAD")

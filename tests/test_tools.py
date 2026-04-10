"""Tests for git_explainer.tools — all git/GitHub calls are mocked."""

from __future__ import annotations

from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# git_utils
# ---------------------------------------------------------------------------
from git_explainer.tools.git_utils import run_git


class TestRunGit:
    def test_invalid_repo_path(self, tmp_path):
        bad = str(tmp_path / "no_such_dir")
        with pytest.raises(ValueError, match="does not exist"):
            run_git(bad, ["status"])

    def test_command_failure(self, tmp_path):
        import subprocess

        with patch("git_explainer.tools.git_utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "git", stderr=b"fatal: bad revision"
            )
            with pytest.raises(ValueError, match="bad revision"):
                run_git(str(tmp_path), ["log"])

    def test_success(self, tmp_path):
        with patch("git_explainer.tools.git_utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="hello\n")
            result = run_git(str(tmp_path), ["status"])
            assert result == "hello\n"


# ---------------------------------------------------------------------------
# git_blame_trace
# ---------------------------------------------------------------------------
from git_explainer.tools.git_blame_trace import get_blame, get_commit_detail, get_commit_log

# Sample porcelain blame output (two lines attributed to different commits)
PORCELAIN_BLAME = dedent("""\
    a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2 10 10 1
    author Alice
    author-mail <alice@example.com>
    author-time 1700000000
    author-tz +0000
    committer Alice
    committer-mail <alice@example.com>
    committer-time 1700000000
    committer-tz +0000
    summary Initial commit
    filename src/app.py
    \tdef hello():
    b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3 11 11 1
    author Bob
    author-mail <bob@example.com>
    author-time 1700100000
    author-tz +0000
    committer Bob
    committer-mail <bob@example.com>
    committer-time 1700100000
    committer-tz +0000
    summary Add greeting
    filename src/app.py
    \t    return "hi"
""")

COMMIT_LOG_OUTPUT = dedent("""\
    abc1234|Alice|2024-06-15|Add greeting
    def5678|Bob|2024-06-01|Initial commit
""")

COMMIT_SHOW_OUTPUT = dedent("""\
    commit abc1234
    Author: Alice <alice@example.com>
    Date:   Sat Jun 15 12:00:00 2024 +0000

        Add greeting

    diff --git a/src/app.py b/src/app.py
    --- a/src/app.py
    +++ b/src/app.py
    @@ -1 +1,2 @@
     def hello():
    +    return "hi"
""")


class TestGetBlame:
    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_full_file_blame(self, mock_git):
        mock_git.return_value = PORCELAIN_BLAME
        result = get_blame("/repo", "src/app.py")

        mock_git.assert_called_once_with("/repo", ["blame", "--porcelain", "src/app.py"])
        assert len(result) == 2

        assert result[0]["sha"] == "a1b2c3d"
        assert result[0]["author"] == "Alice"
        assert result[0]["date"] == "2023-11-14"
        assert result[0]["line"] == 10
        assert result[0]["content"] == "def hello():"

        assert result[1]["sha"] == "b2c3d4e"
        assert result[1]["author"] == "Bob"
        assert result[1]["date"] == "2023-11-16"
        assert result[1]["line"] == 11
        assert result[1]["content"] == '    return "hi"'

    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_line_range(self, mock_git):
        mock_git.return_value = PORCELAIN_BLAME
        get_blame("/repo", "src/app.py", start_line=10, end_line=20)

        mock_git.assert_called_once_with(
            "/repo", ["blame", "--porcelain", "-L10,20", "src/app.py"]
        )

    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_single_line(self, mock_git):
        mock_git.return_value = PORCELAIN_BLAME
        get_blame("/repo", "src/app.py", start_line=10)

        mock_git.assert_called_once_with(
            "/repo", ["blame", "--porcelain", "-L10,10", "src/app.py"]
        )

    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_empty_output(self, mock_git):
        mock_git.return_value = ""
        assert get_blame("/repo", "empty.py") == []

    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_untracked_file_raises(self, mock_git):
        mock_git.side_effect = ValueError("fatal: no such path 'nope.py'")
        with pytest.raises(ValueError, match="no such path"):
            get_blame("/repo", "nope.py")


class TestGetCommitLog:
    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_default_log(self, mock_git):
        mock_git.return_value = COMMIT_LOG_OUTPUT
        result = get_commit_log("/repo", "src/app.py")

        mock_git.assert_called_once_with("/repo", [
            "log", "--format=%h|%an|%ad|%s", "--date=short", "-n10", "--", "src/app.py"
        ])
        assert len(result) == 2
        assert result[0] == {
            "sha": "abc1234", "author": "Alice",
            "date": "2024-06-15", "message": "Add greeting",
        }
        assert result[1] == {
            "sha": "def5678", "author": "Bob",
            "date": "2024-06-01", "message": "Initial commit",
        }

    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_custom_max_count(self, mock_git):
        mock_git.return_value = "abc1234|Alice|2024-06-15|msg\n"
        get_commit_log("/repo", "f.py", max_count=5)

        mock_git.assert_called_once_with("/repo", [
            "log", "--format=%h|%an|%ad|%s", "--date=short", "-n5", "--", "f.py"
        ])

    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_pipe_in_message(self, mock_git):
        mock_git.return_value = "abc1234|Alice|2024-06-15|fix: a|b edge case\n"
        result = get_commit_log("/repo", "f.py")

        assert result[0]["message"] == "fix: a|b edge case"

    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_empty_log(self, mock_git):
        mock_git.return_value = ""
        assert get_commit_log("/repo", "f.py") == []


class TestGetCommitDetail:
    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_returns_raw_output(self, mock_git):
        mock_git.return_value = COMMIT_SHOW_OUTPUT
        result = get_commit_detail("/repo", "abc1234")

        mock_git.assert_called_once_with("/repo", ["show", "abc1234"])
        assert "Add greeting" in result
        assert 'return "hi"' in result

    @patch("git_explainer.tools.git_blame_trace.run_git")
    def test_invalid_sha_raises(self, mock_git):
        mock_git.side_effect = ValueError("fatal: bad object deadbeef")
        with pytest.raises(ValueError, match="bad object"):
            get_commit_detail("/repo", "deadbeef")


# ---------------------------------------------------------------------------
# git_diff_reader
# ---------------------------------------------------------------------------
from git_explainer.tools.git_diff_reader import get_diff, get_diff_stats

# Sample unified diffs used across tests
MODIFIED_DIFF = dedent("""\
    diff --git a/src/app.py b/src/app.py
    index 1234567..abcdefg 100644
    --- a/src/app.py
    +++ b/src/app.py
    @@ -10,6 +10,8 @@ def hello():
         pass
         pass
         pass
    +    # new comment
    +    x = 1
         pass
         pass
         pass
    @@ -30,4 +32,3 @@ def goodbye():
         a = 1
    -    b = 2
    -    c = 3
    +    b = 20
""")

ADDED_FILE_DIFF = dedent("""\
    diff --git a/new_file.py b/new_file.py
    new file mode 100644
    index 0000000..abcdefg
    --- /dev/null
    +++ b/new_file.py
    @@ -0,0 +1,3 @@
    +line1
    +line2
    +line3
""")

DELETED_FILE_DIFF = dedent("""\
    diff --git a/old.py b/old.py
    deleted file mode 100644
    index abcdefg..0000000
    --- a/old.py
    +++ /dev/null
    @@ -1,2 +0,0 @@
    -line1
    -line2
""")

RENAMED_DIFF = dedent("""\
    diff --git a/old_name.py b/new_name.py
    similarity index 95%
    rename from old_name.py
    rename to new_name.py
    index 1234567..abcdefg 100644
    --- a/old_name.py
    +++ b/new_name.py
    @@ -1,3 +1,3 @@
     same
    -old
    +new
     same
""")

BINARY_DIFF = dedent("""\
    diff --git a/image.png b/image.png
    index 1234567..abcdefg 100644
    Binary files a/image.png and b/image.png differ
""")

MULTI_FILE_DIFF = MODIFIED_DIFF + ADDED_FILE_DIFF + DELETED_FILE_DIFF


class TestGetDiff:
    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_single_commit_modified_file(self, mock_git):
        mock_git.return_value = MODIFIED_DIFF
        result = get_diff("/repo", "abc123")

        assert result["total_files_changed"] == 1
        f = result["files"][0]
        assert f["status"] == "modified"
        assert f["old_path"] == "src/app.py"
        assert f["additions"] == 3
        assert f["deletions"] == 2
        assert len(f["hunks"]) == 2

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_added_file(self, mock_git):
        mock_git.return_value = ADDED_FILE_DIFF
        result = get_diff("/repo", "abc123")

        f = result["files"][0]
        assert f["status"] == "added"
        assert f["old_path"] == "/dev/null"
        assert f["additions"] == 3
        assert f["deletions"] == 0

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_deleted_file(self, mock_git):
        mock_git.return_value = DELETED_FILE_DIFF
        result = get_diff("/repo", "abc123")

        f = result["files"][0]
        assert f["status"] == "deleted"
        assert f["new_path"] == "/dev/null"
        assert f["deletions"] == 2

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_renamed_file(self, mock_git):
        mock_git.return_value = RENAMED_DIFF
        result = get_diff("/repo", "abc123")

        f = result["files"][0]
        assert f["status"] == "renamed"
        assert f["old_path"] == "old_name.py"
        assert f["new_path"] == "new_name.py"

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_binary_file(self, mock_git):
        mock_git.return_value = BINARY_DIFF
        result = get_diff("/repo", "abc123")

        f = result["files"][0]
        assert f["is_binary"] is True
        assert f["hunks"] == []

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_empty_diff(self, mock_git):
        mock_git.return_value = ""
        result = get_diff("/repo", "abc123")

        assert result["files"] == []
        assert result["total_additions"] == 0
        assert result["total_deletions"] == 0
        assert result["total_files_changed"] == 0

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_base_revision_constructs_correct_cmd(self, mock_git):
        mock_git.return_value = ""
        get_diff("/repo", "head_sha", base_revision="base_sha")

        args = mock_git.call_args[0]
        cmd = args[1]
        assert "base_sha..head_sha" in " ".join(cmd)

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_file_path_filter(self, mock_git):
        mock_git.return_value = ""
        get_diff("/repo", "abc123", file_path="src/app.py")

        cmd = mock_git.call_args[0][1]
        assert "--" in cmd
        assert "src/app.py" in cmd

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_multiple_files(self, mock_git):
        mock_git.return_value = MULTI_FILE_DIFF
        result = get_diff("/repo", "abc123")

        assert result["total_files_changed"] == 3

    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_hunk_line_numbers(self, mock_git):
        mock_git.return_value = MODIFIED_DIFF
        result = get_diff("/repo", "abc123")

        hunk = result["files"][0]["hunks"][0]
        assert hunk["old_start"] == 10
        assert hunk["new_start"] == 10

        # First 3 lines are context, then 2 additions, then 3 context
        add_lines = [l for l in hunk["lines"] if l["type"] == "add"]
        assert len(add_lines) == 2
        # Additions should have new_line set, old_line None
        for al in add_lines:
            assert al["old_line"] is None
            assert al["new_line"] is not None


class TestGetDiffStats:
    @patch("git_explainer.tools.git_diff_reader.run_git")
    def test_numstat_parsing(self, mock_git):
        mock_git.return_value = "10\t5\tsrc/app.py\n0\t0\tREADME.md\n-\t-\timage.png\n"
        result = get_diff_stats("/repo", "abc123")

        assert len(result) == 3
        assert result[0] == {"file": "src/app.py", "additions": 10, "deletions": 5}
        assert result[2] == {"file": "image.png", "additions": 0, "deletions": 0}


# ---------------------------------------------------------------------------
# commit_search
# ---------------------------------------------------------------------------
from git_explainer.tools.commit_search import count_commits, search_commits

SAMPLE_LOG = dedent("""\
    abc1234|Alice|2024-06-01|Add login feature
    def5678|Bob|2024-05-15|Fix auth bug
    ghi9012|Alice|2024-05-01|Initial commit
""").strip()


class TestSearchCommits:
    @patch("git_explainer.tools.commit_search.run_git")
    def test_search_by_grep(self, mock_git):
        mock_git.return_value = SAMPLE_LOG
        result = search_commits("/repo", grep="login")

        cmd = mock_git.call_args[0][1]
        assert "--grep" in cmd
        assert "login" in cmd
        assert len(result) == 3
        assert result[0]["sha"] == "abc1234"

    @patch("git_explainer.tools.commit_search.run_git")
    def test_search_by_author(self, mock_git):
        mock_git.return_value = SAMPLE_LOG
        search_commits("/repo", author="Alice")

        cmd = mock_git.call_args[0][1]
        assert "--author" in cmd
        assert "Alice" in cmd

    @patch("git_explainer.tools.commit_search.run_git")
    def test_search_by_date_range(self, mock_git):
        mock_git.return_value = SAMPLE_LOG
        search_commits("/repo", since="2024-01-01", until="2024-12-31")

        cmd = mock_git.call_args[0][1]
        assert "--since" in cmd
        assert "--until" in cmd

    @patch("git_explainer.tools.commit_search.run_git")
    def test_search_by_path(self, mock_git):
        mock_git.return_value = SAMPLE_LOG
        search_commits("/repo", path="src/app.py")

        cmd = mock_git.call_args[0][1]
        assert "--" in cmd
        assert "src/app.py" in cmd

    @patch("git_explainer.tools.commit_search.run_git")
    def test_search_all_match(self, mock_git):
        mock_git.return_value = SAMPLE_LOG
        search_commits("/repo", grep="fix", author="Bob", all_match=True)

        cmd = mock_git.call_args[0][1]
        assert "--all-match" in cmd

    def test_no_criteria_raises(self):
        with pytest.raises(ValueError, match="At least one search criterion"):
            search_commits("/repo")

    @patch("git_explainer.tools.commit_search.run_git")
    def test_no_results(self, mock_git):
        mock_git.return_value = ""
        result = search_commits("/repo", grep="nonexistent")
        assert result == []

    @patch("git_explainer.tools.commit_search.run_git")
    def test_search_with_branch(self, mock_git):
        mock_git.return_value = SAMPLE_LOG
        search_commits("/repo", grep="fix", branch="develop")

        cmd = mock_git.call_args[0][1]
        assert "develop" in cmd

    @patch("git_explainer.tools.commit_search.run_git")
    def test_max_count(self, mock_git):
        mock_git.return_value = SAMPLE_LOG
        search_commits("/repo", grep="fix", max_count=10)

        cmd = mock_git.call_args[0][1]
        assert "-n10" in cmd


class TestCountCommits:
    @patch("git_explainer.tools.commit_search.run_git")
    def test_count(self, mock_git):
        mock_git.return_value = "abc1234\ndef5678\nghi9012\n"
        result = count_commits("/repo", grep="fix")
        assert result == 3

    def test_no_criteria_raises(self):
        with pytest.raises(ValueError, match="At least one search criterion"):
            count_commits("/repo")


# ---------------------------------------------------------------------------
# commit_range_analyzer
# ---------------------------------------------------------------------------
from git_explainer.tools.commit_range_analyzer import (
    RangeAnalysis,
    analyze_range,
    list_range_commits,
)

RANGE_LOG = dedent("""\
    abc1234|Alice|2024-06-15|Add feature X
    def5678|Bob|2024-06-10|Fix bug in Y
    ghi9012|Alice|2024-06-01|Refactor Z
""").strip()

RANGE_SHORTSTAT = dedent("""\
    abc1234
     3 files changed, 20 insertions(+), 5 deletions(-)

    def5678
     1 file changed, 3 insertions(+)

    ghi9012
     2 files changed, 10 insertions(+), 8 deletions(-)
""")


class TestListRangeCommits:
    @patch("git_explainer.tools.commit_range_analyzer.run_git")
    def test_basic(self, mock_git):
        mock_git.side_effect = [RANGE_LOG, RANGE_SHORTSTAT]
        result = list_range_commits("/repo", "v1.0..v1.1")

        assert len(result) == 3
        assert result[0]["sha"] == "abc1234"
        assert result[0]["files_changed"] == 3
        assert result[0]["additions"] == 20
        assert result[0]["deletions"] == 5
        assert result[1]["files_changed"] == 1
        assert result[1]["additions"] == 3
        assert result[1]["deletions"] == 0


class TestAnalyzeRange:
    @patch("git_explainer.tools.commit_range_analyzer.fetch_issues")
    @patch("git_explainer.tools.commit_range_analyzer.extract_issue_refs")
    @patch("git_explainer.tools.commit_range_analyzer.fetch_pr")
    @patch("git_explainer.tools.commit_range_analyzer.find_prs_for_commit")
    @patch("git_explainer.tools.commit_range_analyzer.get_diff")
    @patch("git_explainer.tools.commit_range_analyzer.run_git")
    def test_full_analysis(
        self, mock_git, mock_diff, mock_find_prs, mock_fetch_pr,
        mock_extract_issues, mock_fetch_issues
    ):
        mock_git.side_effect = [RANGE_LOG, RANGE_SHORTSTAT]
        mock_diff.return_value = {
            "files": [], "total_additions": 33,
            "total_deletions": 13, "total_files_changed": 4,
        }
        mock_find_prs.side_effect = [[42], [], [42]]
        mock_fetch_pr.return_value = {"number": 42, "title": "Feature X"}
        mock_extract_issues.side_effect = [[1, 2], [1], []]
        mock_fetch_issues.return_value = [
            {"number": 1, "title": "Bug report"},
            {"number": 2, "title": "Feature request"},
        ]

        result = analyze_range(
            "/repo", "v1.0..v1.1", owner="octocat", repo_name="hello"
        )

        assert result["total_commits"] == 3
        assert result["base_revision"] == "v1.0"
        assert result["head_revision"] == "v1.1"
        assert len(result["associated_prs"]) == 1  # deduplicated
        assert len(result["associated_issues"]) == 2
        assert result["date_range"]["earliest"] == "2024-06-01"
        assert result["date_range"]["latest"] == "2024-06-15"

    @patch("git_explainer.tools.commit_range_analyzer.get_diff")
    @patch("git_explainer.tools.commit_range_analyzer.run_git")
    def test_no_github_lookups(self, mock_git, mock_diff):
        mock_git.side_effect = [RANGE_LOG, RANGE_SHORTSTAT]
        mock_diff.return_value = {
            "files": [], "total_additions": 0,
            "total_deletions": 0, "total_files_changed": 0,
        }

        result = analyze_range(
            "/repo", "v1.0..v1.1",
            include_prs=False, include_issues=False,
        )

        assert result["associated_prs"] == []
        assert result["associated_issues"] == []

    def test_missing_owner_raises(self):
        with pytest.raises(ValueError, match="owner and repo_name are required"):
            analyze_range("/repo", "v1.0..v1.1", include_prs=True)

    @patch("git_explainer.tools.commit_range_analyzer.get_diff")
    @patch("git_explainer.tools.commit_range_analyzer.run_git")
    def test_empty_range(self, mock_git, mock_diff):
        mock_git.side_effect = ["", ""]
        mock_diff.return_value = {
            "files": [], "total_additions": 0,
            "total_deletions": 0, "total_files_changed": 0,
        }

        result = analyze_range(
            "/repo", "v1.0..v1.1",
            include_prs=False, include_issues=False,
        )

        assert result["total_commits"] == 0
        assert result["commits"] == []
        assert result["date_range"] == {"earliest": "", "latest": ""}

    @patch("git_explainer.tools.commit_range_analyzer.fetch_pr")
    @patch("git_explainer.tools.commit_range_analyzer.find_prs_for_commit")
    @patch("git_explainer.tools.commit_range_analyzer.get_diff")
    @patch("git_explainer.tools.commit_range_analyzer.run_git")
    def test_deduplicates_prs(
        self, mock_git, mock_diff, mock_find_prs, mock_fetch_pr
    ):
        log_2 = "aaa1111|A|2024-01-02|c1\nbbb2222|B|2024-01-01|c2"
        mock_git.side_effect = [log_2, "aaa1111\n 1 file changed, 1 insertion(+)\n\nbbb2222\n 1 file changed, 1 insertion(+)\n"]
        mock_diff.return_value = {
            "files": [], "total_additions": 0,
            "total_deletions": 0, "total_files_changed": 0,
        }
        mock_find_prs.side_effect = [[99], [99]]
        mock_fetch_pr.return_value = {"number": 99, "title": "Same PR"}

        result = analyze_range(
            "/repo", "a..b", owner="o", repo_name="r",
            include_issues=False,
        )

        assert len(result["associated_prs"]) == 1
        assert mock_fetch_pr.call_count == 1

    @patch("git_explainer.tools.commit_range_analyzer.get_diff")
    @patch("git_explainer.tools.commit_range_analyzer.run_git")
    def test_author_stats(self, mock_git, mock_diff):
        log_5 = "\n".join([
            "a|Alice|2024-01-05|c1",
            "b|Alice|2024-01-04|c2",
            "c|Bob|2024-01-03|c3",
            "d|Alice|2024-01-02|c4",
            "e|Bob|2024-01-01|c5",
        ])
        stat_5 = "\n".join([
            "a", " 1 file changed, 1 insertion(+)", "",
            "b", " 1 file changed, 1 insertion(+)", "",
            "c", " 1 file changed, 1 insertion(+)", "",
            "d", " 1 file changed, 1 insertion(+)", "",
            "e", " 1 file changed, 1 insertion(+)", "",
        ])
        mock_git.side_effect = [log_5, stat_5]
        mock_diff.return_value = {
            "files": [], "total_additions": 0,
            "total_deletions": 0, "total_files_changed": 0,
        }

        result = analyze_range(
            "/repo", "a..b", include_prs=False, include_issues=False
        )

        assert result["authors"][0] == {"name": "Alice", "commits": 3}
        assert result["authors"][1] == {"name": "Bob", "commits": 2}

    @patch("git_explainer.tools.commit_range_analyzer.get_diff")
    @patch("git_explainer.tools.commit_range_analyzer.run_git")
    def test_date_range(self, mock_git, mock_diff):
        mock_git.side_effect = [RANGE_LOG, RANGE_SHORTSTAT]
        mock_diff.return_value = {
            "files": [], "total_additions": 0,
            "total_deletions": 0, "total_files_changed": 0,
        }

        result = analyze_range(
            "/repo", "v1.0..v1.1",
            include_prs=False, include_issues=False,
        )

        assert result["date_range"]["earliest"] == "2024-06-01"
        assert result["date_range"]["latest"] == "2024-06-15"

    def test_invalid_range_spec(self):
        with pytest.raises(ValueError, match="must contain"):
            analyze_range("/repo", "just_a_tag", include_prs=False, include_issues=False)


# ---------------------------------------------------------------------------
# file_context_reader
# ---------------------------------------------------------------------------
from git_explainer.tools.file_context_reader import read_file_at_revision


class TestReadFileAtRevision:
    """Tests for read_file_at_revision — all git calls are mocked."""

    # -- worktree (revision=None) -------------------------------------------

    def test_worktree_reads_utf8_file(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "hello.py").write_text("print('hi')\n", encoding="utf-8")

        result = read_file_at_revision(tmp_path, "hello.py")
        assert result == "print('hi')\n"

    def test_worktree_missing_file_returns_none(self, tmp_path):
        (tmp_path / ".git").mkdir()

        assert read_file_at_revision(tmp_path, "nope.py") is None

    def test_worktree_binary_file_returns_sentinel(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

        assert read_file_at_revision(tmp_path, "img.png") == "[binary file]"

    # -- revision -----------------------------------------------------------

    @patch("git_explainer.tools.file_context_reader.subprocess.run")
    def test_revision_reads_file(self, mock_run, tmp_path):
        (tmp_path / ".git").mkdir()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"line1\nline2\nline3\n",
        )

        result = read_file_at_revision(tmp_path, "app.py", revision="abc123")
        assert result == "line1\nline2\nline3\n"

        cmd = mock_run.call_args[0][0]
        assert cmd == ["git", "show", "abc123:app.py"]

    @patch("git_explainer.tools.file_context_reader.subprocess.run")
    def test_revision_missing_file_returns_none(self, mock_run, tmp_path):
        (tmp_path / ".git").mkdir()
        mock_run.return_value = MagicMock(
            returncode=128,
            stdout=b"",
            stderr=b"fatal: path 'x.py' does not exist in 'abc123'",
        )

        assert read_file_at_revision(tmp_path, "x.py", revision="abc123") is None

    @patch("git_explainer.tools.file_context_reader.subprocess.run")
    def test_revision_invalid_revision_returns_none(self, mock_run, tmp_path):
        (tmp_path / ".git").mkdir()
        mock_run.return_value = MagicMock(
            returncode=128,
            stdout=b"",
            stderr=b"fatal: invalid object name 'badrev'",
        )

        assert read_file_at_revision(tmp_path, "x.py", revision="badrev") is None

    @patch("git_explainer.tools.file_context_reader.subprocess.run")
    def test_revision_binary_file_returns_sentinel(self, mock_run, tmp_path):
        (tmp_path / ".git").mkdir()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"\x89PNG\r\n\x1a\n\x00\x00",
        )

        assert read_file_at_revision(tmp_path, "img.png", revision="abc123") == "[binary file]"

    # -- line range ---------------------------------------------------------

    def test_start_and_end_line(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

        result = read_file_at_revision(tmp_path, "f.txt", start_line=2, end_line=4)
        assert result == "b\nc\nd\n"

    def test_start_line_only(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\nc\n", encoding="utf-8")

        result = read_file_at_revision(tmp_path, "f.txt", start_line=2)
        assert result == "b\nc\n"

    def test_end_line_only(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\nc\n", encoding="utf-8")

        result = read_file_at_revision(tmp_path, "f.txt", end_line=2)
        assert result == "a\nb\n"

    def test_line_range_beyond_file_length(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\n", encoding="utf-8")

        result = read_file_at_revision(tmp_path, "f.txt", start_line=5)
        assert result == ""

    def test_line_range_not_applied_to_binary(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

        result = read_file_at_revision(tmp_path, "img.png", start_line=1, end_line=5)
        assert result == "[binary file]"

    def test_line_range_not_applied_to_none(self, tmp_path):
        (tmp_path / ".git").mkdir()

        result = read_file_at_revision(tmp_path, "nope.py", start_line=1, end_line=5)
        assert result is None

    # -- repo validation ----------------------------------------------------

    def test_invalid_repo_raises_valueerror(self, tmp_path):
        bad = tmp_path / "not_a_repo"
        bad.mkdir()

        with pytest.raises(ValueError, match="Not a git repository"):
            read_file_at_revision(bad, "f.txt")

    def test_nonexistent_path_raises_valueerror(self):
        with pytest.raises(ValueError, match="Not a git repository"):
            read_file_at_revision("/no/such/path", "f.txt")

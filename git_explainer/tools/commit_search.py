"""Search commits by message, author, date range, and/or file path."""

from __future__ import annotations

from typing import TypedDict

from git_explainer.tools.git_utils import run_git


class CommitInfo(TypedDict):
    sha: str       # Abbreviated SHA (7 chars)
    author: str
    date: str      # YYYY-MM-DD
    message: str   # Subject line


def search_commits(
    repo_path: str,
    *,
    grep: str | None = None,
    author: str | None = None,
    since: str | None = None,
    until: str | None = None,
    path: str | None = None,
    branch: str | None = None,
    max_count: int = 50,
    all_match: bool = False,
) -> list[CommitInfo]:
    """Search commits by message, author, date range, and/or file path.

    Args:
        repo_path: Absolute path to the local git repository.
        grep: Filter commits whose message matches this pattern (regex).
        author: Filter by author name or email (regex).
        since: Only commits after this date (e.g. "2024-01-01", "2 weeks ago").
        until: Only commits before this date.
        path: Only commits that touched this file or directory.
        branch: Branch or revision to search from (default: HEAD).
        max_count: Maximum number of commits to return (default 50).
        all_match: If True, ALL filter criteria must match (AND). Default
            is False (OR across grep/author).

    Raises:
        ValueError: If no filter criteria are provided, or if the repo path
            is invalid / git command fails.
    """
    if not any([grep, author, since, until, path]):
        raise ValueError("At least one search criterion must be provided")

    cmd = _build_log_cmd(
        fmt="--format=%h|%an|%ad|%s",
        grep=grep,
        author=author,
        since=since,
        until=until,
        path=path,
        branch=branch,
        max_count=max_count,
        all_match=all_match,
    )

    output = run_git(repo_path, cmd)
    return _parse_log_output(output)


def count_commits(
    repo_path: str,
    *,
    grep: str | None = None,
    author: str | None = None,
    since: str | None = None,
    until: str | None = None,
    path: str | None = None,
    branch: str | None = None,
) -> int:
    """Return the count of matching commits without fetching full details."""
    if not any([grep, author, since, until, path]):
        raise ValueError("At least one search criterion must be provided")

    cmd = _build_log_cmd(
        fmt="--format=%h",
        grep=grep,
        author=author,
        since=since,
        until=until,
        path=path,
        branch=branch,
        max_count=None,
        all_match=False,
    )

    output = run_git(repo_path, cmd)
    lines = [line for line in output.strip().splitlines() if line]
    return len(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_log_cmd(
    *,
    fmt: str,
    grep: str | None,
    author: str | None,
    since: str | None,
    until: str | None,
    path: str | None,
    branch: str | None,
    max_count: int | None,
    all_match: bool,
) -> list[str]:
    cmd = ["log", fmt, "--date=short"]
    if max_count is not None:
        cmd.append(f"-n{max_count}")
    if grep is not None:
        cmd += ["--grep", grep]
    if author is not None:
        cmd += ["--author", author]
    if since is not None:
        cmd += ["--since", since]
    if until is not None:
        cmd += ["--until", until]
    if all_match:
        cmd.append("--all-match")
    if branch is not None:
        cmd.append(branch)
    if path is not None:
        cmd += ["--", path]
    return cmd


def _parse_log_output(output: str) -> list[CommitInfo]:
    entries: list[CommitInfo] = []
    for line in output.strip().splitlines():
        if not line:
            continue
        sha, author, date, message = line.split("|", 3)
        entries.append(CommitInfo(sha=sha, author=author, date=date, message=message))
    return entries

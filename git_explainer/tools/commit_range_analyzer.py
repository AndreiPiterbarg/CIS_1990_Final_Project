"""Analyze a range of commits with associated PRs and issues."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TypedDict

from git_explainer.tools.git_diff_reader import DiffSummary, get_diff
from git_explainer.tools.git_utils import run_git
from git_explainer.tools.github_issue_lookup import extract_issue_refs, fetch_issues
from git_explainer.tools.github_pr_lookup import fetch_pr, find_prs_for_commit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class RangeCommit(TypedDict):
    sha: str
    author: str
    date: str
    message: str
    files_changed: int
    additions: int
    deletions: int


class RangeAnalysis(TypedDict):
    base_revision: str
    head_revision: str
    total_commits: int
    commits: list[RangeCommit]
    aggregate_diff: DiffSummary
    authors: list[dict[str, str | int]]
    associated_prs: list[dict]
    associated_issues: list[dict]
    date_range: dict[str, str]


# ---------------------------------------------------------------------------
# Regex for parsing --shortstat output
# ---------------------------------------------------------------------------

_SHORTSTAT_RE = re.compile(
    r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_range(
    repo_path: str,
    range_spec: str,
    *,
    owner: str | None = None,
    repo_name: str | None = None,
    include_prs: bool = True,
    include_issues: bool = True,
    max_commits: int = 200,
) -> RangeAnalysis:
    """Analyze a range of commits and gather associated metadata.

    Args:
        repo_path: Absolute path to the local git repository.
        range_spec: A git revision range, e.g. ``"v1.0..v1.1"`` or
            ``"main..feature-branch"``.  Must contain ``..``.
        owner: GitHub repository owner.  Required when *include_prs* or
            *include_issues* is True.
        repo_name: GitHub repository name.  Required when *include_prs* or
            *include_issues* is True.
        include_prs: Look up associated PRs via GitHub API.
        include_issues: Extract and fetch issue references from commit
            messages.
        max_commits: Safety limit on commits to process (default 200).

    Raises:
        ValueError: If *range_spec* has no ``..``, repo path is invalid,
            or GitHub owner/repo are missing when lookups are requested.
    """
    base_revision, head_revision = _parse_range_spec(range_spec)

    if (include_prs or include_issues) and (owner is None or repo_name is None):
        raise ValueError(
            "owner and repo_name are required when include_prs or "
            "include_issues is True"
        )

    commits = list_range_commits(repo_path, range_spec, max_count=max_commits)

    # Aggregate diff across the full range
    aggregate_diff = get_diff(
        repo_path, head_revision, base_revision=base_revision
    )

    # Author statistics
    author_counts = Counter(c["author"] for c in commits)
    authors: list[dict[str, str | int]] = [
        {"name": name, "commits": count}
        for name, count in author_counts.most_common()
    ]

    # Date range
    if commits:
        date_range = {
            "earliest": commits[-1]["date"],  # oldest (git log is newest-first)
            "latest": commits[0]["date"],
        }
    else:
        date_range = {"earliest": "", "latest": ""}

    # GitHub enrichment
    associated_prs = _fetch_associated_prs(
        commits, owner, repo_name  # type: ignore[arg-type]
    ) if include_prs else []

    associated_issues = _fetch_associated_issues(
        commits, owner, repo_name  # type: ignore[arg-type]
    ) if include_issues else []

    return RangeAnalysis(
        base_revision=base_revision,
        head_revision=head_revision,
        total_commits=len(commits),
        commits=commits,
        aggregate_diff=aggregate_diff,
        authors=authors,
        associated_prs=associated_prs,
        associated_issues=associated_issues,
        date_range=date_range,
    )


def list_range_commits(
    repo_path: str,
    range_spec: str,
    *,
    max_count: int = 200,
) -> list[RangeCommit]:
    """List commits in a range with per-commit stats.

    A lightweight alternative to :func:`analyze_range` when you only need
    the commit list and diff stats without GitHub enrichment.
    """
    # 1) Get commit metadata
    log_output = run_git(repo_path, [
        "log", "--format=%h|%an|%ad|%s", "--date=short",
        f"-n{max_count}", range_spec,
    ])

    raw_commits: list[dict[str, str]] = []
    for line in log_output.strip().splitlines():
        if not line:
            continue
        sha, author, date, message = line.split("|", 3)
        raw_commits.append(
            {"sha": sha, "author": author, "date": date, "message": message}
        )

    if not raw_commits:
        return []

    # 2) Get per-commit shortstat
    stat_output = run_git(repo_path, [
        "log", "--format=%h", "--shortstat",
        f"-n{max_count}", range_spec,
    ])

    stat_map = _parse_shortstat_output(stat_output)

    # 3) Merge
    results: list[RangeCommit] = []
    for c in raw_commits:
        stats = stat_map.get(c["sha"], (0, 0, 0))
        results.append(RangeCommit(
            sha=c["sha"],
            author=c["author"],
            date=c["date"],
            message=c["message"],
            files_changed=stats[0],
            additions=stats[1],
            deletions=stats[2],
        ))

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_range_spec(range_spec: str) -> tuple[str, str]:
    """Split a range spec on ``..`` and return (base, head)."""
    for sep in ("...", ".."):
        if sep in range_spec:
            parts = range_spec.split(sep, 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[0], parts[1]
    raise ValueError(
        f"range_spec must contain '..' to specify a range, got: {range_spec!r}"
    )


def _parse_shortstat_output(output: str) -> dict[str, tuple[int, int, int]]:
    """Parse interleaved sha / shortstat lines into a dict.

    Returns {sha: (files_changed, additions, deletions)}.
    """
    stat_map: dict[str, tuple[int, int, int]] = {}
    current_sha = ""

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        m = _SHORTSTAT_RE.search(stripped)
        if m:
            files = int(m.group(1))
            adds = int(m.group(2)) if m.group(2) else 0
            dels = int(m.group(3)) if m.group(3) else 0
            if current_sha:
                stat_map[current_sha] = (files, adds, dels)
        else:
            # Assume it's a SHA line
            current_sha = stripped

    return stat_map


def _fetch_associated_prs(
    commits: list[RangeCommit],
    owner: str,
    repo_name: str,
) -> list[dict]:
    """Fetch PRs associated with commits, deduplicated."""
    seen: set[int] = set()
    results: list[dict] = []

    for commit in commits:
        try:
            pr_numbers = find_prs_for_commit(owner, repo_name, commit["sha"])
        except Exception:
            logger.warning("Failed to look up PRs for commit %s", commit["sha"])
            continue

        for pr_num in pr_numbers:
            if pr_num in seen:
                continue
            seen.add(pr_num)
            try:
                pr_data = fetch_pr(owner, repo_name, pr_num)
                if pr_data is not None:
                    results.append(pr_data)
            except Exception:
                logger.warning("Failed to fetch PR #%d", pr_num)

    return results


def _fetch_associated_issues(
    commits: list[RangeCommit],
    owner: str,
    repo_name: str,
) -> list[dict]:
    """Extract issue references from commit messages and fetch them."""
    all_issue_nums: set[int] = set()
    for commit in commits:
        all_issue_nums.update(extract_issue_refs(commit["message"]))

    if not all_issue_nums:
        return []

    try:
        return fetch_issues(owner, repo_name, sorted(all_issue_nums))
    except Exception:
        logger.warning("Failed to fetch issues: %s", sorted(all_issue_nums))
        return []

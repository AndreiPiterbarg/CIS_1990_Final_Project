"""Git-explainer tool modules."""

from git_explainer.tools.git_utils import run_git

from git_explainer.tools.file_context_reader import read_file_at_revision
from git_explainer.tools.git_blame_trace import get_blame, get_commit_detail, get_commit_log
from git_explainer.tools.git_diff_reader import get_diff, get_diff_stats
from git_explainer.tools.commit_search import count_commits, search_commits
from git_explainer.tools.commit_range_analyzer import analyze_range, list_range_commits

from git_explainer.tools.github_issue_lookup import extract_issue_refs, fetch_issue, fetch_issues
from git_explainer.tools.github_pr_lookup import (
    fetch_pr,
    fetch_pr_comments,
    find_prs_for_commit,
)

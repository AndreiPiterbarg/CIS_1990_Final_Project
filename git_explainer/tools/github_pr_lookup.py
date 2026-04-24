"""Fetch GitHub pull-request data via the REST API."""

from __future__ import annotations

from typing import Any

import requests

from git_explainer.config import GITHUB_API_BASE, github_headers
from git_explainer.tools.github_http import GitHubResponse, github_get_json

_HEADERS = github_headers()


def _interpret(resp: GitHubResponse) -> GitHubResponse | None:
    """Map HTTP status codes to the legacy success/None/raise contract."""
    if resp.status_code == 404:
        return None
    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"GitHub auth failed – check GITHUB_TOKEN ({resp.status_code})"
        )
    if resp.status_code == 429:
        raise RuntimeError("GitHub rate limit exceeded – retry later")
    if resp.status_code not in (200, 304):
        # Mimic requests.Response.raise_for_status() for unexpected codes so
        # callers keep seeing HTTPError on 5xx that exhausted retries.
        raise requests.HTTPError(
            f"GitHub request failed with status {resp.status_code}"
        )
    return resp


def fetch_pr(
    owner: str,
    repo: str,
    pr_number: int,
    *,
    memory: Any = None,
) -> dict | None:
    """Fetch a single pull request. Returns None if not found."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
    resp = _interpret(_get(url, memory=memory))
    if resp is None:
        return None
    data = resp.data or {}
    state = "merged" if data.get("merged") else data["state"]
    return {
        "number": data["number"],
        "title": data["title"],
        "state": state,
        "body": data.get("body") or "",
        "user": data["user"]["login"],
        "created_at": data["created_at"],
        "merged_at": data.get("merged_at"),
        "base_branch": data["base"]["ref"],
        "head_branch": data["head"]["ref"],
        "merge_commit_sha": data.get("merge_commit_sha"),
    }


def fetch_pr_comments(
    owner: str,
    repo: str,
    pr_number: int,
    *,
    memory: Any = None,
) -> list[dict]:
    """Fetch review comments on a PR (first page, up to 30)."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    resp = _interpret(_get(url, memory=memory))
    if resp is None:
        return []
    payload = resp.data or []
    return [
        {
            "user": c["user"]["login"],
            "body": c["body"],
            "path": c["path"],
            "created_at": c["created_at"],
        }
        for c in payload
    ]


def find_prs_for_commit(
    owner: str,
    repo: str,
    sha: str,
    *,
    memory: Any = None,
) -> list[int]:
    """Find which PR(s) a commit belongs to. Returns list of PR numbers."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}/pulls"
    resp = _interpret(_get(url, memory=memory))
    if resp is None:
        return []
    payload = resp.data or []
    return [pr["number"] for pr in payload]


def _get(
    url: str,
    *,
    retries: int = 3,
    memory: Any = None,
) -> GitHubResponse:
    """Compatibility shim delegating to the shared HTTP helper.

    Kept as a private function so any historical callers or monkey-patches
    in tests continue to work. New code should call
    :func:`git_explainer.tools.github_http.github_get_json` directly.
    """
    return github_get_json(url, headers=_HEADERS, retries=retries, memory=memory)

"""Fetch GitHub pull-request data via the REST API."""

import time

import requests

from git_explainer.config import GITHUB_API_BASE, github_headers

_HEADERS = github_headers()


def _handle_response(resp: requests.Response) -> requests.Response | None:
    """Return the response on success, None on 404, raise on auth/rate errors."""
    if resp.status_code == 404:
        return None
    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"GitHub auth failed – check GITHUB_TOKEN ({resp.status_code})"
        )
    if resp.status_code == 429:
        raise RuntimeError("GitHub rate limit exceeded – retry later")
    resp.raise_for_status()
    return resp


def fetch_pr(owner: str, repo: str, pr_number: int) -> dict | None:
    """Fetch a single pull request. Returns None if not found."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
    resp = _handle_response(_get(url))
    if resp is None:
        return None
    data = resp.json()
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


def fetch_pr_comments(owner: str, repo: str, pr_number: int) -> list[dict]:
    """Fetch review comments on a PR (first page, up to 30)."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    resp = _handle_response(_get(url))
    if resp is None:
        return []
    return [
        {
            "user": c["user"]["login"],
            "body": c["body"],
            "path": c["path"],
            "created_at": c["created_at"],
        }
        for c in resp.json()
    ]


def find_prs_for_commit(owner: str, repo: str, sha: str) -> list[int]:
    """Find which PR(s) a commit belongs to. Returns list of PR numbers."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}/pulls"
    resp = _handle_response(_get(url))
    if resp is None:
        return []
    return [pr["number"] for pr in resp.json()]


def _get(url: str, *, retries: int = 3) -> requests.Response:
    """Make a GitHub GET request with lightweight backoff on rate limiting."""
    response: requests.Response | None = None
    for attempt in range(retries):
        response = requests.get(url, headers=_HEADERS, timeout=10)
        if response.status_code not in (403, 429) or attempt == retries - 1:
            return response
        time.sleep(2 ** attempt)
    assert response is not None
    return response

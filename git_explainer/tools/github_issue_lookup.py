from __future__ import annotations

import re
from typing import Any

import requests

from git_explainer.config import GITHUB_API_BASE, github_headers
from git_explainer.tools.github_http import GitHubResponse, github_get_json

_HEADERS: dict[str, str] = github_headers()

_ISSUE_RE = re.compile(
    r"(?:(?:fix(?:es|ed)?|clos(?:es|ed)?|resolv(?:es|ed)?|related\s+to)\s+)?#(\d+)",
    re.IGNORECASE,
)


def fetch_issue(
    owner: str,
    repo: str,
    issue_number: int,
    *,
    memory: Any = None,
) -> dict[str, str | int | list[str]] | None:
    """Fetch a single GitHub issue by number.

    Returns ``None`` when the number does not exist (404) **or when it
    refers to a pull request rather than a true issue**. GitHub treats
    PRs as a subtype of issues and returns them through this endpoint,
    but they are also surfaced separately via the PR lookup. Without
    this filter the same artifact appears twice in the agent's evidence
    (once as a PR, once as an "issue"), which is a real source of
    misleading explanations on any repo where a commit message
    references its own PR by number.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}"
    resp = _get(url, memory=memory)

    if resp.status_code == 404:
        return None
    if resp.status_code in (401, 403):
        raise requests.HTTPError(
            f"GitHub auth error ({resp.status_code}): check GITHUB_TOKEN",
        )
    if resp.status_code == 429:
        raise requests.HTTPError("GitHub rate limit exceeded (429)")
    if resp.status_code not in (200, 304):
        raise requests.HTTPError(
            f"GitHub request failed with status {resp.status_code}"
        )

    data = resp.data or {}
    # GitHub sets ``pull_request`` on the issue payload iff the number
    # actually refers to a PR. https://docs.github.com/en/rest/issues
    if data.get("pull_request"):
        return None
    return {
        "number": data["number"],
        "title": data["title"],
        "state": data["state"],
        "body": data["body"] or "",
        "labels": [label["name"] for label in data["labels"]],
        "created_at": data["created_at"],
        "user": data["user"]["login"],
    }


def fetch_issue_comments(
    owner: str,
    repo: str,
    issue_number: int,
    *,
    memory: Any = None,
) -> list[dict[str, str]]:
    """Fetch issue comments (first page, up to 30)."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}/comments"
    resp = _get(url, memory=memory)
    if resp.status_code == 404:
        return []
    if resp.status_code not in (200, 304):
        raise requests.HTTPError(
            f"GitHub request failed with status {resp.status_code}"
        )
    payload = resp.data or []
    return [
        {
            "user": comment["user"]["login"],
            "body": comment["body"],
            "created_at": comment["created_at"],
        }
        for comment in payload
    ]


def extract_issue_refs(text: str) -> list[int]:
    """Extract deduplicated issue numbers from a commit message."""
    return list(dict.fromkeys(int(m) for m in _ISSUE_RE.findall(text)))


def fetch_issues(
    owner: str,
    repo: str,
    issue_numbers: list[int],
    *,
    memory: Any = None,
) -> list[dict[str, str | int | list[str]]]:
    """Fetch multiple issues, skipping any that don't exist."""
    results = []
    for num in issue_numbers:
        issue = fetch_issue(owner, repo, num, memory=memory)
        if issue is not None:
            results.append(issue)
    return results


def _get(
    url: str,
    *,
    retries: int = 3,
    memory: Any = None,
) -> GitHubResponse:
    """Compatibility shim delegating to the shared HTTP helper.

    New code should call
    :func:`git_explainer.tools.github_http.github_get_json` directly.
    """
    return github_get_json(url, headers=_HEADERS, retries=retries, memory=memory)

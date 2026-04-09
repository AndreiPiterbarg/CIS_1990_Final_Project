from __future__ import annotations

import re

import requests

from git_explainer.config import GITHUB_API_BASE, GITHUB_TOKEN

_HEADERS: dict[str, str] = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

_ISSUE_RE = re.compile(
    r"(?:(?:fix(?:es|ed)?|clos(?:es|ed)?|resolv(?:es|ed)?|related\s+to)\s+)?#(\d+)",
    re.IGNORECASE,
)


def fetch_issue(
    owner: str,
    repo: str,
    issue_number: int,
) -> dict[str, str | int | list[str]] | None:
    """Fetch a single GitHub issue by number. Returns None if not found."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}"
    resp = requests.get(url, headers=_HEADERS, timeout=10)

    if resp.status_code == 404:
        return None
    if resp.status_code in (401, 403):
        raise requests.HTTPError(
            f"GitHub auth error ({resp.status_code}): check GITHUB_TOKEN",
            response=resp,
        )
    if resp.status_code == 429:
        raise requests.HTTPError(
            "GitHub rate limit exceeded (429)", response=resp
        )
    resp.raise_for_status()

    data = resp.json()
    return {
        "number": data["number"],
        "title": data["title"],
        "state": data["state"],
        "body": data["body"] or "",
        "labels": [label["name"] for label in data["labels"]],
        "created_at": data["created_at"],
        "user": data["user"]["login"],
    }


def extract_issue_refs(text: str) -> list[int]:
    """Extract deduplicated issue numbers from a commit message."""
    return list(dict.fromkeys(int(m) for m in _ISSUE_RE.findall(text)))


def fetch_issues(
    owner: str,
    repo: str,
    issue_numbers: list[int],
) -> list[dict[str, str | int | list[str]]]:
    """Fetch multiple issues, skipping any that don't exist."""
    results = []
    for num in issue_numbers:
        issue = fetch_issue(owner, repo, num)
        if issue is not None:
            results.append(issue)
    return results

"""Small JSON-backed cache used by the Git explainer agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from git_explainer import config


def _default_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "commit_prs": {},
        "prs": {},
        "pr_comments": {},
        "issues": {},
        "issue_comments": {},
        "contexts": {},
    }


@dataclass(slots=True)
class ExplainerMemory:
    repo_path: str
    cache_path: str | None = None
    _payload: dict[str, Any] = field(init=False, repr=False)
    _dirty: bool = field(default=False, init=False, repr=False)
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)
    writes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        target = Path(self.cache_path) if self.cache_path else Path(self.repo_path) / config.CACHE_FILENAME
        self.cache_path = str(target)
        self._payload = _default_payload()
        self._load()

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}

    def get_commit_prs(self, sha: str) -> list[int] | None:
        return self._get("commit_prs", sha)

    def set_commit_prs(self, sha: str, pr_numbers: list[int]) -> None:
        self._set("commit_prs", sha, pr_numbers)

    def get_pr(self, pr_number: int) -> dict | None:
        return self._get("prs", str(pr_number))

    def set_pr(self, pr_number: int, pr_data: dict) -> None:
        self._set("prs", str(pr_number), pr_data)

    def get_pr_comments(self, pr_number: int) -> list[dict] | None:
        return self._get("pr_comments", str(pr_number))

    def set_pr_comments(self, pr_number: int, comments: list[dict]) -> None:
        self._set("pr_comments", str(pr_number), comments)

    def get_issue(self, issue_number: int) -> dict | None:
        return self._get("issues", str(issue_number))

    def set_issue(self, issue_number: int, issue_data: dict) -> None:
        self._set("issues", str(issue_number), issue_data)

    def get_issue_comments(self, issue_number: int) -> list[dict] | None:
        return self._get("issue_comments", str(issue_number))

    def set_issue_comments(self, issue_number: int, comments: list[dict]) -> None:
        self._set("issue_comments", str(issue_number), comments)

    def get_context(self, key: str) -> str | None:
        return self._get("contexts", key)

    def set_context(self, key: str, context: str) -> None:
        self._set("contexts", key, context)

    def flush(self) -> None:
        if not self._dirty:
            return
        path = Path(self.cache_path or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._payload, indent=2, sort_keys=True), encoding="utf-8")
        self.writes += 1
        self._dirty = False

    def _load(self) -> None:
        path = Path(self.cache_path or "")
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            merged = _default_payload()
            merged.update(data)
            self._payload = merged

    def _get(self, section: str, key: str):
        bucket = self._payload.setdefault(section, {})
        if key in bucket:
            self.hits += 1
            return bucket[key]
        self.misses += 1
        return None

    def _set(self, section: str, key: str, value: Any) -> None:
        bucket = self._payload.setdefault(section, {})
        if bucket.get(key) == value:
            return
        bucket[key] = value
        self._dirty = True

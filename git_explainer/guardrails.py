"""Input validation and heuristics for the Git explainer agent."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from git_explainer import config
from git_explainer.tools.file_context_reader import read_file_at_revision
from git_explainer.tools.git_utils import run_git

_GENERIC_MESSAGE_RE = re.compile(
    r"\b(fix|update|cleanup|refactor|format|lint|tweak|changes?|misc|wip)\b",
    re.IGNORECASE,
)
_GITHUB_REMOTE_RE = re.compile(
    r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"
)


@dataclass(slots=True)
class ExplainerQuery:
    repo_path: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    question: str | None = None
    owner: str | None = None
    repo_name: str | None = None
    max_commits: int = config.DEFAULT_MAX_COMMITS
    context_radius: int = config.DEFAULT_CONTEXT_RADIUS
    enforce_public_repo: bool = False

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


def validate_query(query: ExplainerQuery) -> ExplainerQuery:
    """Validate and normalize a query before orchestration begins."""
    repo = Path(query.repo_path).expanduser().resolve()
    if not repo.is_dir() or not (repo / ".git").exists():
        raise ValueError(f"Not a git repository: {repo}")

    question = query.question.strip() if query.question is not None else None
    question = question or None

    file_path: str | None = None
    if query.file_path is not None:
        file_path = normalize_file_path(str(repo), query.file_path)

    if question is not None:
        if query.start_line is not None or query.end_line is not None:
            raise ValueError("question mode does not accept start_line or end_line")
        if file_path is not None:
            _read_text_file(str(repo), file_path)
        start_line = None
        end_line = None
    else:
        if file_path is None or query.start_line is None or query.end_line is None:
            raise ValueError(
                "Provide file_path, start_line, and end_line, or use question mode"
            )
        if query.start_line <= 0 or query.end_line <= 0:
            raise ValueError("Line numbers must be positive integers")
        if query.end_line < query.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")

        span = query.end_line - query.start_line + 1
        if span > config.DEFAULT_MAX_LINE_SPAN:
            raise ValueError(
                f"Requested line range spans {span} lines; maximum is "
                f"{config.DEFAULT_MAX_LINE_SPAN}"
            )

        line_count = len(_read_text_file(str(repo), file_path).splitlines())
        if query.end_line > line_count:
            raise ValueError(
                f"Requested end_line {query.end_line} exceeds file length {line_count}"
            )
        start_line = query.start_line
        end_line = query.end_line

    owner = query.owner
    repo_name = query.repo_name
    if owner is None or repo_name is None:
        inferred = infer_github_repo(str(repo))
        if inferred is not None:
            owner = owner or inferred[0]
            repo_name = repo_name or inferred[1]

    normalized = ExplainerQuery(
        repo_path=str(repo),
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        question=question,
        owner=owner,
        repo_name=repo_name,
        max_commits=max(1, min(query.max_commits, 20)),
        context_radius=max(0, min(query.context_radius, 200)),
        enforce_public_repo=query.enforce_public_repo,
    )

    if normalized.enforce_public_repo:
        if normalized.owner is None or normalized.repo_name is None:
            raise ValueError(
                "owner and repo_name are required when enforce_public_repo is enabled"
            )
        ensure_public_github_repo(normalized.owner, normalized.repo_name)

    return normalized


def _read_text_file(repo_path: str, file_path: str) -> str:
    file_text = read_file_at_revision(repo_path, file_path, revision="HEAD")
    if file_text is None:
        file_text = read_file_at_revision(repo_path, file_path)
    if file_text is None:
        raise ValueError(f"File not found in repository: {file_path}")
    if file_text == "[binary file]":
        raise ValueError(f"Binary files are not supported: {file_path}")
    return file_text


def normalize_file_path(repo_path: str, file_path: str) -> str:
    """Return a repository-relative, POSIX-style file path."""
    repo = Path(repo_path).resolve()
    candidate = Path(file_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            rel = resolved.relative_to(repo)
        except ValueError as e:
            raise ValueError(f"File path must be inside the repository: {file_path}") from e
    else:
        rel = candidate
    normalized = rel.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized.startswith("../"):
        raise ValueError(f"Invalid file path: {file_path}")
    return normalized


def infer_github_repo(repo_path: str) -> tuple[str, str] | None:
    """Infer owner/repo from the local git remote when possible."""
    try:
        remote = run_git(repo_path, ["remote", "get-url", "origin"]).strip()
    except ValueError:
        return None

    match = _GITHUB_REMOTE_RE.search(remote)
    if not match:
        return None
    return match.group("owner"), match.group("repo")


def ensure_public_github_repo(owner: str, repo_name: str) -> dict:
    """Raise when a repo is missing or private."""
    url = f"{config.GITHUB_API_BASE}/repos/{owner}/{repo_name}"
    response = requests.get(url, headers=config.github_headers(), timeout=10)

    if response.status_code == 404:
        raise ValueError(f"GitHub repository is private or does not exist: {owner}/{repo_name}")
    response.raise_for_status()

    payload = response.json()
    if payload.get("private", False):
        raise ValueError(f"Private repositories are not supported: {owner}/{repo_name}")
    return payload


def should_fetch_file_context(commit_message: str, pr_body: str = "") -> bool:
    """Use file context for terse or ambiguous changes."""
    message = commit_message.strip()
    if len(message) < 24:
        return True
    if _GENERIC_MESSAGE_RE.search(message):
        return True
    if pr_body and len(pr_body.strip()) >= 40:
        return False
    return "#" not in message and ":" not in message

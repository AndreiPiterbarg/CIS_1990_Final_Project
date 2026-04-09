"""Read file contents from a local Git repository at a specific revision."""

import subprocess
from pathlib import Path


def read_file_at_revision(
    repo_path: str | Path,
    file_path: str,
    *,
    revision: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str | None:
    """Return file contents as a string, '[binary file]' for binaries, or None if missing.

    Args:
        repo_path: Path to the local git repository.
        file_path: File path relative to the repo root.
        revision: Git revision (SHA, branch, tag, HEAD~N). None reads from working tree.
        start_line: 1-indexed first line to include (None = from start).
        end_line: 1-indexed last line to include (None = to end).
    """
    repo = Path(repo_path)
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"Not a git repository: {repo}")

    if revision is None:
        content = _read_from_worktree(repo, file_path)
    else:
        content = _read_from_revision(repo, file_path, revision)

    if content is None or content == "[binary file]":
        return content

    if start_line is not None:
        lines = content.splitlines(keepends=True)
        start = max(start_line - 1, 0)
        lines = lines[start:end_line]
        content = "".join(lines)

    return content


def _read_from_worktree(repo: Path, file_path: str) -> str | None:
    target = repo / file_path
    if not target.exists():
        return None
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "[binary file]"


def _read_from_revision(repo: Path, file_path: str, revision: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{file_path}"],
        capture_output=True,
        cwd=repo,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        if "does not exist" in stderr or "exists on disk, but not in" in stderr:
            return None
        raise ValueError(f"git show failed: {stderr.strip()}")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return "[binary file]"

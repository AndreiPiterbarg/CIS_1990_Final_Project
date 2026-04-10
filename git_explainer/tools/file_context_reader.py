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
    """Return file contents as a string, ``"[binary file]"`` for binaries, or
    ``None`` if the file does not exist.

    Args:
        repo_path: Path to the local git repository.
        file_path: File path relative to the repo root.
        revision: Git revision (SHA, branch, tag, HEAD~N).  *None* reads from
            the working tree.
        start_line: 1-indexed first line to include (*None* = from start).
        end_line: 1-indexed last line to include, inclusive (*None* = to end).
    """
    repo = Path(repo_path)
    if not repo.is_dir() or not (repo / ".git").exists():
        raise ValueError(f"Not a git repository: {repo}")

    if revision is None:
        content = _read_from_worktree(repo, file_path)
    else:
        content = _read_from_revision(repo, file_path, revision)

    if content is None or content == "[binary file]":
        return content

    if start_line is not None or end_line is not None:
        lines = content.splitlines(keepends=True)
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        content = "".join(lines[s:e])

    return content


# --- Internal helpers --------------------------------------------------------

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
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return "[binary file]"

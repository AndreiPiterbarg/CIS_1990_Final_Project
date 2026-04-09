"""Shared git subprocess helpers."""

import subprocess
from pathlib import Path


def run_git(repo_path: str, args: list[str]) -> str:
    """Execute a git command in *repo_path* and return its stdout.

    Raises ValueError if the repo path does not exist or the command fails.
    """
    repo = Path(repo_path)
    if not repo.is_dir():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    try:
        result = subprocess.run(
            ["git"] + args, cwd=repo_path, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(e.stderr.strip() or str(e)) from e
    return result.stdout

"""Trace line-level history using git blame, log, and show."""

from datetime import datetime, timezone

from git_explainer.tools.git_utils import run_git


def get_blame(
    repo_path: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> list[dict[str, str | int]]:
    """Return structured blame data for a file, optionally limited to a line range."""
    cmd = ["blame", "--porcelain"]
    if start_line is not None:
        end = end_line if end_line is not None else start_line
        cmd += [f"-L{start_line},{end}"]
    cmd.append(file_path)

    output = run_git(repo_path, cmd)
    if not output.strip():
        return []

    entries: list[dict[str, str | int]] = []
    sha = ""
    line_no = 0
    author = ""
    timestamp = 0

    for raw_line in output.splitlines():
        parts = raw_line.split()
        if len(parts) >= 3 and len(parts[0]) == 40:
            sha = parts[0][:7]
            line_no = int(parts[2])
        elif raw_line.startswith("author "):
            author = raw_line[7:]
        elif raw_line.startswith("author-time "):
            timestamp = int(raw_line[12:])
        elif raw_line.startswith("\t"):
            date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
            entries.append({
                "sha": sha,
                "author": author,
                "date": date_str,
                "line": line_no,
                "content": raw_line[1:],
            })

    return entries


def get_commit_log(
    repo_path: str, file_path: str, max_count: int = 10
) -> list[dict[str, str]]:
    """Return commit history for a file as a list of dicts."""
    output = run_git(repo_path, [
        "log", f"--format=%h|%an|%ad|%s", "--date=short", f"-n{max_count}", "--", file_path
    ])
    entries: list[dict[str, str]] = []
    for line in output.strip().splitlines():
        if not line:
            continue
        sha, author, date, message = line.split("|", 3)
        entries.append({"sha": sha, "author": author, "date": date, "message": message})
    return entries


def get_commit_detail(repo_path: str, sha: str) -> str:
    """Return the full diff and message for a single commit."""
    return run_git(repo_path, ["show", sha]).strip()

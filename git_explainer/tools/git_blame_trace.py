"""Trace line-level history using git blame, log, and show."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from git_explainer.tools.git_utils import run_git

_COMMIT_FORMAT = "%H|%an|%ad|%s"
_NULL_SHA = "0" * 40
_FOLLOW_HISTORY_LIMIT = 200


def get_blame(
    repo_path: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    *,
    ignore_revs_file: str | None = None,
) -> list[dict[str, str | int]]:
    """Return structured blame data for a file, optionally limited to a line range."""
    cmd = ["blame", "--porcelain", "-M"]
    if ignore_revs_file is not None:
        cmd += ["--ignore-revs-file", ignore_revs_file]
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
            sha = parts[0]
            line_no = int(parts[2])
        elif raw_line.startswith("author "):
            author = raw_line[7:]
        elif raw_line.startswith("author-time "):
            timestamp = int(raw_line[12:])
        elif raw_line.startswith("\t"):
            date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
            entries.append({
                "sha": sha[:7],
                "full_sha": sha,
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


def trace_line_history(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    *,
    max_count: int = 5,
) -> list[dict[str, str]]:
    """Trace the commit history for a line range using ``git log -L``.

    ``git log -L`` is the best line-range primitive we have, but it is not
    robust against every refactor scenario. To mitigate that, we supplement it
    with two signals when needed:

    1. ``git blame -M`` with ``.git-blame-ignore-revs`` when present, so
       formatting-only revisions can be skipped while attributing the current
       lines.
    2. ``git log --follow -M -- <file>`` so we can recover commit metadata
       across file renames when the line trace is incomplete or empty.
    """
    entries = _parse_commit_lines(_trace_line_history_log(
        repo_path,
        file_path,
        start_line,
        end_line,
        max_count=max_count,
    ))
    seen = {entry["full_sha"] for entry in entries}

    ignore_revs_file = find_blame_ignore_revs_file(repo_path)
    blame_shas: list[str] = []
    if ignore_revs_file is not None or not entries:
        blame_shas = _blame_lineage(
            repo_path,
            file_path,
            start_line,
            end_line,
            ignore_revs_file=ignore_revs_file,
        )

    missing_blame_shas = [sha for sha in blame_shas if sha not in seen]
    if not entries or missing_blame_shas:
        follow_history = _parse_commit_lines(_follow_file_history(
            repo_path,
            file_path,
            max_count=min(_FOLLOW_HISTORY_LIMIT, max(max_count * 10, max_count)),
        ))

        if missing_blame_shas:
            missing_lookup = set(missing_blame_shas)
            for entry in follow_history:
                if entry["full_sha"] in missing_lookup and entry["full_sha"] not in seen:
                    entries.append(entry)
                    seen.add(entry["full_sha"])

            for sha in missing_blame_shas:
                if sha in seen:
                    continue
                commit = _read_commit_metadata(repo_path, sha)
                if commit is None:
                    continue
                entries.append(commit)
                seen.add(commit["full_sha"])

        if not entries:
            for entry in follow_history:
                if entry["full_sha"] in seen:
                    continue
                entries.append(entry)
                seen.add(entry["full_sha"])
                if len(entries) >= max_count:
                    break

    return entries[:max_count]


def find_blame_ignore_revs_file(repo_path: str) -> str | None:
    """Return the repo-local ``.git-blame-ignore-revs`` path when present."""
    candidate = Path(repo_path) / ".git-blame-ignore-revs"
    if candidate.is_file():
        return str(candidate)
    return None


def _trace_line_history_log(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    *,
    max_count: int,
) -> str:
    return run_git(repo_path, [
        "log",
        f"-L{start_line},{end_line}:{file_path}",
        f"--format={_COMMIT_FORMAT}",
        "--date=short",
        "--no-patch",
        f"-n{max_count}",
    ])


def _follow_file_history(repo_path: str, file_path: str, *, max_count: int) -> str:
    return run_git(repo_path, [
        "log",
        "--follow",
        "-M",
        f"--format={_COMMIT_FORMAT}",
        "--date=short",
        f"-n{max_count}",
        "--",
        file_path,
    ])


def _read_commit_metadata(repo_path: str, sha: str) -> dict[str, str] | None:
    try:
        output = run_git(repo_path, [
            "show",
            f"--format={_COMMIT_FORMAT}",
            "--date=short",
            "--no-patch",
            sha,
        ])
    except ValueError:
        return None

    entries = _parse_commit_lines(output)
    return entries[0] if entries else None


def _blame_lineage(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    *,
    ignore_revs_file: str | None,
) -> list[str]:
    try:
        blame_entries = get_blame(
            repo_path,
            file_path,
            start_line,
            end_line,
            ignore_revs_file=ignore_revs_file,
        )
    except ValueError:
        return []

    lineage: list[str] = []
    seen: set[str] = set()
    for entry in blame_entries:
        full_sha = str(entry.get("full_sha", ""))
        if not full_sha or full_sha == _NULL_SHA or full_sha in seen:
            continue
        seen.add(full_sha)
        lineage.append(full_sha)
    return lineage


def _parse_commit_lines(output: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in output.strip().splitlines():
        if not line or "|" not in line:
            continue
        sha, author, date, message = line.split("|", 3)
        if sha in seen:
            continue
        seen.add(sha)
        entries.append({
            "sha": sha[:7],
            "full_sha": sha,
            "author": author,
            "date": date,
            "message": message,
        })
    return entries

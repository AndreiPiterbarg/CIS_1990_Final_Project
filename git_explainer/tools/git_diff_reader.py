"""Structured diff parsing for git commits and revision ranges."""

from __future__ import annotations

import re
from typing import TypedDict

from git_explainer.tools.git_utils import run_git


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class HunkLine(TypedDict):
    type: str              # "add" | "delete" | "context"
    content: str           # Line text without the leading +/-/space
    old_line: int | None   # Line number in old file (None for additions)
    new_line: int | None   # Line number in new file (None for deletions)


class Hunk(TypedDict):
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str            # The full @@ line including function context
    lines: list[HunkLine]


class FileDiff(TypedDict):
    old_path: str          # "/dev/null" for added files
    new_path: str          # "/dev/null" for deleted files
    status: str            # "added" | "deleted" | "modified" | "renamed"
    additions: int
    deletions: int
    is_binary: bool
    hunks: list[Hunk]


class DiffSummary(TypedDict):
    files: list[FileDiff]
    total_additions: int
    total_deletions: int
    total_files_changed: int


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")
_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)
_BINARY_RE = re.compile(r"^Binary files .* differ$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_diff(
    repo_path: str,
    revision: str,
    *,
    base_revision: str | None = None,
    file_path: str | None = None,
    context_lines: int = 3,
) -> DiffSummary:
    """Return structured diff data for a commit or between two revisions.

    Args:
        repo_path: Absolute path to the local git repository.
        revision: The target revision (SHA, branch, tag).  For a single
            commit's diff, pass the SHA here and leave *base_revision* as
            ``None``.
        base_revision: If provided, diff from *base_revision* to *revision*.
            If ``None``, diffs the commit against its first parent.
        file_path: Limit the diff to this single file.
        context_lines: Number of context lines around each change (default 3).
    """
    raw = _raw_diff(repo_path, revision, base_revision, file_path, context_lines)
    return _parse_diff(raw)


def get_diff_stats(
    repo_path: str,
    revision: str,
    *,
    base_revision: str | None = None,
) -> list[dict[str, str | int]]:
    """Return lightweight file-level stats (file, additions, deletions).

    Uses ``git diff --numstat`` or ``git diff-tree --numstat``.
    """
    cmd = _numstat_cmd(revision, base_revision)
    output = run_git(repo_path, cmd)
    results: list[dict[str, str | int]] = []
    for line in output.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        add_str, del_str, fpath = parts
        results.append({
            "file": fpath,
            "additions": 0 if add_str == "-" else int(add_str),
            "deletions": 0 if del_str == "-" else int(del_str),
        })
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raw_diff(
    repo_path: str,
    revision: str,
    base_revision: str | None,
    file_path: str | None,
    context_lines: int,
) -> str:
    """Run the appropriate git diff command and return raw output."""
    if base_revision is not None:
        cmd = ["diff", f"-U{context_lines}", "--no-color",
               f"{base_revision}..{revision}"]
    else:
        # diff-tree -p --root handles root commits natively
        cmd = ["diff-tree", "-p", "--root", f"-U{context_lines}",
               "--no-color", revision]

    if file_path is not None:
        cmd += ["--", file_path]

    return run_git(repo_path, cmd)


def _numstat_cmd(
    revision: str,
    base_revision: str | None,
) -> list[str]:
    if base_revision is not None:
        return ["diff", "--numstat", f"{base_revision}..{revision}"]
    return ["diff-tree", "--numstat", "--root", revision]


def _parse_diff(raw: str) -> DiffSummary:
    """Parse unified diff output into a DiffSummary."""
    file_sections = _split_file_sections(raw)
    files: list[FileDiff] = []
    total_add = 0
    total_del = 0

    for section in file_sections:
        fd = _parse_file_section(section)
        files.append(fd)
        total_add += fd["additions"]
        total_del += fd["deletions"]

    return DiffSummary(
        files=files,
        total_additions=total_add,
        total_deletions=total_del,
        total_files_changed=len(files),
    )


def _split_file_sections(raw: str) -> list[str]:
    """Split raw diff output into per-file sections."""
    sections: list[str] = []
    current: list[str] = []

    for line in raw.splitlines(keepends=True):
        if _DIFF_HEADER_RE.match(line.rstrip("\n")):
            if current:
                sections.append("".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("".join(current))

    return sections


def _parse_file_section(section: str) -> FileDiff:
    """Parse a single file's diff section."""
    lines = section.splitlines()
    if not lines:
        return _empty_file_diff()

    # Extract paths from diff --git header
    m = _DIFF_HEADER_RE.match(lines[0])
    old_path = m.group(1) if m else ""
    new_path = m.group(2) if m else ""

    # Determine status and check for binary
    status = "modified"
    is_binary = False

    for line in lines[1:]:
        if line.startswith("new file mode"):
            status = "added"
            old_path = "/dev/null"
        elif line.startswith("deleted file mode"):
            status = "deleted"
            new_path = "/dev/null"
        elif line.startswith("rename from"):
            status = "renamed"
        elif _BINARY_RE.match(line):
            is_binary = True

    # Parse hunks
    hunks: list[Hunk] = []
    if not is_binary:
        hunks = _parse_hunks(lines)

    additions = sum(
        1 for h in hunks for hl in h["lines"] if hl["type"] == "add"
    )
    deletions = sum(
        1 for h in hunks for hl in h["lines"] if hl["type"] == "delete"
    )

    return FileDiff(
        old_path=old_path,
        new_path=new_path,
        status=status,
        additions=additions,
        deletions=deletions,
        is_binary=is_binary,
        hunks=hunks,
    )


def _parse_hunks(lines: list[str]) -> list[Hunk]:
    """Extract hunks from the lines of a single file diff section."""
    hunks: list[Hunk] = []
    current_hunk: Hunk | None = None
    old_line = 0
    new_line = 0

    for line in lines:
        hm = _HUNK_RE.match(line)
        if hm:
            current_hunk = Hunk(
                old_start=int(hm.group(1)),
                old_count=int(hm.group(2)) if hm.group(2) else 1,
                new_start=int(hm.group(3)),
                new_count=int(hm.group(4)) if hm.group(4) else 1,
                header=line,
                lines=[],
            )
            hunks.append(current_hunk)
            old_line = current_hunk["old_start"]
            new_line = current_hunk["new_start"]
            continue

        if current_hunk is None:
            continue

        if line.startswith("+"):
            current_hunk["lines"].append(HunkLine(
                type="add",
                content=line[1:],
                old_line=None,
                new_line=new_line,
            ))
            new_line += 1
        elif line.startswith("-"):
            current_hunk["lines"].append(HunkLine(
                type="delete",
                content=line[1:],
                old_line=old_line,
                new_line=None,
            ))
            old_line += 1
        elif line.startswith(" "):
            current_hunk["lines"].append(HunkLine(
                type="context",
                content=line[1:],
                old_line=old_line,
                new_line=new_line,
            ))
            old_line += 1
            new_line += 1
        # Skip "\ No newline at end of file" and other non-diff lines

    return hunks


def _empty_file_diff() -> FileDiff:
    return FileDiff(
        old_path="",
        new_path="",
        status="modified",
        additions=0,
        deletions=0,
        is_binary=False,
        hunks=[],
    )

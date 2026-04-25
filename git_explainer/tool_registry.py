"""Tool registry exposed to the Planner LLM.

The Planner does not call git or GitHub directly. It picks a *tool name*
from this registry and supplies arguments that match the declared JSON
schema; the registry's dispatcher then runs the deterministic backend
function with the agent's repo path, owner/repo, and shared
:class:`ExplainerMemory` instance injected from a side-channel context.

This split is deliberate:

* The LLM only sees / produces JSON-shaped tool calls. It cannot smuggle
  paths, credentials, or repo identifiers into arguments because those
  are not part of any tool's schema.
* The deterministic backends are unchanged. They keep their existing
  redaction, caching, and error handling.
* Adding a new tool is one entry in :data:`TOOL_SPECS` plus one branch
  in :func:`dispatch_tool`. Nothing else moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from git_explainer.memory import ExplainerMemory
from git_explainer.tools.commit_search import search_commits
from git_explainer.tools.file_context_reader import read_file_at_revision
from git_explainer.tools.git_blame_trace import trace_line_history
from git_explainer.tools.git_diff_reader import get_diff
from git_explainer.tools.github_issue_lookup import (
    extract_issue_refs,
    fetch_issue,
    fetch_issue_comments,
)
from git_explainer.tools.github_pr_lookup import (
    fetch_pr,
    fetch_pr_comments,
    find_prs_for_commit,
)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCallContext:
    """Side-channel state injected into every tool dispatch.

    The Planner LLM never sees these fields. They are supplied by the
    orchestrator and used by :func:`dispatch_tool` to bind tool calls to
    a concrete repository, GitHub repo, and cache.
    """

    repo_path: str
    owner: str | None
    repo_name: str | None
    memory: ExplainerMemory


@dataclass(slots=True)
class ToolCallRecord:
    """Audit record of a single tool invocation made by the Planner."""

    iteration: int
    tool: str
    arguments: dict[str, Any]
    # ``ok`` means the tool returned a value (which may itself be empty
    # -- e.g. an empty PR list). ``error`` means the dispatcher refused
    # the call (unknown tool, missing required arg, missing GitHub
    # owner/repo) or the backend raised an exception we caught.
    status: str = "ok"
    error: str | None = None
    # ``result_summary`` is a small human-readable string describing the
    # result shape -- e.g. ``"3 PR(s)"`` or ``"<not found>"``. The full
    # result is merged into the running evidence dict by the orchestrator
    # and is not duplicated here, both for log-size reasons and so the
    # audit trail stays JSON-serializable for the eval harness.
    result_summary: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "tool": self.tool,
            "arguments": self.arguments,
            "status": self.status,
            "error": self.error,
            "result_summary": self.result_summary,
            "reasoning": self.reasoning,
        }


class ToolDispatchError(Exception):
    """Raised when a tool call cannot be executed.

    Distinct from arbitrary backend exceptions: this signals the
    Planner-level contract was violated (unknown tool name, missing
    required argument, GitHub call without owner/repo configured) so the
    Planner can be told to pick a different action.
    """


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

# JSON schemas use the OpenAI tool-use draft (compatible with Groq). We
# keep them strict (``additionalProperties: False``) so the Planner does
# not get away with hallucinating fields. Each tool's ``description`` is
# *the* signal the LLM uses to decide when to call it -- be explicit
# about cost (network vs. local) and what the return shape looks like.

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "trace_line_history",
        "description": (
            "LOCAL git operation. Trace the commit history of a specific "
            "line range in a file using `git blame` + `git log -L`. Returns "
            "a list of commits (sha, full_sha, author, date, message) "
            "ordered most-recent first. Use this as the FIRST step when "
            "the question is scoped to a specific file/line range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Repository-relative POSIX path.",
                },
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "max_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["file_path", "start_line", "end_line"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_commits",
        "description": (
            "LOCAL git operation. Search commits by message, author, date "
            "range, and/or path. At least one filter must be provided. "
            "Use this when the question is broad (e.g., 'why did we add "
            "feature X' without a known line range), or as a fallback "
            "when trace_line_history returned no results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "grep": {"type": "string"},
                "author": {"type": "string"},
                "since": {"type": "string"},
                "until": {"type": "string"},
                "path": {"type": "string"},
                "max_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "find_prs_for_commit",
        "description": (
            "GITHUB API call. Find PR numbers that contain the given "
            "commit. Returns a list of integers (possibly empty). Cheap "
            "(one HTTP GET) but counts against your rate limit. Always "
            "do this before fetch_pr if you do not already know the PR "
            "number from a commit message."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commit_sha": {
                    "type": "string",
                    "description": "Full or short SHA of the commit.",
                },
            },
            "required": ["commit_sha"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_pr",
        "description": (
            "GITHUB API call. Fetch a single PR's title, body, state, "
            "branches, and merge commit. Use this once per relevant PR. "
            "Returns null if the PR does not exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pr_number": {"type": "integer", "minimum": 1},
            },
            "required": ["pr_number"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_pr_comments",
        "description": (
            "GITHUB API call. Fetch review comments on a PR (first page). "
            "Useful when the PR body alone does not document the design "
            "intent. Skip if you only need a one-line PR title."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pr_number": {"type": "integer", "minimum": 1},
            },
            "required": ["pr_number"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_issue",
        "description": (
            "GITHUB API call. Fetch a GitHub issue by number. Returns "
            "null if the number is actually a PR (PRs and issues share a "
            "namespace on GitHub) or does not exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "issue_number": {"type": "integer", "minimum": 1},
            },
            "required": ["issue_number"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_issue_comments",
        "description": (
            "GITHUB API call. Fetch issue comments (first page). Useful "
            "for issues whose body is short or only links to the PR."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "issue_number": {"type": "integer", "minimum": 1},
            },
            "required": ["issue_number"],
            "additionalProperties": False,
        },
    },
    {
        "name": "extract_issue_refs",
        "description": (
            "LOCAL string operation. Pull GitHub issue references "
            "(`#123`, `fixes #45`) out of a piece of free-form text "
            "(commit message or PR body). Cheap; use to discover linked "
            "issues before calling fetch_issue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_diff",
        "description": (
            "LOCAL git operation. Return the structured diff for a "
            "commit, optionally limited to one file. Sensitive literals "
            "are auto-redacted. Use this when you need to know WHAT "
            "actually changed (not just the message)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commit_sha": {"type": "string"},
                "file_path": {
                    "type": "string",
                    "description": "Optional. If omitted, returns the full diff.",
                },
            },
            "required": ["commit_sha"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file_at_revision",
        "description": (
            "LOCAL git operation. Read file contents at a specific "
            "revision, optionally limited to a line range. Use this when "
            "a commit message is generic ('fix', 'cleanup') and you need "
            "the surrounding code to reason about intent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "revision": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["file_path", "revision"],
            "additionalProperties": False,
        },
    },
]


_TOOL_SPECS_BY_NAME: dict[str, dict[str, Any]] = {
    spec["name"]: spec for spec in TOOL_SPECS
}


def get_tool_spec(name: str) -> dict[str, Any] | None:
    return _TOOL_SPECS_BY_NAME.get(name)


def tool_names() -> list[str]:
    return [spec["name"] for spec in TOOL_SPECS]


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _validate_arguments(spec: dict[str, Any], arguments: dict[str, Any]) -> None:
    """Lightweight schema validation.

    We do not pull in jsonschema for one feature -- the schemas here are
    flat and the only constraints we care about are: required keys are
    present, no unknown keys, and values are of the declared primitive
    type. Anything more exotic (enum constraints, min/max checking) is
    enforced by the backend functions themselves; the Planner does not
    benefit from a second layer.
    """
    schema = spec.get("input_schema", {})
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    allow_additional = schema.get("additionalProperties", True)

    missing = required - set(arguments.keys())
    if missing:
        raise ToolDispatchError(
            f"Tool {spec['name']!r} missing required arguments: "
            f"{sorted(missing)}"
        )

    if allow_additional is False:
        unknown = set(arguments.keys()) - set(properties.keys())
        if unknown:
            raise ToolDispatchError(
                f"Tool {spec['name']!r} got unknown argument(s): "
                f"{sorted(unknown)}"
            )

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for key, value in arguments.items():
        prop = properties.get(key)
        if not prop:
            continue
        expected = prop.get("type")
        if expected is None:
            continue
        py_type = type_map.get(expected)
        if py_type is None:
            continue
        # Reject ``True``/``False`` for integer fields. ``bool`` is a
        # subclass of ``int`` in Python and would otherwise sneak past
        # an isinstance(value, int) check.
        if expected == "integer" and isinstance(value, bool):
            raise ToolDispatchError(
                f"Tool {spec['name']!r} argument {key!r} must be an integer"
            )
        if not isinstance(value, py_type):
            raise ToolDispatchError(
                f"Tool {spec['name']!r} argument {key!r} must be of type "
                f"{expected!r}; got {type(value).__name__}"
            )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _summarize_result(name: str, result: Any) -> str:
    """One-line description of what a tool returned, for the audit log."""
    if result is None:
        return "<not found>"
    if name == "trace_line_history" or name == "search_commits":
        return f"{len(result)} commit(s)"
    if name == "find_prs_for_commit":
        return f"{len(result)} PR number(s)"
    if name == "fetch_pr":
        return f"PR #{result.get('number', '?')} ({result.get('state', '?')})"
    if name == "fetch_pr_comments":
        return f"{len(result)} review comment(s)"
    if name == "fetch_issue":
        return f"issue #{result.get('number', '?')} ({result.get('state', '?')})"
    if name == "fetch_issue_comments":
        return f"{len(result)} issue comment(s)"
    if name == "extract_issue_refs":
        return f"{len(result)} ref(s)"
    if name == "get_diff":
        files = result.get("files", []) if isinstance(result, dict) else []
        return f"diff over {len(files)} file(s)"
    if name == "read_file_at_revision":
        if isinstance(result, str):
            return f"{len(result)} char(s) of context"
        return "<no content>"
    return ""


def _require_github(context: ToolCallContext, tool_name: str) -> None:
    if not context.owner or not context.repo_name:
        raise ToolDispatchError(
            f"Tool {tool_name!r} requires GitHub owner/repo, but neither "
            "was inferred from the local remote nor provided explicitly."
        )


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    context: ToolCallContext,
) -> Any:
    """Run the deterministic backend behind a Planner-issued tool call.

    On success returns the backend's return value. On Planner-contract
    violations raises :class:`ToolDispatchError`. Backend exceptions
    (``ValueError`` from bad git args, network errors, etc.) are NOT
    swallowed here -- the Planner driver catches them and records them
    as a failed iteration.
    """
    spec = _TOOL_SPECS_BY_NAME.get(name)
    if spec is None:
        raise ToolDispatchError(f"Unknown tool: {name!r}")

    if not isinstance(arguments, dict):
        raise ToolDispatchError(
            f"Tool {name!r} arguments must be an object; got {type(arguments).__name__}"
        )
    _validate_arguments(spec, arguments)

    memory = context.memory

    if name == "trace_line_history":
        return trace_line_history(
            context.repo_path,
            arguments["file_path"],
            int(arguments["start_line"]),
            int(arguments["end_line"]),
            max_count=int(arguments.get("max_count", 5)),
        )

    if name == "search_commits":
        # ``search_commits`` requires at least one filter. Surface that
        # as a ToolDispatchError so the Planner can self-correct rather
        # than letting a bare ValueError escape.
        if not any(
            arguments.get(k)
            for k in ("grep", "author", "since", "until", "path")
        ):
            raise ToolDispatchError(
                "search_commits requires at least one of: grep, author, "
                "since, until, path"
            )
        return search_commits(
            context.repo_path,
            grep=arguments.get("grep"),
            author=arguments.get("author"),
            since=arguments.get("since"),
            until=arguments.get("until"),
            path=arguments.get("path"),
            max_count=int(arguments.get("max_count", 20)),
        )

    if name == "find_prs_for_commit":
        _require_github(context, name)
        sha = arguments["commit_sha"]
        cached = memory.get_commit_prs(sha)
        if cached is not None:
            return cached
        result = find_prs_for_commit(
            context.owner, context.repo_name, sha, memory=memory,
        )
        memory.set_commit_prs(sha, result)
        return result

    if name == "fetch_pr":
        _require_github(context, name)
        pr_number = int(arguments["pr_number"])
        cached = memory.get_pr(pr_number)
        if cached is not None:
            return cached
        result = fetch_pr(
            context.owner, context.repo_name, pr_number, memory=memory,
        )
        if result is not None:
            memory.set_pr(pr_number, result)
        return result

    if name == "fetch_pr_comments":
        _require_github(context, name)
        pr_number = int(arguments["pr_number"])
        cached = memory.get_pr_comments(pr_number)
        if cached is not None:
            return cached
        result = fetch_pr_comments(
            context.owner, context.repo_name, pr_number, memory=memory,
        )
        memory.set_pr_comments(pr_number, result)
        return result

    if name == "fetch_issue":
        _require_github(context, name)
        issue_number = int(arguments["issue_number"])
        cached = memory.get_issue(issue_number)
        if cached is not None:
            return cached
        result = fetch_issue(
            context.owner, context.repo_name, issue_number, memory=memory,
        )
        if result is not None:
            memory.set_issue(issue_number, result)
        return result

    if name == "fetch_issue_comments":
        _require_github(context, name)
        issue_number = int(arguments["issue_number"])
        cached = memory.get_issue_comments(issue_number)
        if cached is not None:
            return cached
        result = fetch_issue_comments(
            context.owner, context.repo_name, issue_number, memory=memory,
        )
        memory.set_issue_comments(issue_number, result)
        return result

    if name == "extract_issue_refs":
        return extract_issue_refs(arguments["text"])

    if name == "get_diff":
        sha = arguments["commit_sha"]
        file_path = arguments.get("file_path")
        cache_key = f"{sha}:{file_path or '*'}"
        cached = memory.get_diff(cache_key)
        if cached is not None:
            return cached
        try:
            result = get_diff(
                context.repo_path,
                sha,
                file_path=file_path,
                context_lines=1,
            )
        except (ValueError, OSError) as exc:
            # Surface as ToolDispatchError so the Planner sees a clean
            # contract failure (typically a bad SHA the LLM hallucinated)
            # rather than crashing the whole loop.
            raise ToolDispatchError(f"get_diff failed: {exc}") from exc
        memory.set_diff(cache_key, result)
        return result

    if name == "read_file_at_revision":
        return read_file_at_revision(
            context.repo_path,
            arguments["file_path"],
            revision=arguments.get("revision"),
            start_line=arguments.get("start_line"),
            end_line=arguments.get("end_line"),
        )

    # Defensive default: every name in TOOL_SPECS must have a branch above.
    raise ToolDispatchError(f"Tool {name!r} has no dispatch implementation")


# ---------------------------------------------------------------------------
# Result -> Evidence merging
# ---------------------------------------------------------------------------


def _empty_evidence() -> dict[str, Any]:
    return {
        "commits": [],
        "pull_requests": [],
        "issues": [],
        "file_contexts": [],
        "diffs": [],
        # ``candidate_issue_refs`` and ``candidate_pr_numbers`` are
        # working-set lists the Planner can use across iterations to
        # remember what it discovered but has not yet fetched. They are
        # NOT included in the synthesis prompt.
        "candidate_issue_refs": [],
        "candidate_pr_numbers": [],
    }


def merge_tool_result(
    evidence: dict[str, Any],
    tool: str,
    arguments: dict[str, Any],
    result: Any,
) -> None:
    """Fold a tool's return value into the running evidence dict.

    The transformations here mirror what
    :meth:`GitExplainerAgent._collect_evidence` does in the fixed-sequence
    path, so the synthesis prompt sees the same shape regardless of
    which path filled the evidence.

    Mutates ``evidence`` in place.
    """
    if tool == "trace_line_history" or tool == "search_commits":
        seen = {c.get("full_sha", c.get("sha")) for c in evidence["commits"]}
        for commit in result or []:
            key = commit.get("full_sha", commit.get("sha"))
            if key in seen:
                continue
            evidence["commits"].append(commit)
            seen.add(key)
        return

    if tool == "find_prs_for_commit":
        bucket = evidence.setdefault("candidate_pr_numbers", [])
        for pr_number in result or []:
            if pr_number not in bucket:
                bucket.append(pr_number)
        return

    if tool == "fetch_pr":
        if result is None:
            return
        existing_numbers = {pr["number"] for pr in evidence["pull_requests"]}
        if result["number"] in existing_numbers:
            return
        record = dict(result)
        record.setdefault("review_comments", [])
        evidence["pull_requests"].append(record)
        return

    if tool == "fetch_pr_comments":
        pr_number = int(arguments.get("pr_number", -1))
        for pr in evidence["pull_requests"]:
            if pr["number"] == pr_number:
                pr["review_comments"] = list(result or [])
                return
        # If the planner asked for comments before fetching the PR
        # itself, stash them on a placeholder so they aren't lost.
        evidence.setdefault("orphan_pr_comments", {})[str(pr_number)] = list(
            result or []
        )
        return

    if tool == "fetch_issue":
        if result is None:
            return
        existing_numbers = {iss["number"] for iss in evidence["issues"]}
        if result["number"] in existing_numbers:
            return
        record = dict(result)
        record.setdefault("comments", [])
        evidence["issues"].append(record)
        return

    if tool == "fetch_issue_comments":
        issue_number = int(arguments.get("issue_number", -1))
        for issue in evidence["issues"]:
            if issue["number"] == issue_number:
                issue["comments"] = list(result or [])
                return
        evidence.setdefault("orphan_issue_comments", {})[str(issue_number)] = list(
            result or []
        )
        return

    if tool == "extract_issue_refs":
        bucket = evidence.setdefault("candidate_issue_refs", [])
        for ref in result or []:
            if ref not in bucket:
                bucket.append(ref)
        return

    if tool == "get_diff":
        # Mirror ``_compact_diff`` in orchestrator.py so synthesis sees
        # the same compact shape it expects from the fixed path.
        from git_explainer.orchestrator import _compact_diff

        sha = arguments.get("commit_sha", "")
        short_sha = sha[:7] if isinstance(sha, str) else ""
        compact = _compact_diff(result or {"files": []}, short_sha)
        if not compact.get("hunks"):
            return
        # Avoid duplicates if the planner asked twice.
        for d in evidence["diffs"]:
            if d.get("commit_sha") == compact.get("commit_sha"):
                return
        evidence["diffs"].append(compact)
        return

    if tool == "read_file_at_revision":
        if not result or not isinstance(result, str) or result == "[binary file]":
            return
        record = {
            "commit_sha": (arguments.get("revision") or "")[:7],
            "file_path": arguments.get("file_path", ""),
            "start_line": arguments.get("start_line"),
            "end_line": arguments.get("end_line"),
            "content": result,
        }
        evidence["file_contexts"].append(record)
        return


__all__ = [
    "TOOL_SPECS",
    "ToolCallContext",
    "ToolCallRecord",
    "ToolDispatchError",
    "dispatch_tool",
    "get_tool_spec",
    "tool_names",
    "merge_tool_result",
    "_empty_evidence",
    "_summarize_result",
]

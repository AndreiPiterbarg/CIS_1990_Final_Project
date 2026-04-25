"""Tests for the Planner-facing tool registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from git_explainer.tool_registry import (
    TOOL_SPECS,
    ToolCallContext,
    ToolDispatchError,
    _empty_evidence,
    dispatch_tool,
    get_tool_spec,
    merge_tool_result,
    tool_names,
)


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_every_tool_has_required_schema_fields():
    assert TOOL_SPECS, "registry should not be empty"
    for spec in TOOL_SPECS:
        assert "name" in spec
        assert "description" in spec
        assert "input_schema" in spec
        schema = spec["input_schema"]
        assert schema.get("type") == "object"
        assert "properties" in schema
        # Strict mode: planner cannot smuggle extra fields.
        assert schema.get("additionalProperties") is False, (
            f"{spec['name']} must have additionalProperties: False"
        )


def test_tool_names_are_unique():
    names = tool_names()
    assert len(names) == len(set(names))


def test_get_tool_spec_returns_none_for_unknown():
    assert get_tool_spec("does_not_exist") is None


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _make_context() -> ToolCallContext:
    return ToolCallContext(
        repo_path="/tmp/repo",
        owner="octocat",
        repo_name="hello",
        memory=MagicMock(),
    )


def test_unknown_tool_raises():
    with pytest.raises(ToolDispatchError, match="Unknown tool"):
        dispatch_tool("frobnicate", {}, _make_context())


def test_missing_required_argument_raises():
    with pytest.raises(ToolDispatchError, match="missing required arguments"):
        dispatch_tool("fetch_pr", {}, _make_context())


def test_unknown_argument_raises():
    with pytest.raises(ToolDispatchError, match="unknown argument"):
        dispatch_tool(
            "fetch_pr",
            {"pr_number": 42, "extra": "nope"},
            _make_context(),
        )


def test_wrong_argument_type_raises():
    with pytest.raises(ToolDispatchError, match="must be of type"):
        dispatch_tool("fetch_pr", {"pr_number": "forty-two"}, _make_context())


def test_boolean_rejected_for_integer_field():
    # ``bool`` is a subclass of ``int`` in Python; the validator must
    # reject it for fields declared as integer.
    with pytest.raises(ToolDispatchError, match="must be an integer"):
        dispatch_tool("fetch_pr", {"pr_number": True}, _make_context())


def test_search_commits_requires_at_least_one_filter():
    with pytest.raises(ToolDispatchError, match="at least one"):
        dispatch_tool("search_commits", {}, _make_context())


def test_github_tool_without_owner_raises():
    ctx = ToolCallContext(
        repo_path="/tmp/repo",
        owner=None,
        repo_name=None,
        memory=MagicMock(),
    )
    with pytest.raises(ToolDispatchError, match="requires GitHub owner"):
        dispatch_tool("fetch_pr", {"pr_number": 42}, ctx)


# ---------------------------------------------------------------------------
# Dispatch wires through to the right backend
# ---------------------------------------------------------------------------


def test_dispatch_uses_memory_cache_for_fetch_pr(monkeypatch):
    memory = MagicMock()
    memory.get_pr.return_value = {"number": 42, "title": "cached"}
    ctx = ToolCallContext(
        repo_path="/tmp/repo",
        owner="o",
        repo_name="r",
        memory=memory,
    )
    fake_fetch = MagicMock()
    monkeypatch.setattr(
        "git_explainer.tool_registry.fetch_pr", fake_fetch
    )
    result = dispatch_tool("fetch_pr", {"pr_number": 42}, ctx)
    assert result == {"number": 42, "title": "cached"}
    # No backend call when cache hit.
    fake_fetch.assert_not_called()


def test_dispatch_calls_backend_on_cache_miss_and_writes_back(monkeypatch):
    memory = MagicMock()
    memory.get_pr.return_value = None
    ctx = ToolCallContext(
        repo_path="/tmp/repo",
        owner="o",
        repo_name="r",
        memory=memory,
    )
    monkeypatch.setattr(
        "git_explainer.tool_registry.fetch_pr",
        MagicMock(return_value={"number": 42, "title": "fresh"}),
    )
    dispatch_tool("fetch_pr", {"pr_number": 42}, ctx)
    memory.set_pr.assert_called_once_with(42, {"number": 42, "title": "fresh"})


def test_dispatch_get_diff_wraps_backend_error():
    ctx = _make_context()
    ctx.memory.get_diff.return_value = None

    def boom(*args, **kwargs):
        raise ValueError("bad sha")

    import git_explainer.tool_registry as reg

    original = reg.get_diff
    reg.get_diff = boom
    try:
        with pytest.raises(ToolDispatchError, match="get_diff failed"):
            dispatch_tool(
                "get_diff",
                {"commit_sha": "abc1234", "file_path": "src/a.py"},
                ctx,
            )
    finally:
        reg.get_diff = original


# ---------------------------------------------------------------------------
# Evidence merging
# ---------------------------------------------------------------------------


def test_merge_trace_line_history_appends_unique_commits():
    evidence = _empty_evidence()
    merge_tool_result(
        evidence,
        "trace_line_history",
        {},
        [
            {"sha": "aaa1111", "full_sha": "aaa1111aaa", "message": "first"},
            {"sha": "bbb2222", "full_sha": "bbb2222bbb", "message": "second"},
        ],
    )
    # Re-merging the same data must not duplicate.
    merge_tool_result(
        evidence,
        "trace_line_history",
        {},
        [{"sha": "aaa1111", "full_sha": "aaa1111aaa", "message": "first"}],
    )
    assert len(evidence["commits"]) == 2
    assert evidence["commits"][0]["sha"] == "aaa1111"


def test_merge_fetch_pr_attaches_review_comments_default_empty():
    evidence = _empty_evidence()
    merge_tool_result(
        evidence,
        "fetch_pr",
        {"pr_number": 7},
        {"number": 7, "title": "x", "body": "", "state": "merged"},
    )
    assert evidence["pull_requests"][0]["review_comments"] == []


def test_merge_fetch_pr_comments_attaches_to_existing_pr():
    evidence = _empty_evidence()
    evidence["pull_requests"].append(
        {"number": 7, "title": "x", "body": "", "review_comments": []}
    )
    merge_tool_result(
        evidence,
        "fetch_pr_comments",
        {"pr_number": 7},
        [{"user": "u", "body": "b"}],
    )
    assert evidence["pull_requests"][0]["review_comments"] == [
        {"user": "u", "body": "b"}
    ]


def test_merge_fetch_pr_comments_orphan_when_pr_missing():
    evidence = _empty_evidence()
    merge_tool_result(
        evidence,
        "fetch_pr_comments",
        {"pr_number": 7},
        [{"user": "u", "body": "b"}],
    )
    # No PR 7 in evidence yet; comments preserved as orphans rather
    # than dropped silently.
    assert evidence["orphan_pr_comments"]["7"] == [{"user": "u", "body": "b"}]


def test_merge_fetch_pr_skips_none_result():
    evidence = _empty_evidence()
    merge_tool_result(evidence, "fetch_pr", {"pr_number": 99}, None)
    assert evidence["pull_requests"] == []


def test_merge_fetch_issue_dedups_by_number():
    evidence = _empty_evidence()
    issue = {"number": 5, "title": "t", "body": "b", "state": "open"}
    merge_tool_result(evidence, "fetch_issue", {"issue_number": 5}, issue)
    merge_tool_result(evidence, "fetch_issue", {"issue_number": 5}, issue)
    assert len(evidence["issues"]) == 1


def test_merge_extract_issue_refs_into_candidate_set():
    evidence = _empty_evidence()
    merge_tool_result(evidence, "extract_issue_refs", {"text": ""}, [42, 7])
    merge_tool_result(evidence, "extract_issue_refs", {"text": ""}, [7, 99])
    assert evidence["candidate_issue_refs"] == [42, 7, 99]


def test_merge_get_diff_compacts_and_skips_empty_hunks():
    evidence = _empty_evidence()
    raw_diff = {
        "files": [
            {
                "hunks": [
                    {
                        "header": "@@ -1,1 +1,2 @@",
                        "lines": [
                            {"type": "add", "content": "x", "old_line": None, "new_line": 1},
                        ],
                    }
                ],
            }
        ],
    }
    merge_tool_result(
        evidence,
        "get_diff",
        {"commit_sha": "abc1234", "file_path": "f.py"},
        raw_diff,
    )
    assert len(evidence["diffs"]) == 1
    assert evidence["diffs"][0]["commit_sha"] == "abc1234"

    # Empty hunks should not produce a diff entry.
    merge_tool_result(
        evidence,
        "get_diff",
        {"commit_sha": "deadbee"},
        {"files": []},
    )
    assert len(evidence["diffs"]) == 1


def test_merge_read_file_at_revision_drops_empty_and_binary():
    evidence = _empty_evidence()
    merge_tool_result(
        evidence,
        "read_file_at_revision",
        {"file_path": "f", "revision": "abc"},
        "[binary file]",
    )
    merge_tool_result(
        evidence,
        "read_file_at_revision",
        {"file_path": "f", "revision": "abc"},
        "",
    )
    assert evidence["file_contexts"] == []

    merge_tool_result(
        evidence,
        "read_file_at_revision",
        {"file_path": "f", "revision": "abc1234"},
        "real content",
    )
    assert len(evidence["file_contexts"]) == 1
    assert evidence["file_contexts"][0]["content"] == "real content"

"""End-to-end integration tests for the Planner + Critic orchestrator path.

These tests mock the LLM transports (Groq chat for the Planner +
synthesizer, Anthropic for the Critic) but exercise the full
:class:`GitExplainerAgent` flow. Goals:

1. Planner-driven evidence collection produces the same downstream
   shape as the fixed-sequence path.
2. The Critic's ``needs_more_evidence`` verdict actually triggers a
   re-plan + re-synthesis round.
3. Every failure mode falls back cleanly: missing planner LLM,
   missing critic LLM, planner producing no useful calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from git_explainer.guardrails import ExplainerQuery
from git_explainer.orchestrator import GitExplainerAgent


def _make_query(tmp_path: Path) -> ExplainerQuery:
    repo = tmp_path / "repo"
    repo.mkdir()
    return ExplainerQuery(
        repo_path=str(repo),
        file_path="src/app.py",
        start_line=10,
        end_line=12,
        owner="octocat",
        repo_name="hello",
        max_commits=5,
    )


def _good_synthesis_json() -> str:
    return json.dumps({
        "what_changed": "Parser changed [commit:abc1234].",
        "why": "See [pr:#42].",
        "tradeoffs": "Minor [commit:abc1234].",
        "limitations": "Limited to retrieved evidence [commit:abc1234].",
        "summary": "Summary [commit:abc1234].",
    })


# ---------------------------------------------------------------------------
# Planner path: planner runs, fetches PRs/diffs, synthesis succeeds
# ---------------------------------------------------------------------------


@patch("git_explainer.orchestrator.chat")
@patch("git_explainer.orchestrator.is_available", return_value=True)
@patch("git_explainer.tool_registry.get_diff")
@patch("git_explainer.tool_registry.fetch_pr_comments")
@patch("git_explainer.tool_registry.fetch_pr")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_planner_path_drives_evidence_collection(
    mock_validate,
    mock_trace,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_get_diff,
    mock_synth_available,
    mock_synth_chat,
    tmp_path,
):
    """Happy path: planner picks fetch_pr -> get_diff -> done; synthesis
    runs; critic disabled."""
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {
            "sha": "abc1234",
            "full_sha": "abc123456789",
            "author": "Alice",
            "date": "2024-06-01",
            "message": "Fix parser",
        },
    ]
    mock_fetch_pr.return_value = {
        "number": 42,
        "title": "Improve parser",
        "body": "fixes parser bug",
        "state": "merged",
    }
    mock_fetch_pr_comments.return_value = []
    mock_get_diff.return_value = {
        "files": [
            {
                "hunks": [
                    {
                        "header": "@@ -1,1 +1,2 @@",
                        "lines": [
                            {
                                "type": "add",
                                "content": "new line",
                                "old_line": None,
                                "new_line": 2,
                            }
                        ],
                    }
                ]
            }
        ],
    }

    planner_replies = iter(
        [
            json.dumps({
                "action": "call_tool",
                "tool": "fetch_pr",
                "arguments": {"pr_number": 42},
                "reasoning": "fetch the linked PR",
            }),
            json.dumps({
                "action": "call_tool",
                "tool": "get_diff",
                "arguments": {"commit_sha": "abc1234", "file_path": "src/app.py"},
                "reasoning": "see what changed",
            }),
            json.dumps({"action": "done", "reasoning": "have enough"}),
        ]
    )

    def fake_chat(prompt, **kwargs):
        # The planner runs first (returns JSON action). Once the
        # planner says done, the next call is the synthesizer.
        try:
            return next(planner_replies)
        except StopIteration:
            return _good_synthesis_json()

    mock_synth_chat.side_effect = fake_chat

    agent = GitExplainerAgent(use_llm=True, use_planner=True, use_critic=False)
    result = agent.explain(query)

    assert result["used_fallback"] is False
    assert result["planner"] is not None
    assert result["planner"]["fell_back_to_fixed_sequence"] is False
    # Three planner LLM round trips + 1 synthesizer round trip.
    assert mock_synth_chat.call_count == 4

    # PR + diff made it into the evidence the agent returned.
    assert any(pr["number"] == 42 for pr in result["pull_requests"])
    assert len(result["diffs"]) >= 1


# ---------------------------------------------------------------------------
# Planner falls back when the LLM is not available
# ---------------------------------------------------------------------------


@patch("git_explainer.orchestrator.chat")
@patch("git_explainer.orchestrator.is_available", return_value=False)
@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.read_file_at_revision")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_planner_falls_back_to_fixed_sequence_when_llm_unavailable(
    mock_validate,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_read_file,
    mock_get_diff,
    mock_synth_available,
    mock_synth_chat,
    tmp_path,
):
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {
            "sha": "abc1234",
            "full_sha": "abc123456789",
            "author": "Alice",
            "date": "2024-06-01",
            "message": "Fix parser",
        },
    ]
    mock_find_prs.return_value = [42]
    mock_fetch_pr.return_value = {
        "number": 42, "title": "x", "body": "", "state": "merged",
    }
    mock_fetch_pr_comments.return_value = []
    mock_read_file.return_value = "ctx"
    mock_get_diff.return_value = {"files": []}

    agent = GitExplainerAgent(use_llm=True, use_planner=True, use_critic=False)
    result = agent.explain(query)

    # Planner unavailable -> fixed sequence ran; PRs gathered the old way.
    assert result["planner"] is not None
    assert result["planner"]["fell_back_to_fixed_sequence"] is True
    assert any(pr["number"] == 42 for pr in result["pull_requests"])
    # Synthesis LLM also unavailable (mocked), so we used the fallback.
    assert result["used_fallback"] is True
    assert result["fallback_reason"] == "llm_disabled"


# ---------------------------------------------------------------------------
# Critic path: ok verdict accepts the draft
# ---------------------------------------------------------------------------


@patch("git_explainer.orchestrator.critic_mod.critique")
@patch("git_explainer.orchestrator.chat")
@patch("git_explainer.orchestrator.is_available", return_value=True)
@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.read_file_at_revision")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_critic_ok_verdict_keeps_draft_unchanged(
    mock_validate,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_read_file,
    mock_get_diff,
    mock_synth_available,
    mock_chat,
    mock_critique,
    tmp_path,
):
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "A", "date": "2024-06-01", "message": "Fix"},
    ]
    mock_find_prs.return_value = [42]
    mock_fetch_pr.return_value = {"number": 42, "title": "x", "body": "", "state": "merged"}
    mock_fetch_pr_comments.return_value = []
    mock_read_file.return_value = "ctx"
    mock_get_diff.return_value = {"files": []}

    mock_chat.return_value = _good_synthesis_json()

    from git_explainer.critic import CriticReport

    mock_critique.return_value = CriticReport(
        verdict="ok",
        issues=[],
        focus_hints=[],
        reasoning="ok",
        available=True,
        model="claude-haiku-4-5",
    )

    # Critic runs but planner is off, so this exercises the
    # critic-without-planner code path.
    agent = GitExplainerAgent(use_llm=True, use_planner=False, use_critic=True)
    result = agent.explain(query)

    assert result["critic"]["verdict"] == "ok"
    assert mock_chat.call_count == 1  # No re-synthesis.
    assert "[commit:abc1234]" in result["explanation"]["summary"]


# ---------------------------------------------------------------------------
# Critic path: needs_more_evidence triggers re-plan + re-synthesis
# ---------------------------------------------------------------------------


@patch("git_explainer.orchestrator.read_file_at_revision", return_value="ctx")
@patch("git_explainer.orchestrator.critic_mod.critique")
@patch("git_explainer.orchestrator.chat")
@patch("git_explainer.orchestrator.is_available", return_value=True)
@patch("git_explainer.tool_registry.fetch_issue_comments")
@patch("git_explainer.tool_registry.fetch_issue")
@patch("git_explainer.tool_registry.get_diff")
@patch("git_explainer.tool_registry.fetch_pr_comments")
@patch("git_explainer.tool_registry.fetch_pr")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_critic_needs_more_evidence_replans_and_resynthesizes(
    mock_validate,
    mock_trace,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_get_diff,
    mock_fetch_issue,
    mock_fetch_issue_comments,
    mock_synth_available,
    mock_chat,
    mock_critique,
    mock_read_file,
    tmp_path,
):
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "A", "date": "2024-06-01", "message": "Fix"},
    ]
    mock_fetch_pr.return_value = {"number": 42, "title": "x", "body": "fixes #99", "state": "merged"}
    mock_fetch_pr_comments.return_value = []
    mock_fetch_issue.return_value = {
        "number": 99,
        "title": "the bug",
        "body": "important context",
        "state": "open",
        "labels": [],
    }
    mock_fetch_issue_comments.return_value = []
    mock_get_diff.return_value = {"files": []}

    # First planner run: fetch the PR, then done.
    # Second planner run (after critic re-plan): fetch the issue, done.
    planner_replies = iter(
        [
            json.dumps({
                "action": "call_tool",
                "tool": "fetch_pr",
                "arguments": {"pr_number": 42},
                "reasoning": "first pass",
            }),
            json.dumps({"action": "done", "reasoning": "first pass done"}),
            json.dumps({
                "action": "call_tool",
                "tool": "fetch_issue",
                "arguments": {"issue_number": 99},
                "reasoning": "critic asked",
            }),
            json.dumps({"action": "done", "reasoning": "second pass done"}),
        ]
    )

    synth_replies = iter([_good_synthesis_json(), _good_synthesis_json()])

    def fake_chat(prompt, **kwargs):
        # Synthesizer prompts include the literal phrase "Explain why";
        # planner prompts do not. Route by sniffing for that.
        if "Explain why" in prompt:
            return next(synth_replies)
        return next(planner_replies)

    mock_chat.side_effect = fake_chat

    from git_explainer.critic import CriticReport

    # First call: critic flags missing issue. Second call (after
    # re-synthesis) we don't expect to be made -- critic only runs
    # once per ``explain`` call.
    mock_critique.return_value = CriticReport(
        verdict="needs_more_evidence",
        issues=["explanation does not reference issue #99 from PR body"],
        focus_hints=["fetch issue #99 referenced in PR body"],
        reasoning="missing context",
        available=True,
        model="claude-haiku-4-5",
    )

    agent = GitExplainerAgent(use_llm=True, use_planner=True, use_critic=True)
    result = agent.explain(query)

    # Critic ran exactly once.
    assert mock_critique.call_count == 1
    # Re-plan flag was set.
    assert result["critic"]["replanned"] is True
    # Issue #99 was fetched on the re-plan and made it into the evidence.
    assert any(iss["number"] == 99 for iss in result["issues"])
    # Synthesizer ran twice (first draft, then post-replan re-synthesis).
    synth_calls = [
        c for c in mock_chat.call_args_list if "Explain why" in c.args[0]
    ]
    assert len(synth_calls) == 2


# ---------------------------------------------------------------------------
# Critic skipped when synthesis already used the fallback
# ---------------------------------------------------------------------------


@patch("git_explainer.orchestrator.critic_mod.critique")
@patch("git_explainer.orchestrator.is_available", return_value=False)
@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.read_file_at_revision")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_critic_does_not_run_when_synthesis_used_fallback(
    mock_validate,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_read_file,
    mock_get_diff,
    mock_synth_available,
    mock_critique,
    tmp_path,
):
    """Fallback synthesis is by construction faithful (deterministic
    template). Spending a critic call on it is wasted budget."""
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "A", "date": "2024-06-01", "message": "Fix"},
    ]
    mock_find_prs.return_value = []
    mock_fetch_pr.return_value = None
    mock_fetch_pr_comments.return_value = []
    mock_read_file.return_value = ""
    mock_get_diff.return_value = {"files": []}

    agent = GitExplainerAgent(use_llm=True, use_planner=False, use_critic=True)
    result = agent.explain(query)

    assert result["used_fallback"] is True
    mock_critique.assert_not_called()
    # Critic field is None when the critic was skipped before being asked.
    assert result["critic"] is None


# ---------------------------------------------------------------------------
# Backward compat: default agent matches old behavior (no planner, no critic)
# ---------------------------------------------------------------------------


@patch("git_explainer.orchestrator.get_diff")
@patch("git_explainer.orchestrator.read_file_at_revision")
@patch("git_explainer.orchestrator.fetch_pr_comments")
@patch("git_explainer.orchestrator.fetch_pr")
@patch("git_explainer.orchestrator.find_prs_for_commit")
@patch("git_explainer.orchestrator.trace_line_history")
@patch("git_explainer.orchestrator.validate_query")
def test_default_agent_has_no_planner_or_critic_field_set(
    mock_validate,
    mock_trace,
    mock_find_prs,
    mock_fetch_pr,
    mock_fetch_pr_comments,
    mock_read_file,
    mock_get_diff,
    tmp_path,
):
    """Constructing GitExplainerAgent() with no flags should leave the
    new ``planner`` / ``critic`` result fields as None so existing
    callers see exactly the same payload as before."""
    query = _make_query(tmp_path)
    mock_validate.return_value = query
    mock_trace.return_value = [
        {"sha": "abc1234", "full_sha": "abc123456789", "author": "A", "date": "2024-06-01", "message": "Fix"},
    ]
    mock_find_prs.return_value = []
    mock_fetch_pr.return_value = None
    mock_fetch_pr_comments.return_value = []
    mock_read_file.return_value = ""
    mock_get_diff.return_value = {"files": []}

    result = GitExplainerAgent(use_llm=False).explain(query)
    assert result.get("planner") is None
    assert result.get("critic") is None

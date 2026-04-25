"""Class demo: Git History Explainer with Planner + Critic.

DESIGN GOAL: zero failure points. Runs in under 10 seconds, every time.

MODES:
    (default)   Scripted LLMs. The Planner/Synthesizer/Critic responses
                are hand-crafted to match what the real models produced
                in our recorded runs. EVERYTHING else -- the planner
                driver, dispatcher, evidence merging, citation
                validation, re-plan loop, fallback handling -- runs as
                in production. Real `git` commands. Real evidence.
                Mocked GitHub HTTP (canned PR data) so the demo is
                offline and rate-limit-free.

    --live      Same script, but the Critic uses the real Anthropic
                Claude Haiku API. Planner + Synthesizer stay scripted
                so iteration count and prompts stay predictable.

    --full-live Real Anthropic Critic AND real Groq Planner + Synth.
                Best-effort -- not guaranteed to work if rate limits
                are tight. Use only if you want the live flex.

Run with --help to see options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Force the small model so a stray --full-live still uses something
# inside the project's daily quota.
os.environ.setdefault("PLANNER_MODEL", "llama-3.1-8b-instant")
os.environ.setdefault("GROQ_MODEL", "llama-3.1-8b-instant")

import git_explainer.critic as critic_mod
import git_explainer.llm as llm_mod
import git_explainer.tool_registry as tool_registry
import git_explainer.tools.github_http as github_http
from git_explainer.guardrails import ExplainerQuery
from git_explainer.orchestrator import GitExplainerAgent


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


WIDTH = 90


def hr(char: str = "-") -> None:
    print(char * WIDTH)


def title(text: str) -> None:
    print()
    hr("=")
    print(f"  {text}")
    hr("=")


def section(text: str) -> None:
    print()
    print(f">>> {text}")
    hr("-")


def bullet(label: str, value: str) -> None:
    print(f"   - {label:<22} {value}")


def block(label: str, body: str, *, indent: str = "    ") -> None:
    print(f"   [{label}]")
    for line in body.rstrip().splitlines():
        print(f"{indent}{line}")


def trim(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 60] + "\n   ... [trimmed for readability] ..."


# ---------------------------------------------------------------------------
# Scripted LLM brain
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Two-channel deterministic LLM stub.

    Routes prompts to a Planner queue or a Synthesizer queue based on
    the user-prompt header (the synthesizer prompt always starts with
    'Explain why ...', possibly preceded by the orchestrator's
    retry-stricter-prompt preamble; everything else is a Planner turn).

    Each call advances the corresponding queue's cursor. This avoids
    substring collisions with file contents in the prompt -- the prior
    pattern-matching approach was bitten by 'read_file_at_revision'
    appearing inside the file_context_reader.py source code that ended
    up in the synthesizer's evidence dump.
    """

    def __init__(self, *, planner: list[str], synth: list[str]):
        self._planner = list(planner)
        self._synth = list(synth)
        self._p_i = 0
        self._s_i = 0

    @staticmethod
    def _is_synth(prompt: str) -> bool:
        # Both the first-attempt synth prompt and the retry-with-stricter
        # preamble include 'Explain why the selected code exists' near
        # the top. The planner prompt never does.
        return "Explain why the selected code exists" in prompt[:600]

    def chat(self, prompt: str, **kwargs) -> str:
        if self._is_synth(prompt):
            if self._s_i >= len(self._synth):
                # Validation retry: replay the last synth answer (it
                # already passed citation coverage so the second pass
                # also accepts it).
                return self._synth[-1]
            reply = self._synth[self._s_i]
            self._s_i += 1
            return reply
        if self._p_i >= len(self._planner):
            raise RuntimeError(
                f"ScriptedLLM: out of planner replies. "
                f"Prompt head: {prompt[:200]!r}"
            )
        reply = self._planner[self._p_i]
        self._p_i += 1
        return reply


# ---------------------------------------------------------------------------
# Mocked GitHub HTTP -- canned responses keyed by URL
# ---------------------------------------------------------------------------


_GITHUB_FIXTURES: dict[str, Any] = {
    # Repo existence check (guardrails.ensure_public_github_repo) is a
    # plain ``requests.get`` -- not via github_http. Patched separately.
    # PR #1 lookup
    "https://api.github.com/repos/AndreiPiterbarg/CIS_1990_Final_Project/commits/3870c344c0ff5b8da56a85dc8a9a896bfa7bb075/pulls": [
        {"number": 1}
    ],
    "https://api.github.com/repos/AndreiPiterbarg/CIS_1990_Final_Project/pulls/1": {
        "number": 1,
        "title": "initial mockup",
        "body": (
            "Do not merge yet -- still needs some edits/review. "
            "Adds the initial agent skeleton, env-driven config "
            "(replacing hard-coded credentials), and the LLM provider "
            "switch from Kimi to Groq for cost reasons."
        ),
        "state": "open",
        "merged": True,
        "merged_at": "2026-04-14T21:13:26Z",
        "created_at": "2026-04-14T05:55:42Z",
        "user": {"login": "aking526"},
        "base": {"ref": "master"},
        "head": {"ref": "agentv1"},
        "merge_commit_sha": "6d82e7ee9c8739060c02bb5586379566a27a00f7",
    },
    "https://api.github.com/repos/AndreiPiterbarg/CIS_1990_Final_Project/pulls/1/comments": [],
    # Demo 2: file_context_reader commits have no associated PR
    "https://api.github.com/repos/AndreiPiterbarg/CIS_1990_Final_Project/commits/e0f2b80dc00ffa2b5bf063c64f490fe3e53b183e/pulls": [],
    "https://api.github.com/repos/AndreiPiterbarg/CIS_1990_Final_Project/commits/50c52bb5e27dc43b87676facc2338d8c0506aa5c/pulls": [],
}


def _fake_github_get_json(url, *, headers=None, retries=3, memory=None,
                          etag_cache=None):
    from git_explainer.tools.github_http import GitHubResponse

    if url in _GITHUB_FIXTURES:
        return GitHubResponse(
            data=_GITHUB_FIXTURES[url],
            status_code=200,
            headers={},
            from_cache=False,
        )
    return GitHubResponse(data=None, status_code=404, headers={}, from_cache=False)


def _fake_repo_check(owner: str, repo_name: str) -> dict:
    return {"private": False, "default_branch": "master"}


# ---------------------------------------------------------------------------
# Tracing wrapper (records every step for the on-screen narration)
# ---------------------------------------------------------------------------


_step_counter = 0
_tool_call_counter = 0


def _next_step() -> int:
    global _step_counter
    _step_counter += 1
    return _step_counter


def _next_tool() -> int:
    global _tool_call_counter
    _tool_call_counter += 1
    return _tool_call_counter


def _install_tracing(scripted_llm: ScriptedLLM, scripted_critic_text: str | None,
                     *, live_critic: bool) -> None:
    """Wrap the LLM + dispatcher + critic so every call narrates itself."""
    real_dispatch = tool_registry.dispatch_tool

    def traced_chat(prompt, *, system_prompt="", history=None,
                    model=None, max_tokens=None, temperature=0.3):
        if prompt.lstrip().startswith("Explain why"):
            kind = "SYNTHESIZER"
        elif "Decide the next action" in prompt:
            kind = "PLANNER"
        else:
            kind = "LLM"
        n = _next_step()
        section(f"STEP {n}: {kind} -> Groq llama-3.1-8b-instant")
        block(f"{kind} prompt (key fragment)",
              _summarize_prompt(prompt, kind))
        reply = scripted_llm.chat(prompt)
        block(f"{kind} reply", trim(reply, 700))
        return reply

    def traced_dispatch(name, arguments, context):
        n = _next_tool()
        section(f"  TOOL CALL #{n}: {name}({json.dumps(arguments)})")
        result = real_dispatch(name, arguments, context)
        summary = tool_registry._summarize_result(name, result)
        bullet("result", summary)
        return result

    def scripted_critic(prompt: str) -> str:
        n = _next_step()
        section(f"STEP {n}: CRITIC -> Anthropic Claude Haiku 4.5  (SCRIPTED)")
        block("critic prompt (key fragment)", _summarize_prompt(prompt, "CRITIC"))
        block("critic reply", trim(scripted_critic_text or "{}", 800))
        return scripted_critic_text or '{"verdict":"ok","issues":[],"focus_hints":[],"reasoning":""}'

    # Capture the ORIGINAL anthropic call BEFORE we patch the module
    # attribute -- otherwise the wrapper would re-import the (now
    # patched) name and recurse.
    real_anthropic_call = critic_mod._call_anthropic_critic

    def live_critic_call(prompt: str) -> str:
        n = _next_step()
        section(f"STEP {n}: CRITIC -> Anthropic Claude Haiku 4.5  (LIVE)")
        block("critic prompt (key fragment)", _summarize_prompt(prompt, "CRITIC"))
        reply = real_anthropic_call(prompt)
        block("critic reply (LIVE)", trim(reply, 800))
        return reply

    llm_mod.chat = traced_chat
    tool_registry.dispatch_tool = traced_dispatch
    import git_explainer.orchestrator as orch
    orch.chat = traced_chat

    if live_critic:
        critic_mod._call_anthropic_critic = live_critic_call
    else:
        critic_mod._call_anthropic_critic = scripted_critic
        # Force is_available to True so the orchestrator runs the critic.
        critic_mod.is_available = lambda: True

    # Mock GitHub HTTP everywhere it's used.
    github_http.github_get_json = _fake_github_get_json

    # Mock the public-repo guardrail check.
    import git_explainer.guardrails as guardrails
    guardrails.ensure_public_github_repo = _fake_repo_check


def _summarize_prompt(prompt: str, kind: str) -> str:
    """Return a short on-screen-friendly excerpt of the prompt.

    The full prompt is several KB. For the audience we want the part
    that changes per turn (the question, the evidence so far, the
    focus hints) -- not the static system prompt or tool schemas.
    """
    if kind == "PLANNER":
        # Show iteration number, evidence-so-far summary, and focus hints
        bits: list[str] = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("Iteration "):
                bits.append(stripped)
                break
        if "Focus hints from the critic" in prompt:
            idx = prompt.index("Focus hints from the critic")
            tail = prompt[idx:idx + 600]
            bits.append(tail)
        else:
            # Show the evidence-so-far summary line(s)
            if "Evidence collected so far" in prompt:
                idx = prompt.index("Evidence collected so far")
                end = prompt.find("Tool call history", idx)
                if end == -1:
                    end = idx + 800
                bits.append(prompt[idx:min(end, idx + 800)].rstrip())
        return "\n".join(bits) or trim(prompt, 400)

    if kind == "SYNTHESIZER":
        # Show the query and a slim header.
        idx = prompt.find("Query:")
        if idx == -1:
            return trim(prompt, 400)
        end = prompt.find("Evidence:", idx)
        head = prompt[idx:end if end > 0 else idx + 400].rstrip()
        return head

    if kind == "CRITIC":
        idx = prompt.find("Draft explanation:")
        if idx == -1:
            return trim(prompt, 400)
        end = prompt.find("Evidence available", idx)
        return prompt[idx:end if end > 0 else idx + 600].rstrip()

    return trim(prompt, 400)


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------


def _scripts_demo_1() -> tuple[ScriptedLLM, str]:
    """Scenario: config.py:13-19 -- happy path, critic approves."""
    planner = [
        json.dumps({
            "action": "call_tool",
            "tool": "find_prs_for_commit",
            "arguments": {"commit_sha": "3870c344c0ff5b8da56a85dc8a9a896bfa7bb075"},
            "reasoning": "The seeded commits include 3870c34 'initial mockup'. Look up its PR for design rationale.",
        }),
        json.dumps({
            "action": "call_tool",
            "tool": "fetch_pr",
            "arguments": {"pr_number": 1},
            "reasoning": "PR #1 is associated with the seed commit; fetch its body for intent.",
        }),
        json.dumps({
            "action": "call_tool",
            "tool": "get_diff",
            "arguments": {
                "commit_sha": "3870c344c0ff5b8da56a85dc8a9a896bfa7bb075",
                "file_path": "git_explainer/config.py",
            },
            "reasoning": "Get the actual diff so synthesis can ground claims in real changes.",
        }),
        json.dumps({
            "action": "done",
            "reasoning": "Have commits, the linked PR with body, and the diff. Enough for a faithful synthesis.",
        }),
        # --- Re-plan tail: if --live's real critic disagrees with the
        # scripted ok-verdict and asks for more, terminate the re-plan
        # immediately rather than running out of replies and crashing.
        json.dumps({
            "action": "done",
            "reasoning": "No additional evidence available within the planner's tool budget.",
        }),
        json.dumps({
            "action": "done",
            "reasoning": "Re-plan exhausted; returning current evidence.",
        }),
    ]

    synth = [json.dumps({
        "what_changed": (
            "Lines 13-19 of git_explainer/config.py declare the GitHub and Groq "
            "credentials and the Groq endpoint defaults [commit:3870c34]. "
            "The diff shows these were rewritten from a fail-fast lookup to "
            "os.getenv with empty-string defaults [commit:3870c34]."
        ),
        "why": (
            "The associated PR #1 'initial mockup' states the change adds "
            "env-driven config replacing hard-coded credentials and switches "
            "the LLM provider from Kimi to Groq for cost reasons [pr:#1]. "
            "The softer os.getenv default keeps the package importable in "
            "tests and offline workflows that lack credentials [commit:3870c34]."
        ),
        "tradeoffs": (
            "Defaulting to empty strings trades fail-fast behaviour for "
            "import-time robustness; callers must check has_groq_api_key "
            "before issuing LLM calls [commit:3870c34]."
        ),
        "limitations": (
            "PR #1 review comments are empty so we cannot cite individual "
            "reviewer concerns [pr:#1]. The explanation is grounded in the "
            "PR body and diff alone [commit:3870c34]."
        ),
        "summary": (
            "Lines 13-19 establish env-backed credentials with safe defaults, "
            "introduced in PR #1 as part of the Kimi to Groq switch "
            "[pr:#1] [commit:3870c34]."
        ),
    })]

    critic_ok = json.dumps({
        "verdict": "ok",
        "issues": [],
        "focus_hints": [],
        "reasoning": (
            "Every claim cites either commit 3870c34 or PR #1, both of which "
            "are present in the evidence. The PR body supports the "
            "Kimi to Groq rationale. The diff supports the os.environ to "
            "os.getenv claim. No unsupported claims."
        ),
    })

    return ScriptedLLM(planner=planner, synth=synth), critic_ok


def _scripts_demo_2() -> tuple[ScriptedLLM, str]:
    """Scenario: file_context_reader.py:59-70 -- the critic catches a
    thin claim, re-plan fetches surrounding code context, re-synthesis
    produces a stronger explanation.

    Returns (ScriptedLLM, scripted_critic_reply).
    """
    planner = [
        # First-pass round 1: fetch the most-recent diff.
        json.dumps({
            "action": "call_tool",
            "tool": "get_diff",
            "arguments": {
                "commit_sha": "e0f2b80dc00ffa2b5bf063c64f490fe3e53b183e",
                "file_path": "git_explainer/tools/file_context_reader.py",
            },
            "reasoning": "Most-recent commit on these lines. Diff shows what 'simplify error handling' actually means.",
        }),
        # Round 2: older commit's diff.
        json.dumps({
            "action": "call_tool",
            "tool": "get_diff",
            "arguments": {
                "commit_sha": "50c52bb5e27dc43b87676facc2338d8c0506aa5c",
                "file_path": "git_explainer/tools/file_context_reader.py",
            },
            "reasoning": "Older commit -- the original add. Diff establishes baseline behavior.",
        }),
        # Round 3: done.
        json.dumps({
            "action": "done",
            "reasoning": "Two commits, both diffs in evidence. No PRs exist for these commits per the seed search.",
        }),
        # Re-plan round 1: critic asked for surrounding code context.
        json.dumps({
            "action": "call_tool",
            "tool": "read_file_at_revision",
            "arguments": {
                "file_path": "git_explainer/tools/file_context_reader.py",
                "revision": "e0f2b80dc00ffa2b5bf063c64f490fe3e53b183e",
                "start_line": 50,
                "end_line": 80,
            },
            "reasoning": "Critic asked for surrounding code so the rationale can be grounded, not assumed.",
        }),
        # Re-plan round 2: done.
        json.dumps({
            "action": "done",
            "reasoning": "File context fetched. Re-synthesis can now ground the 'why' in actual code.",
        }),
    ]

    synth_v1 = json.dumps({
        "what_changed": (
            "Lines 59-70 of file_context_reader.py implement _read_from_revision, "
            "the helper that runs git show to read a file at a specific SHA "
            "[commit:50c52bb]. Commit e0f2b80 simplified its error handling "
            "[commit:e0f2b80]."
        ),
        "why": (
            "These commits added then refined the git-show based file reader "
            "[commit:50c52bb] [commit:e0f2b80]. Beyond the commit messages, no "
            "PR or issue rationale is available [commit:e0f2b80]."
        ),
        "tradeoffs": (
            "No documented trade-offs in the available metadata "
            "[commit:e0f2b80] [commit:50c52bb]."
        ),
        "limitations": (
            "The diffs alone do not show the surrounding control flow, so we "
            "cannot fully explain why error handling was simplified "
            "[commit:e0f2b80]."
        ),
        "summary": (
            "Lines 59-70 are the _read_from_revision helper, added in 50c52bb "
            "and simplified in e0f2b80 [commit:50c52bb] [commit:e0f2b80]."
        ),
    })

    synth_v2 = json.dumps({
        "what_changed": (
            "Lines 59-70 implement _read_from_revision, which shells out to "
            "git show to read a file at a specific revision [commit:50c52bb]. "
            "The function returns None on a non-zero git exit and decodes "
            "stdout as UTF-8, falling back to '[binary file]' on decode error "
            "[commit:e0f2b80]."
        ),
        "why": (
            "Commit 50c52bb introduced the helper as part of the initial "
            "git-tools scaffolding [commit:50c52bb]. Commit e0f2b80 then "
            "replaced an exception-based error path with a None return so "
            "callers do not have to wrap each read in try/except "
            "[commit:e0f2b80]."
        ),
        "tradeoffs": (
            "Returning None for both 'file does not exist at this revision' "
            "and 'git command failed' merges two failure modes [commit:e0f2b80]. "
            "Callers cannot distinguish them without extra plumbing "
            "[commit:e0f2b80]."
        ),
        "limitations": (
            "No PR or issue is linked, so the only documented intent is the "
            "commit messages and the diff [commit:e0f2b80] [commit:50c52bb]."
        ),
        "summary": (
            "_read_from_revision was added in 50c52bb and reshaped in e0f2b80 "
            "to use a None return instead of raising, simplifying caller code "
            "at the cost of merging two failure modes "
            "[commit:50c52bb] [commit:e0f2b80]."
        ),
    })

    critic_needs_more = json.dumps({
        "verdict": "needs_more_evidence",
        "issues": [
            "The draft says e0f2b80 'simplified error handling' but never shows what "
            "the simplification was. Without the surrounding function context, the "
            "claim is just a paraphrased commit message.",
        ],
        "focus_hints": [
            "Read file_context_reader.py at revision e0f2b80 around lines 50-80 to "
            "see the full _read_from_revision function and verify what 'simplify "
            "error handling' actually means.",
        ],
        "reasoning": (
            "All citations resolve to evidence, but the 'why' section is too thin -- "
            "the e0f2b80 commit message is generic, and the diff alone shows added "
            "lines without the function shape. Reading the file at that revision "
            "would resolve this."
        ),
    })

    return (
        ScriptedLLM(planner=planner, synth=[synth_v1, synth_v2]),
        critic_needs_more,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_scenario(*, title_text: str, query: ExplainerQuery,
                 scripted_llm: ScriptedLLM, critic_text: str,
                 live_critic: bool) -> None:
    global _step_counter, _tool_call_counter
    _step_counter = 0
    _tool_call_counter = 0

    title(title_text)
    bullet("repo", str(Path(query.repo_path).name))
    bullet("file", query.file_path or "<question mode>")
    bullet("lines", f"{query.start_line}-{query.end_line}")
    bullet("github", f"{query.owner}/{query.repo_name}")
    bullet("flags", "use_llm + use_planner + use_critic")
    bullet("LLM mode",
           "scripted Planner+Synth, " +
           ("LIVE Anthropic Critic" if live_critic else "scripted Critic"))

    _install_tracing(scripted_llm, critic_text, live_critic=live_critic)

    t0 = time.time()
    agent = GitExplainerAgent(use_llm=True, use_planner=True, use_critic=True)
    result = agent.explain(query)
    elapsed = time.time() - t0

    title("RESULT")
    bullet("elapsed", f"{elapsed:.2f} s")
    bullet("commits found", str(len(result["commits"])))
    bullet("PRs fetched", str(len(result["pull_requests"])))
    bullet("diffs gathered", str(len(result["diffs"])))
    bullet("planner.iters", str(result["planner"]["iterations_used"])
           if result.get("planner") else "n/a")
    bullet("planner.halted", result["planner"]["halted_reason"]
           if result.get("planner") else "n/a")
    bullet("critic.verdict", result["critic"]["verdict"]
           if result.get("critic") else "n/a")
    bullet("critic.replanned", str(result["critic"].get("replanned", False))
           if result.get("critic") else "n/a")
    bullet("used_fallback", str(result["used_fallback"]))

    section("FINAL EXPLANATION")
    for key in ("what_changed", "why", "tradeoffs", "limitations", "summary"):
        print(f"\n   ## {key}")
        body = result["explanation"].get(key, "") or "(empty)"
        for line in textwrap.wrap(body, width=WIDTH - 6):
            print(f"      {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Class demo for the Git History Explainer."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real Anthropic Claude critic (requires ANTHROPIC_KEY).",
    )
    parser.add_argument(
        "--full-live",
        action="store_true",
        help="Use real LLMs for everything (requires GROQ_API_KEY + ANTHROPIC_KEY).",
    )
    parser.add_argument(
        "--scenario",
        choices=["1", "2", "both"],
        default="both",
    )
    args = parser.parse_args(argv)

    if args.full_live:
        # Fully live mode: don't install scripted patches. Just run the
        # agent for real. No safety net.
        return _run_full_live(args.scenario)

    repo = str(Path(__file__).parent.resolve())

    print()
    hr("#")
    print("  GIT HISTORY EXPLAINER -- LIVE CLASS DEMO")
    hr("#")
    print()
    print(textwrap.fill(
        "Architecture: 3-LLM agentic pipeline. Layer 1 PLANNER decides "
        "which deterministic git/GitHub tools to invoke. Layer 2 "
        "SYNTHESIZER writes the cited explanation prose. Layer 3 CRITIC "
        "(independent provider) grades the draft and may trigger one "
        "re-plan + re-synthesis round.",
        width=WIDTH))
    print()
    print(textwrap.fill(
        "Demo mode: Planner and Synthesizer use deterministic scripted "
        "responses (the actual production code paths run unchanged -- "
        "only the LLM brain is canned for reliability). " + (
            "Critic is LIVE: real Anthropic Claude Haiku 4.5 over the "
            "wire." if args.live else
            "Critic uses a recorded scripted response. Add --live to use "
            "the real Anthropic Claude Haiku 4.5 instead."),
        width=WIDTH))

    if args.scenario in ("1", "both"):
        llm1, critic1 = _scripts_demo_1()
        run_scenario(
            title_text="DEMO 1: config.py:13-19  ('Why these credential lines?')",
            query=ExplainerQuery(
                repo_path=repo,
                file_path="git_explainer/config.py",
                start_line=13,
                end_line=19,
                owner="AndreiPiterbarg",
                repo_name="CIS_1990_Final_Project",
                max_commits=5,
            ),
            scripted_llm=llm1,
            critic_text=critic1,
            live_critic=args.live,
        )

    if args.scenario in ("2", "both"):
        llm2, critic2_v1 = _scripts_demo_2()
        run_scenario(
            title_text="DEMO 2: file_context_reader.py:59-70  (critic catches a thin claim, re-plan fixes it)",
            query=ExplainerQuery(
                repo_path=repo,
                file_path="git_explainer/tools/file_context_reader.py",
                start_line=59,
                end_line=70,
                owner="AndreiPiterbarg",
                repo_name="CIS_1990_Final_Project",
                max_commits=5,
            ),
            scripted_llm=llm2,
            critic_text=critic2_v1,
            live_critic=args.live,
        )

    print()
    title("DEMO COMPLETE")
    print()
    return 0


def _run_full_live(scenario: str) -> int:
    """Hard mode: use real LLMs for everything. No safety net."""
    print("WARN: --full-live is best-effort and may fail if rate limits are tight.",
          file=sys.stderr)
    from main import explain_code_history
    repo = str(Path(__file__).parent.resolve())
    if scenario in ("1", "both"):
        explain_code_history(
            repo, "git_explainer/config.py", 13, 19,
            owner="AndreiPiterbarg", repo_name="CIS_1990_Final_Project",
            use_llm=True, use_planner=True, use_critic=True,
        )
    if scenario in ("2", "both"):
        explain_code_history(
            repo, "git_explainer/tools/file_context_reader.py", 59, 70,
            owner="AndreiPiterbarg", repo_name="CIS_1990_Final_Project",
            use_llm=True, use_planner=True, use_critic=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

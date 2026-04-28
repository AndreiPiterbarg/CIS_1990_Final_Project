"""Reusable scripted fixtures for the live demo.

These are the useful pieces from the older demo harness: presentation
printing helpers, deterministic LLM replies, and canned GitHub responses.
The actual planner loop, tool dispatcher, evidence merging, citation
validation, and critic control flow still run through production code.
"""

from __future__ import annotations

import json
import os
from typing import Any


WIDTH = 90

COLOR_PROMPT = "\033[36m"  # cyan
COLOR_REPLY = "\033[33m"   # yellow
COLOR_RESET = "\033[0m"


def _colors_enabled() -> bool:
    return not os.getenv("NO_COLOR")


def pretty_json(text: str) -> str:
    """Pretty-print a JSON string. If it isn't valid JSON, return as-is."""
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return text
    return json.dumps(parsed, indent=2, ensure_ascii=False)


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


def block(label: str, body: str, *, indent: str = "    ", color: str | None = None) -> None:
    print(f"   [{label}]")
    use_color = color and _colors_enabled()
    for line in body.rstrip().splitlines():
        if use_color:
            print(f"{indent}{color}{line}{COLOR_RESET}")
        else:
            print(f"{indent}{line}")


def trim(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 60] + "\n   ... [trimmed for readability] ..."


class ScriptedLLM:
    """Deterministic two-channel LLM stub for planner and synthesizer turns."""

    def __init__(self, *, planner: list[str], synth: list[str]):
        self._planner = list(planner)
        self._synth = list(synth)
        self._planner_index = 0
        self._synth_index = 0

    @staticmethod
    def _is_synth(prompt: str) -> bool:
        return "Explain why the selected code exists" in prompt[:600]

    def chat(self, prompt: str, **kwargs) -> str:
        if self._is_synth(prompt):
            if self._synth_index >= len(self._synth):
                return self._synth[-1]
            reply = self._synth[self._synth_index]
            self._synth_index += 1
            return reply

        if self._planner_index >= len(self._planner):
            raise RuntimeError(
                "ScriptedLLM: out of planner replies. "
                f"Prompt head: {prompt[:200]!r}"
            )
        reply = self._planner[self._planner_index]
        self._planner_index += 1
        return reply


_GITHUB_FIXTURES: dict[str, Any] = {
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
    "https://api.github.com/repos/AndreiPiterbarg/CIS_1990_Final_Project/commits/e0f2b80dc00ffa2b5bf063c64f490fe3e53b183e/pulls": [],
    "https://api.github.com/repos/AndreiPiterbarg/CIS_1990_Final_Project/commits/50c52bb5e27dc43b87676facc2338d8c0506aa5c/pulls": [],
}


def fake_github_get_json(url, *, headers=None, retries=3, memory=None, etag_cache=None):
    from git_explainer.tools.github_http import GitHubResponse

    if url in _GITHUB_FIXTURES:
        return GitHubResponse(
            data=_GITHUB_FIXTURES[url],
            status_code=200,
            headers={},
            from_cache=False,
        )
    return GitHubResponse(data=None, status_code=404, headers={}, from_cache=False)


def fake_repo_check(owner: str, repo_name: str) -> dict:
    return {"private": False, "default_branch": "master"}


def summarize_prompt(prompt: str, kind: str) -> str:
    """Return a short on-screen-friendly excerpt of a long agent prompt."""
    if kind == "PLANNER":
        bits: list[str] = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("Iteration "):
                bits.append(stripped)
                break
        if "Focus hints from the critic" in prompt:
            idx = prompt.index("Focus hints from the critic")
            bits.append(prompt[idx:idx + 600])
        elif "Evidence collected so far" in prompt:
            idx = prompt.index("Evidence collected so far")
            end = prompt.find("Tool call history", idx)
            if end == -1:
                end = idx + 800
            bits.append(prompt[idx:min(end, idx + 800)].rstrip())
        return "\n".join(bits) or trim(prompt, 400)

    if kind == "SYNTHESIZER":
        idx = prompt.find("Query:")
        if idx == -1:
            return trim(prompt, 400)
        end = prompt.find("Evidence:", idx)
        return prompt[idx:end if end > 0 else idx + 400].rstrip()

    if kind == "CRITIC":
        idx = prompt.find("Draft explanation:")
        if idx == -1:
            return trim(prompt, 400)
        end = prompt.find("Evidence available", idx)
        return prompt[idx:end if end > 0 else idx + 600].rstrip()

    return trim(prompt, 400)


def scripts_demo_1() -> tuple[ScriptedLLM, str]:
    """Happy path: config.py credential/provider lines, critic approves."""
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
            "are present in the evidence. The PR body supports the Kimi to "
            "Groq rationale. The diff supports the os.environ to os.getenv "
            "claim. No unsupported claims."
        ),
    })

    return ScriptedLLM(planner=planner, synth=synth), critic_ok


def scripts_demo_2() -> tuple[ScriptedLLM, str]:
    """Critic loop: first draft is thin, re-plan fetches file context."""
    planner = [
        json.dumps({
            "action": "call_tool",
            "tool": "get_diff",
            "arguments": {
                "commit_sha": "e0f2b80dc00ffa2b5bf063c64f490fe3e53b183e",
                "file_path": "git_explainer/tools/file_context_reader.py",
            },
            "reasoning": "Most-recent commit on these lines. Diff shows what 'simplify error handling' actually means.",
        }),
        json.dumps({
            "action": "call_tool",
            "tool": "get_diff",
            "arguments": {
                "commit_sha": "50c52bb5e27dc43b87676facc2338d8c0506aa5c",
                "file_path": "git_explainer/tools/file_context_reader.py",
            },
            "reasoning": "Older commit -- the original add. Diff establishes baseline behavior.",
        }),
        json.dumps({
            "action": "done",
            "reasoning": "Two commits, both diffs in evidence. No PRs exist for these commits per the seed search.",
        }),
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

    return ScriptedLLM(planner=planner, synth=[synth_v1, synth_v2]), critic_needs_more

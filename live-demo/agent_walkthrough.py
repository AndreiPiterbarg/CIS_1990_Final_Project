#!/usr/bin/env python3
"""Corrected step-by-step walkthrough for the live-demo folder.

This imports the existing demo fixtures from demo_show.py, but installs
the tracing patches against the current planner bindings too. That keeps
the presentation in live-demo while leaving the root demo harness alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import demo_show as base
import git_explainer.critic as critic_mod
import git_explainer.guardrails as guardrails
import git_explainer.llm as llm_mod
import git_explainer.orchestrator as orch
import git_explainer.planner as planner_mod
import git_explainer.tool_registry as tool_registry
import git_explainer.tools.github_http as github_http
import git_explainer.tools.github_issue_lookup as github_issue_lookup
import git_explainer.tools.github_pr_lookup as github_pr_lookup
from git_explainer.guardrails import ExplainerQuery
from git_explainer.orchestrator import GitExplainerAgent


WIDTH = 90
_step_counter = 0
_tool_counter = 0
_pause_between_steps = True


def main(argv: list[str] | None = None) -> int:
    global _pause_between_steps

    parser = argparse.ArgumentParser(description="Step-by-step Git Explainer walkthrough.")
    parser.add_argument("--scenario", choices=["1", "2", "both"], default="both")
    parser.add_argument(
        "--live-critic",
        action="store_true",
        help="Use the real Anthropic critic; planner and synthesizer stay scripted.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run straight through instead of waiting for Enter between steps.",
    )
    args = parser.parse_args(argv)
    _pause_between_steps = not args.no_pause

    print()
    base.hr("#")
    print("  GIT HISTORY EXPLAINER -- LIVE CLASS DEMO")
    base.hr("#")
    print()
    print(textwrap.fill(
        "Architecture: the planner chooses deterministic tools, the tool "
        "dispatcher gathers auditable evidence, the synthesizer writes a "
        "cited explanation, and the critic can request one more retrieval "
        "round when the evidence is thin.",
        width=WIDTH,
    ))
    print()
    print(textwrap.fill(
        "Demo mode: planner and synthesizer replies are scripted for "
        "reliability. The git tools, planner loop, dispatcher, evidence "
        "merging, citation validation, and critic-control flow run through "
        "the real production code.",
        width=WIDTH,
    ))
    if should_pause():
        print()
        print("Interactive mode: press Enter after each step to continue.")

    if args.scenario in ("1", "both"):
        llm, critic_text = base._scripts_demo_1()
        run_scenario(
            "DEMO 1: config.py:13-19  ('Why these credential lines?')",
            ExplainerQuery(
                repo_path=str(PROJECT_ROOT),
                file_path="git_explainer/config.py",
                start_line=13,
                end_line=19,
                owner="AndreiPiterbarg",
                repo_name="CIS_1990_Final_Project",
                max_commits=5,
            ),
            llm,
            critic_text,
            live_critic=args.live_critic,
        )

    if args.scenario in ("2", "both"):
        llm, critic_text = base._scripts_demo_2()
        run_scenario(
            "DEMO 2: file_context_reader.py:59-70  (critic catches a thin claim)",
            ExplainerQuery(
                repo_path=str(PROJECT_ROOT),
                file_path="git_explainer/tools/file_context_reader.py",
                start_line=59,
                end_line=70,
                owner="AndreiPiterbarg",
                repo_name="CIS_1990_Final_Project",
                max_commits=5,
            ),
            llm,
            critic_text,
            live_critic=args.live_critic,
        )

    base.title("DEMO COMPLETE")
    return 0


def install_tracing(
    scripted_llm: base.ScriptedLLM,
    critic_text: str,
    *,
    live_critic: bool,
) -> None:
    real_dispatch = planner_mod.dispatch_tool
    real_anthropic_call = critic_mod._call_anthropic_critic

    def traced_chat(prompt, *, system_prompt="", history=None,
                    model=None, max_tokens=None, temperature=0.3):
        kind = classify_prompt(prompt)
        step = next_step()
        base.section(f"STEP {step}: {kind} -> Groq llama-3.1-8b-instant")
        base.block(f"{kind} prompt (key fragment)", base._summarize_prompt(prompt, kind))
        reply = scripted_llm.chat(prompt)
        base.block(f"{kind} reply", base.trim(reply, 700))
        pause_for_enter()
        return reply

    def traced_dispatch(name, arguments, context):
        tool_number = next_tool()
        base.section(f"  TOOL CALL #{tool_number}: {name}({json.dumps(arguments)})")
        result = real_dispatch(name, arguments, context)
        base.bullet("result", tool_registry._summarize_result(name, result))
        pause_for_enter()
        return result

    def scripted_critic(prompt: str) -> str:
        step = next_step()
        base.section("STEP " + str(step) + ": CRITIC -> Anthropic Claude Haiku 4.5  (SCRIPTED)")
        base.block("critic prompt (key fragment)", base._summarize_prompt(prompt, "CRITIC"))
        base.block("critic reply", base.trim(critic_text, 800))
        pause_for_enter()
        return critic_text

    def live_critic_call(prompt: str) -> str:
        step = next_step()
        base.section("STEP " + str(step) + ": CRITIC -> Anthropic Claude Haiku 4.5  (LIVE)")
        base.block("critic prompt (key fragment)", base._summarize_prompt(prompt, "CRITIC"))
        reply = real_anthropic_call(prompt)
        base.block("critic reply (LIVE)", base.trim(reply, 800))
        pause_for_enter()
        return reply

    llm_mod.chat = traced_chat
    orch.chat = traced_chat
    tool_registry.dispatch_tool = traced_dispatch
    planner_mod.dispatch_tool = traced_dispatch

    if live_critic:
        critic_mod._call_anthropic_critic = live_critic_call
    else:
        critic_mod._call_anthropic_critic = scripted_critic
        critic_mod.is_available = lambda: True

    github_http.github_get_json = base._fake_github_get_json
    github_pr_lookup.github_get_json = base._fake_github_get_json
    github_issue_lookup.github_get_json = base._fake_github_get_json
    guardrails.ensure_public_github_repo = base._fake_repo_check


def run_scenario(
    title_text: str,
    query: ExplainerQuery,
    scripted_llm: base.ScriptedLLM,
    critic_text: str,
    *,
    live_critic: bool,
) -> None:
    global _step_counter, _tool_counter
    _step_counter = 0
    _tool_counter = 0

    base.title(title_text)
    base.bullet("repo", Path(query.repo_path).name)
    base.bullet("file", query.file_path or "<question mode>")
    base.bullet("lines", f"{query.start_line}-{query.end_line}")
    base.bullet("github", f"{query.owner}/{query.repo_name}")
    base.bullet("flags", "use_llm + use_planner + use_critic")
    base.bullet(
        "LLM mode",
        "scripted Planner+Synth, " +
        ("LIVE Anthropic Critic" if live_critic else "scripted Critic"),
    )

    install_tracing(scripted_llm, critic_text, live_critic=live_critic)
    start = time.time()
    result = GitExplainerAgent(use_llm=True, use_planner=True, use_critic=True).explain(query)
    elapsed = time.time() - start

    base.title("RESULT")
    base.bullet("elapsed", f"{elapsed:.2f} s")
    base.bullet("commits found", str(len(result["commits"])))
    base.bullet("PRs fetched", str(len(result["pull_requests"])))
    base.bullet("diffs gathered", str(len(result["diffs"])))
    base.bullet("planner.iters", str(result["planner"]["iterations_used"]) if result.get("planner") else "n/a")
    base.bullet("planner.halted", result["planner"]["halted_reason"] if result.get("planner") else "n/a")
    base.bullet("critic.verdict", result["critic"]["verdict"] if result.get("critic") else "n/a")
    base.bullet("critic.replanned", str(result["critic"].get("replanned", False)) if result.get("critic") else "n/a")
    base.bullet("used_fallback", str(result["used_fallback"]))

    base.section("FINAL EXPLANATION")
    for key in ("what_changed", "why", "tradeoffs", "limitations", "summary"):
        print(f"\n   ## {key}")
        body = result["explanation"].get(key, "") or "(empty)"
        for line in textwrap.wrap(body, width=WIDTH - 6):
            print(f"      {line}")


def classify_prompt(prompt: str) -> str:
    if "Explain why the selected code exists" in prompt[:600]:
        return "SYNTHESIZER"
    if "Decide the next action" in prompt:
        return "PLANNER"
    return "LLM"


def next_step() -> int:
    global _step_counter
    _step_counter += 1
    return _step_counter


def next_tool() -> int:
    global _tool_counter
    _tool_counter += 1
    return _tool_counter


def should_pause() -> bool:
    return _pause_between_steps and sys.stdin.isatty()


def pause_for_enter() -> None:
    if not should_pause():
        return
    print()
    print("[press Enter for next step]", flush=True)
    input()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Step-by-step walkthrough for the live-demo folder.

The scripted fixtures live beside this file, while the planner loop,
dispatcher, evidence merging, citation validation, and critic-control
flow run through production code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import demo_fixtures as demo


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
PREFERRED_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
sys.path.insert(0, str(PROJECT_ROOT))


def reexec_with_venv_if_available() -> None:
    if os.getenv("LIVE_DEMO_NO_REEXEC") == "1":
        return
    if not PREFERRED_PYTHON.exists():
        return
    venv_root = (PROJECT_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() == venv_root:
        return
    os.execv(
        str(PREFERRED_PYTHON),
        [str(PREFERRED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


reexec_with_venv_if_available()

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
_ORIGINAL_DISPATCH = planner_mod.dispatch_tool
_ORIGINAL_ANTHROPIC_CALL = critic_mod._call_anthropic_critic
_step_counter = 0
_tool_counter = 0
_pause_between_steps = True
_verbose_prompts = False


def main(argv: list[str] | None = None) -> int:
    global _pause_between_steps, _verbose_prompts

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
    parser.add_argument(
        "--verbose-prompts",
        action="store_true",
        help="Show fuller prompts, including system prompts, instead of compact excerpts.",
    )
    args = parser.parse_args(argv)
    _pause_between_steps = not args.no_pause
    _verbose_prompts = args.verbose_prompts

    print()
    demo.hr("#")
    print("  GIT HISTORY EXPLAINER -- LIVE CLASS DEMO")
    demo.hr("#")
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
    if _verbose_prompts:
        print()
        print("Verbose prompts: showing fuller prompt traffic for technical narration.")

    if args.scenario in ("1", "both"):
        llm, critic_text = demo.scripts_demo_1()
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
        llm, critic_text = demo.scripts_demo_2()
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

    demo.title("DEMO COMPLETE")
    return 0


def install_tracing(
    scripted_llm: demo.ScriptedLLM,
    critic_text: str,
    *,
    live_critic: bool,
) -> None:
    real_dispatch = _ORIGINAL_DISPATCH
    real_anthropic_call = _ORIGINAL_ANTHROPIC_CALL

    def traced_chat(prompt, *, system_prompt="", history=None,
                    model=None, max_tokens=None, temperature=0.3):
        kind = classify_prompt(prompt)
        step = next_step()
        demo.section(f"STEP {step}: {kind} -> Groq llama-3.1-8b-instant")
        show_prompt(kind, prompt, system_prompt=system_prompt)
        reply = scripted_llm.chat(prompt)
        demo.block(f"{kind} reply", demo.trim(reply, 700))
        pause_for_enter()
        return reply

    def traced_dispatch(name, arguments, context):
        tool_number = next_tool()
        demo.section(f"  TOOL CALL #{tool_number}: {name}({json.dumps(arguments)})")
        result = real_dispatch(name, arguments, context)
        demo.bullet("result", tool_registry._summarize_result(name, result))
        pause_for_enter()
        return result

    def scripted_critic(prompt: str) -> str:
        step = next_step()
        demo.section("STEP " + str(step) + ": CRITIC -> Anthropic Claude Haiku 4.5  (SCRIPTED)")
        show_prompt("CRITIC", prompt)
        demo.block("critic reply", demo.trim(critic_text, 800))
        pause_for_enter()
        return critic_text

    def live_critic_call(prompt: str) -> str:
        step = next_step()
        demo.section("STEP " + str(step) + ": CRITIC -> Anthropic Claude Haiku 4.5  (LIVE)")
        show_prompt("CRITIC", prompt)
        reply = real_anthropic_call(prompt)
        demo.block("critic reply (LIVE)", demo.trim(reply, 800))
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

    github_http.github_get_json = demo.fake_github_get_json
    github_pr_lookup.github_get_json = demo.fake_github_get_json
    github_issue_lookup.github_get_json = demo.fake_github_get_json
    guardrails.ensure_public_github_repo = demo.fake_repo_check


def run_scenario(
    title_text: str,
    query: ExplainerQuery,
    scripted_llm: demo.ScriptedLLM,
    critic_text: str,
    *,
    live_critic: bool,
) -> None:
    global _step_counter, _tool_counter
    _step_counter = 0
    _tool_counter = 0

    demo.title(title_text)
    demo.bullet("repo", Path(query.repo_path).name)
    demo.bullet("file", query.file_path or "<question mode>")
    demo.bullet("lines", f"{query.start_line}-{query.end_line}")
    demo.bullet("github", f"{query.owner}/{query.repo_name}")
    demo.bullet("flags", "use_llm + use_planner + use_critic")
    demo.bullet(
        "LLM mode",
        "scripted Planner+Synth, " +
        ("LIVE Anthropic Critic" if live_critic else "scripted Critic"),
    )

    install_tracing(scripted_llm, critic_text, live_critic=live_critic)
    start = time.time()
    result = GitExplainerAgent(use_llm=True, use_planner=True, use_critic=True).explain(query)
    elapsed = time.time() - start

    demo.title("RESULT")
    demo.bullet("elapsed", f"{elapsed:.2f} s")
    demo.bullet("commits found", str(len(result["commits"])))
    demo.bullet("PRs fetched", str(len(result["pull_requests"])))
    demo.bullet("diffs gathered", str(len(result["diffs"])))
    demo.bullet("planner.iters", str(result["planner"]["iterations_used"]) if result.get("planner") else "n/a")
    demo.bullet("planner.halted", result["planner"]["halted_reason"] if result.get("planner") else "n/a")
    demo.bullet("critic.verdict", result["critic"]["verdict"] if result.get("critic") else "n/a")
    demo.bullet("critic.replanned", str(result["critic"].get("replanned", False)) if result.get("critic") else "n/a")
    demo.bullet("used_fallback", str(result["used_fallback"]))

    demo.section("FINAL EXPLANATION")
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


def show_prompt(kind: str, prompt: str, *, system_prompt: str = "") -> None:
    if _verbose_prompts:
        if system_prompt:
            demo.block("system prompt", demo.trim(system_prompt, 2500))
        demo.block(f"{kind} prompt (verbose)", demo.trim(prompt, 5000))
        return
    demo.block(f"{kind} prompt (key fragment)", demo.summarize_prompt(prompt, kind))


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

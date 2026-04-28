#!/usr/bin/env python3
"""Step-by-step walkthrough for the live-demo folder.

The visible prompt/reply/tool-call formatting lives here, while the
planner, synthesizer, critic, dispatcher, evidence merging, and citation
validation run through production code.
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

import demo_fixtures as demo


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
PREFERRED_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
QUERY_FILE = DEMO_DIR / "sample_queries.json"
LIVE_DEMO_QUERY_ID = "agent-recovery-question"
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

from git_explainer import config
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
_ORIGINAL_CHAT = llm_mod.chat
_ORIGINAL_DISPATCH = planner_mod.dispatch_tool
_ORIGINAL_CRITIQUE = critic_mod.critique
_ORIGINAL_CRITIC_IS_AVAILABLE = critic_mod.is_available
_ORIGINAL_ANTHROPIC_CALL = critic_mod._call_anthropic_critic
_step_counter = 0
_tool_counter = 0
_pause_between_steps = True
_verbose_prompts = False
_synth_attempts = 0


def _provider_label() -> str:
    base = (os.getenv("GROQ_BASE_URL") or "").lower()
    if "anthropic.com" in base:
        return "Anthropic"
    if "groq.com" in base or not base:
        return "Groq"
    return "LLM"


def main(argv: list[str] | None = None) -> int:
    global _pause_between_steps, _verbose_prompts

    parser = argparse.ArgumentParser(description="Step-by-step Git Explainer walkthrough.")
    parser.add_argument(
        "--live-critic",
        action="store_true",
        help="Deprecated; the critic is already real when Anthropic is configured.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run straight through instead of waiting for Enter between steps.",
    )
    parser.add_argument(
        "--verbose-prompts",
        action="store_true",
        help="Show fuller non-planner/non-critic prompt traffic; planner evidence and critic prompts are always full.",
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
    provider = _provider_label()
    print(textwrap.fill(
        f"Demo mode: planner and synthesizer calls go to the real {provider} "
        "OpenAI-compatible endpoint. The critic uses the real Anthropic "
        "path when configured, otherwise production code marks it skipped. "
        "GitHub responses stay on presentation fixtures so the story is "
        "stable while the LLM responses are live.",
        width=WIDTH,
    ))
    print()
    print(textwrap.fill(
        f"Synthesizer model: {config.GROQ_MODEL}. Planner model: {config.PLANNER_MODEL}.",
        width=WIDTH,
    ))
    if should_pause():
        print()
        print("Interactive mode: press Enter after each step to continue.")
    if _verbose_prompts:
        print()
        print("Verbose prompts: showing fuller non-planner/non-critic prompt traffic.")

    query_preset = load_query_preset(LIVE_DEMO_QUERY_ID)
    run_scenario(
        "LIVE QUESTION: critic-guided recovery from thin evidence",
        query_from_preset(query_preset),
        user_query=query_preset["question"],
    )

    demo.title("DEMO COMPLETE")
    return 0


def install_tracing() -> None:
    real_chat = _ORIGINAL_CHAT
    real_dispatch = _ORIGINAL_DISPATCH
    real_critique = _ORIGINAL_CRITIQUE
    real_critic_is_available = _ORIGINAL_CRITIC_IS_AVAILABLE
    real_anthropic_call = _ORIGINAL_ANTHROPIC_CALL

    def traced_chat(prompt, *, system_prompt="", history=None,
                    model=None, max_tokens=None, temperature=0.3):
        global _synth_attempts
        kind = classify_prompt(prompt)
        step = next_step()
        resolved_model = model or config.GROQ_MODEL
        provider = _provider_label()
        suffix = ""
        if kind == "SYNTHESIZER":
            _synth_attempts += 1
            if _synth_attempts > 1:
                suffix = f"  [retry #{_synth_attempts - 1}: prior attempt failed JSON/citation validation]"
        demo.section(f"STEP {step}: {kind} -> {provider} {resolved_model}  (LIVE){suffix}")
        show_prompt(kind, prompt, system_prompt=system_prompt)
        try:
            kwargs = {
                "system_prompt": system_prompt,
                "history": history,
                "temperature": temperature,
            }
            if model is not None:
                kwargs["model"] = model
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            reply = real_chat(prompt, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- shown before production fallback
            show_reply(f"{kind} error", repr(exc))
            pause_for_enter()
            raise
        show_reply(f"{kind} reply (LIVE)", reply)
        pause_for_enter()
        return reply

    def traced_dispatch(name, arguments, context):
        tool_number = next_tool()
        demo.section(f"  TOOL CALL #{tool_number}: {name}({json.dumps(arguments)})")
        result = real_dispatch(name, arguments, context)
        demo.bullet("result", tool_registry._summarize_result(name, result))
        pause_for_enter()
        return result

    def traced_critic_chat(prompt: str) -> str:
        try:
            reply = real_anthropic_call(prompt)
        except Exception as exc:  # noqa: BLE001 -- critique() records this as skipped
            show_reply("critic error", repr(exc))
            raise
        show_reply("critic reply (LIVE)", reply)
        return reply

    def traced_critique(
        *,
        query_dict,
        explanation,
        evidence,
        chat_fn=None,
        is_available_fn=None,
    ):
        step = next_step()
        demo.section(f"STEP {step}: CRITIC -> Anthropic {config.CRITIC_MODEL}  (LIVE)")
        prompt = critic_mod._build_user_prompt(
            query_dict=query_dict,
            explanation=explanation,
            evidence=evidence,
        )
        show_prompt("CRITIC", prompt, system_prompt=critic_mod._SYSTEM_PROMPT)

        raw_reply_seen = False
        base_chat_fn = chat_fn or traced_critic_chat

        def traced_chat_for_critique(prompt_from_critic: str) -> str:
            nonlocal raw_reply_seen
            try:
                reply = base_chat_fn(prompt_from_critic)
            except Exception as exc:  # noqa: BLE001 -- critique() records this as skipped
                if base_chat_fn is not traced_critic_chat:
                    show_reply("critic error", repr(exc))
                raise
            raw_reply_seen = True
            if base_chat_fn is not traced_critic_chat:
                show_reply("critic reply", reply)
            return reply

        report = real_critique(
            query_dict=query_dict,
            explanation=explanation,
            evidence=evidence,
            chat_fn=traced_chat_for_critique,
            is_available_fn=is_available_fn or real_critic_is_available,
        )
        show_critic_response(
            "critic structured response" if raw_reply_seen else "critic response (skipped)",
            report,
        )
        pause_for_enter()
        return report

    llm_mod.chat = traced_chat
    orch.chat = traced_chat
    tool_registry.dispatch_tool = traced_dispatch
    planner_mod.dispatch_tool = traced_dispatch
    critic_mod._call_anthropic_critic = traced_critic_chat
    critic_mod.critique = traced_critique

    github_http.github_get_json = demo.fake_github_get_json
    github_pr_lookup.github_get_json = demo.fake_github_get_json
    github_issue_lookup.github_get_json = demo.fake_github_get_json
    guardrails.ensure_public_github_repo = demo.fake_repo_check


def run_scenario(
    title_text: str,
    query: ExplainerQuery,
    *,
    user_query: str,
) -> None:
    global _step_counter, _tool_counter
    _step_counter = 0
    _tool_counter = 0

    demo.title(title_text)
    demo.bullet("repo", Path(query.repo_path).name)
    if query.question:
        demo.bullet("mode", "natural-language question")
        demo.bullet("file hint", query.file_path or "<none>")
        demo.bullet("lines", "resolved at runtime")
    else:
        demo.bullet("mode", "line range")
        demo.bullet("file", query.file_path or "<question mode>")
        demo.bullet("lines", f"{query.start_line}-{query.end_line}")
    demo.bullet("github", f"{query.owner}/{query.repo_name}")
    demo.bullet("flags", "use_llm + use_planner + use_critic")
    demo.bullet("LLM mode", "LIVE Planner+Synth; LIVE Critic if configured")
    demo.block(
        "original user query",
        textwrap.fill(user_query, width=WIDTH - 8),
        indent="     ",
    )

    install_tracing()
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
    if result.get("critic"):
        show_critic_response("critic final response", result["critic"])

    demo.section("FINAL EXPLANATION")
    for key in ("what_changed", "why", "tradeoffs", "limitations", "summary"):
        print(f"\n   ## {key}")
        body = result["explanation"].get(key, "") or "(empty)"
        for line in textwrap.wrap(body, width=WIDTH - 6):
            print(f"      {line}")


def load_query_preset(query_id: str) -> dict[str, Any]:
    for query in json.loads(QUERY_FILE.read_text(encoding="utf-8")):
        if query["id"] == query_id:
            return query
    raise RuntimeError(f"Missing live-demo query preset: {query_id}")


def query_from_preset(query: dict[str, Any]) -> ExplainerQuery:
    return ExplainerQuery(
        repo_path=str(PROJECT_ROOT),
        file_path=query.get("file_path"),
        start_line=query.get("start_line"),
        end_line=query.get("end_line"),
        question=query.get("question"),
        owner=query["owner"],
        repo_name=query["repo_name"],
        max_commits=int(query.get("max_commits", 5)),
    )


def classify_prompt(prompt: str) -> str:
    if "Explain why the selected code exists" in prompt[:600]:
        return "SYNTHESIZER"
    if "Decide the next action" in prompt:
        return "PLANNER"
    return "LLM"


def show_prompt(kind: str, prompt: str, *, system_prompt: str = "") -> None:
    if kind == "PLANNER":
        demo.block(
            "PLANNER prompt (evidence section, full)",
            planner_evidence_view(prompt),
            color=demo.COLOR_PROMPT,
        )
        return

    if kind == "CRITIC":
        if system_prompt:
            demo.block(
                f"{kind} system prompt (full)",
                system_prompt,
                color=demo.COLOR_PROMPT,
            )
        demo.block(
            f"{kind} prompt (full)",
            prompt,
            color=demo.COLOR_PROMPT,
        )
        return

    if _verbose_prompts:
        if system_prompt:
            demo.block("system prompt", demo.trim(system_prompt, 2500), color=demo.COLOR_PROMPT)
        demo.block(
            f"{kind} prompt (verbose)",
            demo.trim(prompt, 5000),
            color=demo.COLOR_PROMPT,
        )
        return
    demo.block(
        f"{kind} prompt (key fragment)",
        demo.summarize_prompt(prompt, kind),
        color=demo.COLOR_PROMPT,
    )


def planner_evidence_view(prompt: str) -> str:
    bits: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Iteration "):
            bits.append(stripped)
            break

    evidence = prompt_section(
        prompt,
        "Evidence collected so far:",
        [
            "Tool call history (most recent last):",
            "Focus hints from the critic",
            "Decide the next action.",
        ],
    )
    if evidence:
        bits.append(evidence)

    focus_hints = prompt_section(
        prompt,
        "Focus hints from the critic",
        ["Decide the next action."],
    )
    if focus_hints:
        bits.append(focus_hints)

    return "\n\n".join(bits) if bits else demo.summarize_prompt(prompt, "PLANNER")


def prompt_section(prompt: str, start_marker: str, stop_markers: list[str]) -> str:
    start = prompt.find(start_marker)
    if start == -1:
        return ""

    stops = [
        stop
        for marker in stop_markers
        if (stop := prompt.find(marker, start + len(start_marker))) != -1
    ]
    end = min(stops) if stops else len(prompt)
    return prompt[start:end].rstrip()


def show_reply(label: str, reply: str) -> None:
    body = demo.pretty_json(reply)
    demo.block(label, body, color=demo.COLOR_REPLY)


def show_critic_response(label: str, report: Any) -> None:
    if hasattr(report, "to_dict"):
        payload = report.to_dict()
    elif isinstance(report, dict):
        payload = report
    else:
        payload = {"response": str(report)}
    show_reply(label, json.dumps(payload, indent=2, sort_keys=True))


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

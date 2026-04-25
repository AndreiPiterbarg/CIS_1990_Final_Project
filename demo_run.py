"""End-to-end instrumented demo.

Runs the agent twice with use_planner=True, use_critic=True against this
repository, printing every Planner prompt + reply, every tool dispatch +
result, the synthesizer prompt + reply, and the Critic prompt + verdict.

This is a one-off harness for a "does it actually work" demo, not part
of the production pipeline.
"""

from __future__ import annotations

import os
import json
import sys
import textwrap
from pathlib import Path

# Force-load .env before importing the agent (config.py also calls
# load_dotenv but the order matters when this file is run directly).
from dotenv import load_dotenv

load_dotenv()

# Daily token quota on the default 70B model is exhausted in this
# account, so the demo overrides the model env vars before importing
# config.py. The 8B Instant model has a separate per-model quota and
# is plenty for tool-routing JSON output.
os.environ.setdefault("GROQ_MODEL", "llama-3.1-8b-instant")
os.environ.setdefault("PLANNER_MODEL", "llama-3.1-8b-instant")

import git_explainer.critic as critic_mod
import git_explainer.llm as llm_mod
import git_explainer.tool_registry as tool_registry
from git_explainer.orchestrator import GitExplainerAgent
from git_explainer.guardrails import ExplainerQuery


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def hr(char: str = "-", width: int = 100) -> None:
    print(char * width)


def banner(text: str, *, char: str = "=") -> None:
    print()
    hr(char)
    print(f"  {text}")
    hr(char)


def trim(text: str, *, max_chars: int = 4000) -> str:
    """Trim a long string for display while keeping head + tail intact."""
    if len(text) <= max_chars:
        return text
    elided = len(text) - max_chars
    head = text[: max_chars * 2 // 3]
    tail = text[-(max_chars // 3):]
    return f"{head}\n\n...[{elided} chars elided]...\n\n{tail}"


def show_block(label: str, body: str, *, max_chars: int = 4000) -> None:
    print(f"\n+-- {label} --")
    for line in trim(body, max_chars=max_chars).splitlines():
        print(f"|  {line}")
    print("+--")


# ---------------------------------------------------------------------------
# Instrumentation: wrap LLM + dispatch entry points
# ---------------------------------------------------------------------------


_step = 0


def _next_step(kind: str) -> int:
    global _step
    _step += 1
    return _step


def _install_tracing() -> None:
    real_chat = llm_mod.chat
    real_dispatch = tool_registry.dispatch_tool
    real_critic_call = critic_mod._call_anthropic_critic

    def traced_chat(user_content, *, system_prompt="", history=None,
                    model=None, max_tokens=None, temperature=0.3):
        # Heuristic: the synthesizer prompt always starts with "Explain why";
        # everything else is a planner / condenser call. Differentiate them
        # in the log.
        if user_content.lstrip().startswith("Explain why"):
            kind = "SYNTHESIZER"
        elif "decide the next action" in user_content.lower():
            kind = "PLANNER"
        elif "summarize the following" in user_content.lower():
            kind = "EVIDENCE-CONDENSER"
        else:
            kind = "LLM"
        n = _next_step(kind)

        banner(f"STEP {n}: {kind} ->Groq ({model or llm_mod.config.GROQ_MODEL})")
        if system_prompt:
            show_block("system prompt", system_prompt, max_chars=1500)
        show_block("user prompt", user_content, max_chars=4000)

        # Forward kwargs only when set so we don't override defaults.
        kwargs = {"system_prompt": system_prompt, "temperature": temperature}
        if history is not None:
            kwargs["history"] = history
        if model is not None:
            kwargs["model"] = model
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            reply = real_chat(user_content, **kwargs)
        except Exception as exc:
            print(f"  [X] {kind} call FAILED: {type(exc).__name__}: {exc}")
            raise
        show_block(f"{kind} reply", reply, max_chars=2500)
        return reply

    def traced_dispatch(name, arguments, context):
        n = _next_step("TOOL")
        banner(f"STEP {n}: TOOL DISPATCH  ->{name}({json.dumps(arguments)})")
        try:
            result = real_dispatch(name, arguments, context)
        except Exception as exc:
            print(f"  [X] tool raised: {type(exc).__name__}: {exc}")
            raise
        # Compact result preview so logs stay readable.
        preview = json.dumps(result, default=str)[:1200]
        print(f"  [OK] result: {preview}{'...' if len(preview) >= 1200 else ''}")
        return result

    def traced_critic_call(prompt):
        n = _next_step("CRITIC")
        banner(f"STEP {n}: CRITIC ->Anthropic ({critic_mod.config.CRITIC_MODEL})")
        show_block("critic prompt", prompt, max_chars=4000)
        reply = real_critic_call(prompt)
        show_block("critic reply", reply, max_chars=2000)
        return reply

    llm_mod.chat = traced_chat
    tool_registry.dispatch_tool = traced_dispatch
    critic_mod._call_anthropic_critic = traced_critic_call

    # The orchestrator imported ``chat`` at module load time (``from
    # git_explainer.llm import chat``), so patching ``llm_mod.chat``
    # alone does not catch the synthesizer call. Patch the orchestrator's
    # local binding too.
    import git_explainer.orchestrator as orch
    orch.chat = traced_chat


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def run_demo(name: str, query: ExplainerQuery) -> None:
    global _step
    _step = 0

    banner(f"  DEMO: {name}", char="#")
    print(f"  repo:        {query.repo_path}")
    print(f"  file:        {query.file_path}")
    print(f"  line range:  {query.start_line}-{query.end_line}")
    print(f"  github:      {query.owner}/{query.repo_name}")
    print(f"  flags:       use_llm=True, use_planner=True, use_critic=True")

    agent = GitExplainerAgent(
        use_llm=True, use_planner=True, use_critic=True,
    )
    result = agent.explain(query)

    banner("FINAL RESULT", char="=")
    print(f"  used_fallback:    {result['used_fallback']}")
    print(f"  fallback_reason:  {result.get('fallback_reason')}")
    print(f"  commits found:    {len(result['commits'])}")
    print(f"  PRs fetched:      {len(result['pull_requests'])}")
    print(f"  issues fetched:   {len(result['issues'])}")
    print(f"  diffs:            {len(result['diffs'])}")
    print(f"  cache stats:      {result['cache_stats']}")
    if result.get("planner"):
        p = result["planner"]
        print(f"  planner.iters:    {p.get('iterations_used')}")
        print(f"  planner.halted:   {p.get('halted_reason')}")
        print(f"  planner.fallback: {p.get('fell_back_to_fixed_sequence')}")
    if result.get("critic"):
        c = result["critic"]
        print(f"  critic.verdict:   {c.get('verdict')}")
        print(f"  critic.replanned: {c.get('replanned', False)}")
        if c.get("issues"):
            print(f"  critic.issues:    {c['issues']}")
        if c.get("focus_hints"):
            print(f"  critic.hints:     {c['focus_hints']}")

    banner("EXPLANATION", char="-")
    for section in ("what_changed", "why", "tradeoffs", "limitations", "summary"):
        print(f"\n  >{section}:")
        text = result["explanation"].get(section, "")
        for line in textwrap.wrap(text, width=96):
            print(f"    {line}")


def main() -> None:
    if not llm_mod.is_available():
        print("ERROR: GROQ_API_KEY missing -- planner cannot run for real.")
        sys.exit(1)
    if not critic_mod.is_available():
        print(
            "WARN: Anthropic key missing -- critic will return verdict='skipped'.",
            file=sys.stderr,
        )

    _install_tracing()

    repo = str(Path(__file__).parent.resolve())

    # Demo 1 -- a line range we know has linked PR #1 (config switch).
    demo1 = ExplainerQuery(
        repo_path=repo,
        file_path="git_explainer/config.py",
        start_line=13,
        end_line=19,
        owner="AndreiPiterbarg",
        repo_name="CIS_1990_Final_Project",
        max_commits=5,
        enforce_public_repo=True,
    )

    # Demo 2 -- a line range with multiple commits but no PR (file context fix).
    demo2 = ExplainerQuery(
        repo_path=repo,
        file_path="git_explainer/tools/file_context_reader.py",
        start_line=59,
        end_line=70,
        owner="AndreiPiterbarg",
        repo_name="CIS_1990_Final_Project",
        max_commits=5,
        enforce_public_repo=True,
    )

    run_demo("Config switch (Kimi -> Groq) - expects PR #1", demo1)
    run_demo("file_context_reader._read_from_revision bug-fix history", demo2)


if __name__ == "__main__":
    main()

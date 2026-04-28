#!/usr/bin/env python3
"""Presentation runner for the Git Explainer Agent.

All generated files are kept inside live-demo/.cache or live-demo/logs.
The existing project files are imported or executed, but not edited.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
CACHE_DIR = DEMO_DIR / ".cache"
LOG_DIR = DEMO_DIR / "logs"
QUERY_FILE = DEMO_DIR / "sample_queries.json"
PREFERRED_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
WRAP = 96


def main(argv: list[str] | None = None) -> int:
    reexec_with_venv_if_available()

    parser = argparse.ArgumentParser(
        description="Run presentation-friendly demos for the Git Explainer Agent."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Check demo prerequisites and paths.")
    sub.add_parser("list", help="List curated query presets.")

    scripted = sub.add_parser(
        "live",
        aliases=["scripted"],
        help="Run the step-by-step live planner/synthesizer/critic demo.",
    )
    scripted.add_argument(
        "--live-critic",
        action="store_true",
        help="Deprecated; the critic is already real when Anthropic is configured.",
    )
    scripted.add_argument(
        "--no-pause",
        action="store_true",
        help="Run straight through instead of pausing for Enter between steps.",
    )
    scripted.add_argument(
        "--verbose-prompts",
        action="store_true",
        help="Show fuller non-planner/non-critic prompt traffic; planner evidence and critic prompts are always full.",
    )

    safe = sub.add_parser(
        "safe",
        help="Run a deterministic no-network, no-LLM query preset.",
    )
    safe.add_argument("query_id", help="Preset from sample_queries.json.")
    safe.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON result after the presenter summary.",
    )

    agentic = sub.add_parser(
        "agentic",
        help="Run the real CLI with planner/critic enabled. Requires live services for the full effect.",
    )
    agentic.add_argument("query_id", help="Line-range preset from sample_queries.json.")
    agentic.add_argument(
        "--skip-public-check",
        action="store_true",
        help="Skip the public-repo preflight check but still allow GitHub enrichment.",
    )
    agentic.add_argument(
        "--show-command-only",
        action="store_true",
        help="Print the command that would run, then exit.",
    )

    args = parser.parse_args(argv)

    if args.command == "check":
        return check()
    if args.command == "list":
        return list_queries()
    if args.command in ("scripted", "live"):
        return stepwise_live_demo(
            live_critic=args.live_critic,
            pause=not args.no_pause,
            verbose_prompts=args.verbose_prompts,
        )
    if args.command == "safe":
        return safe_query(args.query_id, show_json=args.json)
    if args.command == "agentic":
        return agentic_query(
            args.query_id,
            skip_public_check=args.skip_public_check,
            show_command_only=args.show_command_only,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


def demo_env(cache_name: str = "git_explainer_cache.json") -> dict[str, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv_if_available()
    env = os.environ.copy()
    env["GIT_EXPLAINER_CACHE_FILENAME"] = str(CACHE_DIR / cache_name)
    env.setdefault("GROQ_MODEL", "openai/gpt-oss-120b")
    env.setdefault("PLANNER_MODEL", env["GROQ_MODEL"])
    env.setdefault("GITHUB_RATE_LIMIT_SLEEP_CAP", "5")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def check() -> int:
    print_header("Demo Setup Check")
    required = [
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "git_explainer" / "orchestrator.py",
        DEMO_DIR / "demo_fixtures.py",
        DEMO_DIR / "agent_walkthrough.py",
        QUERY_FILE,
    ]
    ok = True
    for path in required:
        exists = path.exists()
        ok = ok and exists
        print_status("found" if exists else "missing", path)

    env = demo_env()
    print()
    print(f"project root: {PROJECT_ROOT}")
    print(f"python:       {sys.executable}")
    print(f"demo cache:   {env['GIT_EXPLAINER_CACHE_FILENAME']}")
    print(f"demo logs:    {LOG_DIR}")
    print(f"groq model:   {env['GROQ_MODEL']}")
    print(f"planner:      {env['PLANNER_MODEL']}")
    print()
    print_key_status("GROQ_API_KEY")
    print_key_status("ANTHROPIC_API_KEY", fallback="ANTHROPIC_KEY")
    print_key_status("GITHUB_TOKEN")

    print()
    if ok:
        print("Ready. Suggested command:")
        print("  python3 live-demo/run_demo.py live")
        return 0
    print("Setup check failed. Fix missing files before presenting.")
    return 1


def list_queries() -> int:
    print_header("Query Presets")
    for query in load_queries():
        print(f"{query['id']}: {query['title']}")
        print(wrap(f"  {query['why_good']}", width=WRAP))
        if query.get("kind") == "question":
            print(wrap(f"  prompt: {query['question']}", width=WRAP))
        else:
            span = f"{query['file_path']}:{query['start_line']}-{query['end_line']}"
            print(f"  span:   {span}")
            print(wrap(f"  prompt: {query['natural_prompt']}", width=WRAP))
        print()
    return 0


def stepwise_live_demo(
    *,
    live_critic: bool,
    pause: bool,
    verbose_prompts: bool,
) -> int:
    args = [sys.executable, "live-demo/agent_walkthrough.py"]
    label = "live-question"
    if live_critic:
        args.append("--live-critic")
        label += "-live-critic"
    if not pause:
        args.append("--no-pause")
        label += "-no-pause"
    if verbose_prompts:
        args.append("--verbose-prompts")
        label += "-verbose"
    env = demo_env(cache_name=f"{timestamp()}-{label}.cache.json")
    return tee_subprocess(args, label=label, env=env)


def safe_query(query_id: str, *, show_json: bool) -> int:
    query = get_query(query_id)
    env = demo_env(cache_name=f"{timestamp()}-safe-{query_id}.cache.json")
    os.environ.update(
        {
            "GIT_EXPLAINER_CACHE_FILENAME": env["GIT_EXPLAINER_CACHE_FILENAME"],
            "GROQ_MODEL": env["GROQ_MODEL"],
            "PLANNER_MODEL": env["PLANNER_MODEL"],
            "GITHUB_RATE_LIMIT_SLEEP_CAP": env["GITHUB_RATE_LIMIT_SLEEP_CAP"],
        }
    )
    sys.path.insert(0, str(PROJECT_ROOT))

    # Keep the safe mode truly local even when the repository has a GitHub origin.
    from git_explainer import config
    import git_explainer.guardrails as guardrails
    from git_explainer.guardrails import ExplainerQuery
    from git_explainer.orchestrator import GitExplainerAgent

    config.CACHE_FILENAME = env["GIT_EXPLAINER_CACHE_FILENAME"]
    guardrails.infer_github_repo = lambda repo_path: None

    explainer_query = ExplainerQuery(
        repo_path=str(PROJECT_ROOT),
        file_path=query.get("file_path"),
        start_line=query.get("start_line"),
        end_line=query.get("end_line"),
        question=query.get("question"),
        owner=None,
        repo_name=None,
        max_commits=int(query.get("max_commits", 5)),
        enforce_public_repo=False,
    )

    print_header(f"Safe Demo: {query['title']}")
    if query.get("kind") == "question":
        print(wrap(f"Question: {query['question']}"))
    else:
        print(f"Target: {query['file_path']}:{query['start_line']}-{query['end_line']}")
    print("Mode: deterministic fallback, local git only, no GitHub calls, no LLM calls")
    print()

    agent = GitExplainerAgent(use_llm=False, use_planner=False, use_critic=False)
    start = time.time()
    result = agent.explain(explainer_query)
    elapsed = time.time() - start

    print_result_summary(result, elapsed=elapsed)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / f"{timestamp()}-safe-{query_id}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print()
    print(f"saved JSON: {output_path}")

    if show_json:
        print()
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def agentic_query(
    query_id: str,
    *,
    skip_public_check: bool,
    show_command_only: bool,
) -> int:
    query = get_query(query_id)
    if query.get("kind") == "question":
        print("agentic mode currently expects a line-range preset. Use safe mode for question presets.")
        return 2

    args = [
        sys.executable,
        "main.py",
        str(PROJECT_ROOT),
        query["file_path"],
        str(query["start_line"]),
        str(query["end_line"]),
        "--max-commits",
        str(query.get("max_commits", 5)),
        "--owner",
        query["owner"],
        "--repo-name",
        query["repo_name"],
        "--planner",
        "--critic",
    ]
    if skip_public_check:
        args.append("--allow-private-repo")

    print_header(f"Agentic Live CLI: {query['title']}")
    print("This path uses real services when keys/network are available.")
    print_shell(args)
    if show_command_only:
        return 0
    env = demo_env(cache_name=f"{timestamp()}-agentic-{query_id}.cache.json")
    return tee_subprocess(args, label=f"agentic-{query_id}", env=env)


def print_result_summary(result: dict[str, Any], *, elapsed: float) -> None:
    print(f"elapsed:          {elapsed:.2f}s")
    print(f"used_fallback:    {result.get('used_fallback')}")
    print(f"fallback_reason:  {result.get('fallback_reason')}")
    print(f"commits:          {len(result.get('commits', []))}")
    print(f"pull_requests:    {len(result.get('pull_requests', []))}")
    print(f"issues:           {len(result.get('issues', []))}")
    print(f"diffs:            {len(result.get('diffs', []))}")
    if result.get("resolved_target"):
        target = result["resolved_target"]
        print()
        print("resolved target:")
        print(f"  {target['file_path']}:{target['start_line']}-{target['end_line']}")
        print(f"  matched terms: {', '.join(target.get('matched_terms', [])) or '(none)'}")

    print()
    print("explanation:")
    explanation = result.get("explanation", {})
    for key in ("what_changed", "why", "tradeoffs", "limitations", "summary"):
        body = explanation.get(key) or "(empty)"
        print(f"\n{key}:")
        print(wrap(body, prefix="  "))

    commits = result.get("commits", [])
    if commits:
        print()
        print("commits found:")
        for commit in commits:
            print(f"  {commit.get('sha')}  {commit.get('date')}  {commit.get('message')}")


def tee_subprocess(args: list[str], *, label: str, env: dict[str, str]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{timestamp()}-{label}.log"

    print_shell(args)
    print(f"log: {log_path}")
    print()

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            args,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = proc.wait()

    print()
    print(f"exit code: {return_code}")
    print(f"log saved: {log_path}")
    return return_code


def load_queries() -> list[dict[str, Any]]:
    return json.loads(QUERY_FILE.read_text(encoding="utf-8"))


def get_query(query_id: str) -> dict[str, Any]:
    for query in load_queries():
        if query["id"] == query_id:
            return query
    available = ", ".join(query["id"] for query in load_queries())
    raise SystemExit(f"Unknown query id {query_id!r}. Available: {available}")


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def print_header(text: str) -> None:
    print()
    print("=" * WRAP)
    print(text)
    print("=" * WRAP)


def print_shell(args: list[str]) -> None:
    display = [python_label() if part == sys.executable else part for part in args]
    print("$ " + shlex.join(display))


def reexec_with_venv_if_available() -> None:
    """Prefer the project virtualenv so demo commands find dependencies."""
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


def python_label() -> str:
    try:
        return str(Path(sys.executable).relative_to(PROJECT_ROOT))
    except ValueError:
        return sys.executable


def print_status(status: str, path: Path) -> None:
    print(f"{status:>7}: {path}")


def print_key_status(name: str, *, fallback: str | None = None) -> None:
    value = os.getenv(name)
    label = name
    if not value and fallback:
        value = os.getenv(fallback)
        label = f"{name}/{fallback}"
    print(f"{label:<32} {'set' if value else 'missing'}")


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def wrap(text: str, *, width: int = WRAP, prefix: str = "") -> str:
    return textwrap.fill(
        text,
        width=width,
        initial_indent=prefix,
        subsequent_indent=prefix,
        break_long_words=False,
        break_on_hyphens=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())

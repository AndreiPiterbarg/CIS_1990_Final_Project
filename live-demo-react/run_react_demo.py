#!/usr/bin/env python3
"""Live demo runner for using Git Explainer with facebook/react.

This folder is intentionally independent from live-demo/. Generated cache,
logs, and the optional React clone stay under live-demo-react/.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
DEFAULT_REPO_DIR = DEMO_DIR / "repos" / "react"
CACHE_DIR = DEMO_DIR / ".cache"
LOG_DIR = DEMO_DIR / "logs"
QUERY_FILE = DEMO_DIR / "sample_queries.json"
PREFERRED_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
REACT_REMOTE = "https://github.com/facebook/react.git"
REACT_OWNER = "facebook"
REACT_REPO = "react"
WRAP = 96


def main(argv: list[str] | None = None) -> int:
    reexec_with_venv_if_available()

    parser = argparse.ArgumentParser(
        description="Run Git Explainer demos against the public facebook/react repository."
    )
    parser.add_argument(
        "--repo-path",
        default=str(DEFAULT_REPO_DIR),
        help="Path to a local React checkout. Defaults to live-demo-react/repos/react.",
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="React ref to checkout during prepare. Defaults to main.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser("check", help="Check demo prerequisites and paths.")
    check_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if the React checkout is missing or presets cannot resolve.",
    )

    sub.add_parser("prepare", help="Clone or update the public React repository.")
    sub.add_parser("list", help="List curated React query presets.")

    command_parser = sub.add_parser(
        "command",
        help="Print the main.py command for a preset without running it.",
    )
    command_parser.add_argument("query_id", help="Preset from sample_queries.json.")
    command_parser.add_argument(
        "--planner",
        action="store_true",
        help="Include --planner --critic in the printed command.",
    )

    no_llm_parser = sub.add_parser(
        "no-llm",
        aliases=["safe"],
        help="Run the public React demo without LLM synthesis.",
    )
    no_llm_parser.add_argument("query_id", help="Preset from sample_queries.json.")

    live_parser = sub.add_parser(
        "live",
        help="Run the public React demo with planner and critic enabled.",
    )
    live_parser.add_argument("query_id", help="Preset from sample_queries.json.")
    live_parser.add_argument(
        "--show-command-only",
        action="store_true",
        help="Print the resolved command, then exit.",
    )

    args = parser.parse_args(argv)
    repo_path = Path(args.repo_path).expanduser().resolve()

    if args.command == "check":
        return check(repo_path, strict=args.strict)
    if args.command == "prepare":
        return prepare_repo(repo_path, ref=args.ref)
    if args.command == "list":
        return list_queries(repo_path)
    if args.command == "command":
        command = build_main_command(
            repo_path,
            get_query(args.query_id),
            no_llm=not args.planner,
            planner=args.planner,
        )
        print_shell(command)
        return 0
    if args.command in ("no-llm", "safe"):
        return run_query(repo_path, get_query(args.query_id), no_llm=True, planner=False)
    if args.command == "live":
        return run_query(
            repo_path,
            get_query(args.query_id),
            no_llm=False,
            planner=True,
            show_command_only=args.show_command_only,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


def prepare_repo(repo_path: Path, *, ref: str) -> int:
    print_header("Prepare React Checkout")
    repo_parent = repo_path.parent
    repo_parent.mkdir(parents=True, exist_ok=True)

    if (repo_path / ".git").exists():
        print(f"using existing checkout: {repo_path}")
        run_logged(["git", "remote", "set-url", "origin", REACT_REMOTE], cwd=repo_path)
        run_logged(["git", "fetch", "--prune", "--filter=blob:none", "origin"], cwd=repo_path)
    else:
        if repo_path.exists() and any(repo_path.iterdir()):
            print(f"target exists and is not an empty git checkout: {repo_path}")
            return 1
        clone_args = [
            "git",
            "clone",
            "--filter=blob:none",
            REACT_REMOTE,
            str(repo_path),
        ]
        run_logged(clone_args, cwd=PROJECT_ROOT)

    run_logged(["git", "checkout", ref], cwd=repo_path)
    run_logged(["git", "rev-parse", "--short", "HEAD"], cwd=repo_path)
    print()
    print("React checkout is ready.")
    return 0


def check(repo_path: Path, *, strict: bool) -> int:
    print_header("React Demo Setup Check")
    ok = True
    required = [
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "git_explainer",
        QUERY_FILE,
    ]
    for path in required:
        exists = path.exists()
        ok = ok and exists
        print_status("found" if exists else "missing", path)

    git_path = shutil.which("git")
    git_ok = git_path is not None
    ok = ok and git_ok
    print_status("found" if git_ok else "missing", Path(git_path or "git"))

    repo_ok = (repo_path / ".git").exists()
    print_status("found" if repo_ok else "missing", repo_path)

    env = demo_env()
    print()
    print(f"project root: {PROJECT_ROOT}")
    print(f"react repo:   {repo_path}")
    print(f"python:       {sys.executable}")
    print(f"demo cache:   {env['GIT_EXPLAINER_CACHE_FILENAME']}")
    print(f"demo logs:    {LOG_DIR}")
    print(f"groq model:   {env.get('GROQ_MODEL', '(unset)')}")
    print(f"planner:      {env.get('PLANNER_MODEL', '(unset)')}")
    print()
    print_key_status("GROQ_API_KEY")
    print_key_status("ANTHROPIC_API_KEY", fallback="ANTHROPIC_KEY")
    print_key_status("GITHUB_TOKEN")

    if repo_ok:
        print()
        print("preset spans:")
        for query in load_queries():
            try:
                span = resolve_span(repo_path, query)
            except Exception as exc:  # noqa: BLE001 -- check should report all presets.
                ok = False
                print(f"  {query['id']}: unresolved ({exc})")
            else:
                print(
                    f"  {query['id']}: "
                    f"{span['file_path']}:{span['start_line']}-{span['end_line']}"
                )
    else:
        print()
        print("React checkout is missing. Suggested next command:")
        print("  python3 live-demo-react/run_react_demo.py prepare")

    print()
    if ok and repo_ok:
        print("Ready. Suggested no-LLM smoke test:")
        print("  python3 live-demo-react/run_react_demo.py no-llm hooks-use-transition")
        print("Suggested live run:")
        print("  python3 live-demo-react/run_react_demo.py live hooks-use-transition")
    elif strict:
        return 1
    return 0


def list_queries(repo_path: Path) -> int:
    print_header("React Query Presets")
    repo_ok = (repo_path / ".git").exists()
    for query in load_queries():
        print(f"{query['id']}: {query['title']}")
        print(wrap(f"  {query['why_good']}", width=WRAP))
        print(wrap(f"  prompt: {query['natural_prompt']}", width=WRAP))
        print(f"  file:   {query['file_path']}")
        print(f"  anchor: {query['anchor']}")
        if repo_ok:
            try:
                span = resolve_span(repo_path, query)
            except Exception as exc:  # noqa: BLE001 -- listing should keep going.
                print(f"  span:   unresolved ({exc})")
            else:
                print(f"  span:   {span['file_path']}:{span['start_line']}-{span['end_line']}")
        print()
    return 0


def run_query(
    repo_path: Path,
    query: dict[str, Any],
    *,
    no_llm: bool,
    planner: bool,
    show_command_only: bool = False,
) -> int:
    ensure_repo_ready(repo_path)
    span = resolve_span(repo_path, query)
    label_mode = "no-llm" if no_llm else "live"
    print_header(f"React {label_mode} demo: {query['title']}")
    print(wrap(f"Question: {query['natural_prompt']}"))
    print(f"Target:   {span['file_path']}:{span['start_line']}-{span['end_line']}")
    print(f"Repo:     {REACT_OWNER}/{REACT_REPO} at {repo_path}")
    print()

    command = build_main_command(
        repo_path,
        query,
        resolved_span=span,
        no_llm=no_llm,
        planner=planner,
    )
    print_shell(command)
    if show_command_only:
        return 0

    env = demo_env(cache_name=f"{timestamp()}-{label_mode}-{query['id']}.cache.json")
    return tee_subprocess(command, label=f"{label_mode}-{query['id']}", env=env)


def build_main_command(
    repo_path: Path,
    query: dict[str, Any],
    *,
    resolved_span: dict[str, Any] | None = None,
    no_llm: bool,
    planner: bool,
) -> list[str]:
    ensure_repo_ready(repo_path)
    span = resolved_span or resolve_span(repo_path, query)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        str(repo_path),
        span["file_path"],
        str(span["start_line"]),
        str(span["end_line"]),
        "--owner",
        REACT_OWNER,
        "--repo-name",
        REACT_REPO,
        "--max-commits",
        str(query.get("max_commits", 5)),
    ]
    if no_llm:
        command.append("--no-llm")
    if planner:
        command.extend(["--planner", "--critic"])
    return command


def resolve_span(repo_path: Path, query: dict[str, Any]) -> dict[str, Any]:
    file_path = query["file_path"]
    target = repo_path / file_path
    if not target.exists():
        raise FileNotFoundError(f"missing file in React checkout: {file_path}")

    anchor = query["anchor"]
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    anchor_index = None
    for index, line in enumerate(lines, start=1):
        if anchor in line:
            anchor_index = index
            break
    if anchor_index is None:
        raise ValueError(f"anchor not found in {file_path}: {anchor!r}")

    start_line = max(1, anchor_index - int(query.get("span_before", 0)))
    end_line = min(len(lines), anchor_index + int(query.get("span_after", 0)))
    if end_line < start_line:
        end_line = start_line

    return {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "anchor_line": anchor_index,
    }


def ensure_repo_ready(repo_path: Path) -> None:
    if not (repo_path / ".git").exists():
        raise SystemExit(
            "React checkout is missing. Run:\n"
            "  python3 live-demo-react/run_react_demo.py prepare\n"
            "or pass --repo-path /path/to/react."
        )


def demo_env(cache_name: str = "git_explainer_cache.json") -> dict[str, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv_if_available()
    env = os.environ.copy()
    env["GIT_EXPLAINER_CACHE_FILENAME"] = str(CACHE_DIR / cache_name)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY")
    if anthropic_key and not os.getenv("LIVE_DEMO_FORCE_GROQ"):
        env["GROQ_API_KEY"] = anthropic_key
        env["GROQ_BASE_URL"] = "https://api.anthropic.com/v1/"
        env.setdefault("GROQ_MODEL", "claude-haiku-4-5")
        env.setdefault("PLANNER_MODEL", env["GROQ_MODEL"])
        env.setdefault("GROQ_MAX_TOKENS", "16384")
    else:
        env.setdefault("GROQ_MODEL", "openai/gpt-oss-120b")
        env.setdefault("PLANNER_MODEL", env["GROQ_MODEL"])

    env.setdefault("GITHUB_RATE_LIMIT_SLEEP_CAP", "5")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def tee_subprocess(args: list[str], *, label: str, env: dict[str, str]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{timestamp()}-{label}.log"
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


def run_logged(args: list[str], *, cwd: Path) -> None:
    print_shell(args)
    subprocess.run(args, cwd=cwd, check=True)


def load_queries() -> list[dict[str, Any]]:
    return json.loads(QUERY_FILE.read_text(encoding="utf-8"))


def get_query(query_id: str) -> dict[str, Any]:
    for query in load_queries():
        if query["id"] == query_id:
            return query
    available = ", ".join(query["id"] for query in load_queries())
    raise SystemExit(f"Unknown query id {query_id!r}. Available: {available}")


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


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


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

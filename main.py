"""CLI entrypoint for the Git explainer agent."""

from __future__ import annotations

import argparse
import json

from git_explainer.orchestrator import explain_code_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain why a range of code exists.")
    parser.add_argument("repo_path", help="Path to the local git repository")
    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to the target file, relative to the repo root. Optional in question mode.",
    )
    parser.add_argument("start_line", nargs="?", type=int, help="1-indexed start line")
    parser.add_argument("end_line", nargs="?", type=int, help="1-indexed end line")
    parser.add_argument(
        "--question",
        help="Natural-language question to resolve to a relevant code span before tracing history",
    )
    parser.add_argument("--owner", help="GitHub repository owner")
    parser.add_argument("--repo-name", help="GitHub repository name")
    parser.add_argument("--max-commits", type=int, default=5, help="Maximum commits to trace")
    parser.add_argument(
        "--context-radius",
        type=int,
        default=30,
        help="Lines of surrounding code to fetch when extra context is needed",
    )
    parser.add_argument(
        "--allow-private-repo",
        dest="enforce_public_repo",
        action="store_false",
        help=(
            "Opt out of the default public-repo guardrail so private or "
            "unreachable GitHub repositories are accepted. The public-repo "
            "check is now on by default."
        ),
    )
    parser.set_defaults(enforce_public_repo=True)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the synthesis model and use the deterministic fallback summary",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.question:
        if args.start_line is not None or args.end_line is not None:
            parser.error("question mode does not accept start_line or end_line")
    elif args.file_path is None or args.start_line is None or args.end_line is None:
        parser.error("Provide file_path start_line end_line, or use --question")

    result = explain_code_history(
        args.repo_path,
        args.file_path,
        args.start_line,
        args.end_line,
        question=args.question,
        owner=args.owner,
        repo_name=args.repo_name,
        max_commits=args.max_commits,
        context_radius=args.context_radius,
        enforce_public_repo=args.enforce_public_repo,
        use_llm=not args.no_llm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

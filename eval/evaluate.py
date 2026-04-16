"""Evaluation harness for the Git Explainer agent.

Loads benchmark cases from benchmark.json, runs the agent on each,
scores outputs against expected fields, and produces a summary report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from git_explainer.orchestrator import ExplanationResult, explain_code_history


@dataclass
class BenchmarkCase:
    """A single evaluation case loaded from benchmark.json."""

    id: str
    description: str
    repo_url: str
    owner: str | None
    repo_name: str | None
    file_path: str | None
    start_line: int | None
    end_line: int | None
    question: str | None
    max_commits: int
    use_llm: bool
    expected: dict[str, Any]
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCase:
        """Construct a BenchmarkCase from a raw JSON dict."""
        return cls(
            id=data["id"],
            description=data["description"],
            repo_url=data["repo_url"],
            owner=data.get("owner"),
            repo_name=data.get("repo_name"),
            file_path=data.get("file_path"),
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
            question=data.get("question"),
            max_commits=data["max_commits"],
            use_llm=data["use_llm"],
            expected=data.get("expected", {}),
            tags=data.get("tags", []),
        )


@dataclass
class CaseScore:
    """Result of scoring a single benchmark case."""

    case_id: str
    passed: bool
    checks: dict[str, bool]
    elapsed_seconds: float
    error: str | None = None


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    """Load and parse benchmark cases from a JSON file."""
    if not path.exists():
        print(f"Benchmark file not found: {path}")
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        print(f"Benchmark file is empty: {path}")
        sys.exit(1)

    return [BenchmarkCase.from_dict(case) for case in data]


def _is_local_repo(repo_url: str) -> bool:
    """Return True if the repo_url points to the current project."""
    cwd = Path.cwd().resolve()
    try:
        candidate = Path(repo_url).resolve()
        return candidate == cwd
    except (OSError, ValueError):
        return False


def setup_repos(cases: list[BenchmarkCase]) -> dict[str, str]:
    """Clone (or reuse) repos for all unique repo_urls. Returns url -> local path."""
    repo_map: dict[str, str] = {}
    for case in cases:
        url = case.repo_url
        if url in repo_map:
            continue
        if _is_local_repo(url):
            repo_map[url] = str(Path.cwd())
        else:
            dest = tempfile.mkdtemp(prefix="eval_repo_")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "50", url, dest],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                repo_map[url] = dest
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: Failed to clone {url}: {exc.stderr.strip()}")
                repo_map[url] = ""  # empty string signals clone failure
    return repo_map


def score_case(case: BenchmarkCase, result: ExplanationResult, elapsed: float) -> CaseScore:
    """Score an agent result against expected values from the benchmark case."""
    expected = case.expected
    checks: dict[str, bool] = {}

    if "min_commits" in expected:
        checks["min_commits"] = len(result["commits"]) >= expected["min_commits"]

    if "max_commits" in expected:
        checks["max_commits"] = len(result["commits"]) <= expected["max_commits"]

    if "commit_message_contains" in expected:
        messages = [c["message"].lower() for c in result["commits"]]
        checks["commit_message_contains"] = all(
            any(kw.lower() in msg for msg in messages)
            for kw in expected["commit_message_contains"]
        )

    if "pr_numbers" in expected:
        found_prs = {pr["number"] for pr in result["pull_requests"]}
        checks["pr_numbers"] = all(n in found_prs for n in expected["pr_numbers"])

    if "issue_numbers" in expected:
        found_issues = {issue["number"] for issue in result["issues"]}
        checks["issue_numbers"] = all(n in found_issues for n in expected["issue_numbers"])

    if "explanation_contains" in expected:
        sections = result["explanation"]
        all_text = " ".join([
            sections["what_changed"],
            sections["why"],
            sections["tradeoffs"],
            sections["limitations"],
            sections["summary"],
        ]).lower()
        checks["explanation_contains"] = all(
            phrase.lower() in all_text
            for phrase in expected["explanation_contains"]
        )

    if "used_fallback" in expected:
        checks["used_fallback"] = result["used_fallback"] == expected["used_fallback"]

    resolved = result.get("resolved_target") or {}
    if "resolved_file_path" in expected:
        checks["resolved_file_path"] = resolved.get("file_path") == expected["resolved_file_path"]

    if "resolved_matched_terms" in expected:
        found_terms = set(resolved.get("matched_terms", []))
        checks["resolved_matched_terms"] = all(
            term in found_terms for term in expected["resolved_matched_terms"]
        )

    if "resolved_preview_contains" in expected:
        preview = str(resolved.get("preview", "")).lower()
        checks["resolved_preview_contains"] = all(
            phrase.lower() in preview
            for phrase in expected["resolved_preview_contains"]
        )

    passed = all(checks.values()) if checks else True
    return CaseScore(
        case_id=case.id,
        passed=passed,
        checks=checks,
        elapsed_seconds=round(elapsed, 3),
    )


def run_case(
    case: BenchmarkCase,
    repo_path: str,
    *,
    no_llm: bool = False,
) -> CaseScore:
    """Run a single benchmark case and return its score."""
    if not repo_path:
        return CaseScore(
            case_id=case.id,
            passed=False,
            checks={},
            elapsed_seconds=0.0,
            error="Repository clone failed",
        )

    use_llm = case.use_llm and not no_llm
    t0 = time.monotonic()
    try:
        result = explain_code_history(
            repo_path=repo_path,
            file_path=case.file_path,
            start_line=case.start_line,
            end_line=case.end_line,
            question=case.question,
            owner=case.owner,
            repo_name=case.repo_name,
            max_commits=case.max_commits,
            use_llm=use_llm,
        )
        elapsed = time.monotonic() - t0
        return score_case(case, result, elapsed)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return CaseScore(
            case_id=case.id,
            passed=False,
            checks={},
            elapsed_seconds=round(elapsed, 3),
            error=f"{type(exc).__name__}: {exc}",
        )


def print_report(scores: list[CaseScore], total_elapsed: float) -> None:
    """Print a human-readable summary report to stdout."""
    print()
    print("=== Git Explainer Evaluation Report ===")
    print(f"Ran {len(scores)} cases in {total_elapsed:.1f}s")
    print()

    passed_count = 0
    failed_count = 0
    error_count = 0

    for s in scores:
        if s.error:
            error_count += 1
            print(f"ERROR   {s.case_id}  ({s.elapsed_seconds:.1f}s)  {s.error}")
        elif s.passed:
            passed_count += 1
            num_checks = len(s.checks)
            print(f"PASSED  {s.case_id}  ({s.elapsed_seconds:.1f}s)  [all {num_checks} checks passed]")
        else:
            failed_count += 1
            failed_checks = [name for name, ok in s.checks.items() if not ok]
            print(f"FAILED  {s.case_id}  ({s.elapsed_seconds:.1f}s)  [{', '.join(failed_checks)}]")

    print()
    total = len(scores)
    rate = (passed_count / total * 100) if total else 0.0
    print(f"Summary: {passed_count} passed, {failed_count} failed, {error_count} errors out of {total} total")
    print(f"Pass rate: {rate:.1f}%")


def save_results(scores: list[CaseScore], path: Path) -> None:
    """Write full scoring results to a JSON file."""
    path.write_text(
        json.dumps([asdict(s) for s in scores], indent=2) + "\n",
        encoding="utf-8",
    )


def filter_cases(
    cases: list[BenchmarkCase],
    *,
    tags: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[BenchmarkCase]:
    """Filter benchmark cases by tags or IDs."""
    if ids:
        id_set = set(ids)
        cases = [c for c in cases if c.id in id_set]
    if tags:
        tag_set = set(tags)
        cases = [c for c in cases if tag_set.intersection(c.tags)]
    return cases


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the Git Explainer evaluation harness.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Run only cases matching at least one of these tags.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Run only cases with these IDs.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Override all cases to use_llm=False.",
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path("eval/benchmark.json"),
        help="Path to benchmark JSON file (default: eval/benchmark.json).",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("eval/results.json"),
        help="Path to write results JSON (default: eval/results.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the evaluation harness."""
    args = build_parser().parse_args(argv)

    cases = load_benchmark(args.benchmark_file)
    cases = filter_cases(cases, tags=args.tags, ids=args.ids)

    if not cases:
        print("No cases to run after filtering.")
        sys.exit(0)

    print(f"Setting up repos for {len(cases)} case(s)...")
    repo_map = setup_repos(cases)

    scores: list[CaseScore] = []
    total_start = time.monotonic()

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Running {case.id}...")
        repo_path = repo_map.get(case.repo_url, "")
        score = run_case(case, repo_path, no_llm=args.no_llm)
        scores.append(score)

    total_elapsed = time.monotonic() - total_start

    print_report(scores, total_elapsed)
    save_results(scores, args.results_file)
    print(f"\nResults written to {args.results_file}")


if __name__ == "__main__":
    main()

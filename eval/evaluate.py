"""Evaluation harness for the Git Explainer agent.

Loads benchmark cases from benchmark.json, runs the agent on each,
scores outputs against expected fields, and produces a summary report.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from git_explainer import llm
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
    # Mirrors the CLI flag. Defaults to True so new cases inherit the
    # stricter guardrail; per-case opt-out is done by setting this to
    # ``false`` in benchmark.json (e.g., for no-owner/no-repo cases).
    enforce_public_repo: bool = True

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
            enforce_public_repo=bool(data.get("enforce_public_repo", True)),
        )


@dataclass
class CaseScore:
    """Result of scoring a single benchmark case."""

    case_id: str
    passed: bool
    checks: dict[str, bool]
    elapsed_seconds: float
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None


_CITATION_TOKEN_RE = re.compile(
    r"\[(?P<kind>commit|pr|issue):(?P<value>(?:[0-9a-f]{7,40}|none)|#\d+)\]"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    """Load and parse benchmark cases from a JSON file."""
    if not path.exists():
        print(f"Benchmark file not found: {path}")
        sys.exit(1)

    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw.get("cases", []) if isinstance(raw, dict) else raw
    if not data:
        print(f"Benchmark file is empty: {path}")
        sys.exit(1)

    return [BenchmarkCase.from_dict(case) for case in data]


def _canonicalize_repo_ref(repo_ref: str) -> str:
    """Normalize repo URLs so local-project aliases compare consistently."""
    normalized = repo_ref.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.split(":", 1)[1]
    return normalized.lower()


def _current_repo_aliases() -> set[str]:
    """Return identifiers that should be treated as aliases for the cwd repo."""
    cwd = Path.cwd().resolve()
    aliases = {str(cwd).lower()}
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return aliases

    if remote:
        aliases.add(_canonicalize_repo_ref(remote))
    return aliases


def _is_local_repo(repo_url: str) -> bool:
    """Return True if the repo_url points to the current project."""
    cwd = Path.cwd().resolve()
    try:
        candidate = Path(repo_url).resolve()
        if candidate == cwd:
            return True
    except (OSError, ValueError):
        pass
    return _canonicalize_repo_ref(repo_url) in _current_repo_aliases()


def _requires_local_no_origin_copy(case: BenchmarkCase) -> bool:
    """Use an isolated local copy when owner/repo are omitted for the cwd repo."""
    return case.owner is None and case.repo_name is None and _is_local_repo(case.repo_url)


def _repo_cache_key(case: BenchmarkCase) -> str:
    """Cache key for repo setup, including local no-origin copies when needed."""
    if _requires_local_no_origin_copy(case):
        return f"{_canonicalize_repo_ref(case.repo_url)}::no-origin"
    return case.repo_url


def _make_local_repo_copy_without_origin(source: Path) -> str:
    """Copy the cwd repo so owner-less cases cannot infer GitHub metadata from origin."""
    dest = Path(tempfile.mkdtemp(prefix="eval_repo_"))
    shutil.copytree(source, dest, dirs_exist_ok=True)
    subprocess.run(
        ["git", "remote", "remove", "origin"],
        cwd=dest,
        check=False,
        capture_output=True,
        text=True,
    )
    return str(dest)


def setup_repos(cases: list[BenchmarkCase]) -> dict[str, str]:
    """Clone (or reuse) repos for benchmark cases. Returns repo-key -> local path."""
    repo_map: dict[str, str] = {}
    for case in cases:
        key = _repo_cache_key(case)
        if key in repo_map:
            continue
        url = case.repo_url
        if _requires_local_no_origin_copy(case):
            repo_map[key] = _make_local_repo_copy_without_origin(Path.cwd())
            continue
        if _is_local_repo(url):
            repo_map[key] = str(Path.cwd())
        else:
            dest = tempfile.mkdtemp(prefix="eval_repo_")
            # Depth 2000 so external repos reach far enough back for the
            # ground-truth commits referenced by `expected_commit_shas`.
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "2000", url, dest],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                repo_map[key] = dest
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: Failed to clone {url}: {exc.stderr.strip()}")
                repo_map[key] = ""  # empty string signals clone failure
    return repo_map


def score_error_case(case: BenchmarkCase, error: str | None, elapsed: float) -> CaseScore:
    """Score a case that is expected to raise an error."""
    expected = case.expected
    checks: dict[str, bool] = {}

    if error is None:
        # Expected an error but none occurred
        checks["expects_error"] = False
        return CaseScore(
            case_id=case.id,
            passed=False,
            checks=checks,
            elapsed_seconds=round(elapsed, 3),
            error=None,
            metrics={},
        )

    checks["expects_error"] = True

    if "error_contains" in expected:
        checks["error_contains"] = expected["error_contains"].lower() in error.lower()

    passed = all(checks.values())
    return CaseScore(
        case_id=case.id,
        passed=passed,
        checks=checks,
        elapsed_seconds=round(elapsed, 3),
        error=None,  # Expected error — not a harness failure
        metrics={},
    )


def score_case(
    case: BenchmarkCase,
    result: ExplanationResult,
    elapsed: float,
    *,
    use_llm_judge: bool = False,
) -> CaseScore:
    """Score an agent result against expected values from the benchmark case."""
    expected = case.expected
    checks: dict[str, bool] = {}

    if "min_commits" in expected:
        checks["min_commits"] = len(result["commits"]) >= expected["min_commits"]

    if "max_commits" in expected:
        checks["max_commits"] = len(result["commits"]) <= expected["max_commits"]

    if "commit_message_contains" in expected:
        commit_message_targets = expected["commit_message_contains"]
        if commit_message_targets:
            messages = [c["message"].lower() for c in result["commits"]]
            checks["commit_message_contains"] = all(
                any(kw.lower() in msg for msg in messages)
                for kw in commit_message_targets
            )
        # Empty commit-message target lists are treated as intentionally vacuous
        # and skipped so they do not show up as misleading "passed" checks.

    if "expects_no_pull_requests" in expected:
        expects_none = bool(expected["expects_no_pull_requests"])
        checks["expects_no_pull_requests"] = (len(result["pull_requests"]) == 0) == expects_none
    elif "pr_numbers" in expected:
        expected_pr_numbers = expected["pr_numbers"]
        if expected_pr_numbers == []:
            checks["pr_numbers"] = len(result["pull_requests"]) == 0
        else:
            found_prs = {pr["number"] for pr in result["pull_requests"]}
            checks["pr_numbers"] = all(n in found_prs for n in expected_pr_numbers)

    if "expects_no_issues" in expected:
        expects_none = bool(expected["expects_no_issues"])
        checks["expects_no_issues"] = (len(result["issues"]) == 0) == expects_none
    elif "issue_numbers" in expected:
        expected_issue_numbers = expected["issue_numbers"]
        if expected_issue_numbers == []:
            checks["issue_numbers"] = len(result["issues"]) == 0
        else:
            found_issues = {issue["number"] for issue in result["issues"]}
            checks["issue_numbers"] = all(n in found_issues for n in expected_issue_numbers)

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

    if "expected_commit_shas" in expected:
        expected_shas = expected["expected_commit_shas"] or []
        if expected_shas:
            checks["expected_commit_shas"] = all(
                _commit_sha_matches(sha, result["commits"]) for sha in expected_shas
            )

    retrieval_metrics = _compute_retrieval_metrics(case, result)
    citation_metrics = _compute_citation_metrics(result)
    faithfulness_metrics = _compute_faithfulness_metrics(
        case,
        result,
        retrieval_metrics,
        citation_metrics,
    )

    if "retrieval_accuracy_min" in expected:
        checks["retrieval_accuracy_min"] = (
            retrieval_metrics["retrieval_accuracy"] is not None
            and retrieval_metrics["retrieval_accuracy"] >= expected["retrieval_accuracy_min"]
        )

    if "citation_coverage_min" in expected:
        checks["citation_coverage_min"] = (
            citation_metrics["citation_coverage"] is not None
            and citation_metrics["citation_coverage"] >= expected["citation_coverage_min"]
        )

    if "citation_validity_min" in expected:
        checks["citation_validity_min"] = (
            citation_metrics["citation_validity"] is not None
            and citation_metrics["citation_validity"] >= expected["citation_validity_min"]
        )

    if "faithfulness_rubric_min" in expected:
        checks["faithfulness_rubric_min"] = (
            faithfulness_metrics["overall"] >= expected["faithfulness_rubric_min"]
        )

    llm_judge: dict[str, Any] | None = None
    if use_llm_judge:
        llm_judge = _compute_llm_judge_faithfulness(result, case)

        min_rating = expected.get("llm_judge_min_rating")
        if min_rating is not None:
            checks["llm_judge_min_rating"] = _llm_judge_meets_min_rating(
                llm_judge.get("rating", "unscored"),
                min_rating,
            )

    passed = all(checks.values()) if checks else True
    metrics: dict[str, Any] = {
        **retrieval_metrics,
        **citation_metrics,
        "faithfulness_rubric": faithfulness_metrics,
    }
    if llm_judge is not None:
        metrics["llm_judge"] = llm_judge

    return CaseScore(
        case_id=case.id,
        passed=passed,
        checks=checks,
        elapsed_seconds=round(elapsed, 3),
        metrics=metrics,
    )


def _compute_retrieval_metrics(
    case: BenchmarkCase,
    result: ExplanationResult,
) -> dict[str, Any]:
    """Compute benchmark retrieval accuracy against gold targets in the case."""
    expected = case.expected
    matched = 0
    targets = 0
    breakdown: dict[str, dict[str, int]] = {}

    commit_targets = expected.get("commit_message_contains", [])
    if commit_targets:
        messages = [commit["message"].lower() for commit in result["commits"]]
        hit_count = sum(
            1
            for needle in commit_targets
            if any(needle.lower() in message for message in messages)
        )
        breakdown["commit_messages"] = {"matched": hit_count, "targets": len(commit_targets)}
        matched += hit_count
        targets += len(commit_targets)

    expected_shas = expected.get("expected_commit_shas", []) or []
    if expected_shas:
        hit_count = sum(
            1 for sha in expected_shas if _commit_sha_matches(sha, result["commits"])
        )
        breakdown["commit_shas"] = {"matched": hit_count, "targets": len(expected_shas)}
        matched += hit_count
        targets += len(expected_shas)

    pr_targets = expected.get("pr_numbers", [])
    if pr_targets:
        found_prs = {pr["number"] for pr in result["pull_requests"]}
        hit_count = sum(1 for number in pr_targets if number in found_prs)
        breakdown["pull_requests"] = {"matched": hit_count, "targets": len(pr_targets)}
        matched += hit_count
        targets += len(pr_targets)

    issue_targets = expected.get("issue_numbers", [])
    if issue_targets:
        found_issues = {issue["number"] for issue in result["issues"]}
        hit_count = sum(1 for number in issue_targets if number in found_issues)
        breakdown["issues"] = {"matched": hit_count, "targets": len(issue_targets)}
        matched += hit_count
        targets += len(issue_targets)

    resolved = result.get("resolved_target") or {}
    if "resolved_file_path" in expected:
        hit_count = int(resolved.get("file_path") == expected["resolved_file_path"])
        breakdown["resolved_file_path"] = {"matched": hit_count, "targets": 1}
        matched += hit_count
        targets += 1

    matched_terms = expected.get("resolved_matched_terms", [])
    if matched_terms:
        found_terms = set(resolved.get("matched_terms", []))
        hit_count = sum(1 for term in matched_terms if term in found_terms)
        breakdown["resolved_matched_terms"] = {"matched": hit_count, "targets": len(matched_terms)}
        matched += hit_count
        targets += len(matched_terms)

    accuracy = _safe_ratio(matched, targets)
    sha_breakdown = breakdown.get("commit_shas", {"matched": 0, "targets": 0})
    return {
        "retrieval_matched_count": matched,
        "retrieval_target_count": targets,
        "retrieval_accuracy": accuracy,
        "retrieval_breakdown": breakdown,
        "commit_sha_matches": sha_breakdown["matched"],
        "commit_sha_total": sha_breakdown["targets"],
    }


def _commit_sha_matches(expected_sha: str, commits: list[dict[str, Any]]) -> bool:
    """Return True if any commit in the result matches `expected_sha`.

    The match is order-insensitive: the expected SHA is taken as the ground
    truth. Either the commit's full SHA equals `expected_sha`, or the commit's
    short SHA is a prefix of `expected_sha` (common when the agent only
    surfaces 7-character SHAs), or vice versa.
    """
    if not expected_sha:
        return False
    expected = expected_sha.lower()
    for commit in commits:
        full = str(commit.get("full_sha", "")).lower()
        short = str(commit.get("sha", "")).lower()
        if full and (full == expected or full.startswith(expected) or expected.startswith(full)):
            return True
        if short and (short == expected or expected.startswith(short) or short.startswith(expected)):
            return True
    return False


def _compute_citation_metrics(result: ExplanationResult) -> dict[str, Any]:
    """Measure sentence-level citation coverage and citation validity."""
    sections = result["explanation"]
    citable_sentences = 0
    cited_sentences = 0
    total_citations = 0
    valid_citations = 0

    for text in sections.values():
        for sentence in _iter_citable_sentences(str(text)):
            citable_sentences += 1
            citations = list(_CITATION_TOKEN_RE.finditer(sentence))
            if citations:
                cited_sentences += 1
            total_citations += len(citations)
            valid_citations += sum(
                1
                for match in citations
                if _citation_is_valid(result, match.group("kind"), match.group("value"))
            )

    return {
        "citable_sentence_count": citable_sentences,
        "cited_sentence_count": cited_sentences,
        "citation_coverage": _safe_ratio(cited_sentences, citable_sentences),
        "citation_count": total_citations,
        "valid_citation_count": valid_citations,
        "citation_validity": _safe_ratio(valid_citations, total_citations),
    }


def _compute_faithfulness_metrics(
    case: BenchmarkCase,
    result: ExplanationResult,
    retrieval_metrics: dict[str, Any],
    citation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Estimate rubric-style faithfulness from retrieval, citations, and answer completeness."""
    sections = result["explanation"]
    all_text = " ".join(str(sections[key]) for key in sections).lower()

    non_empty_sections = sum(1 for text in sections.values() if str(text).strip())
    completeness_ratio = non_empty_sections / len(sections) if sections else 0.0

    expected_phrases = case.expected.get("explanation_contains", [])
    if expected_phrases:
        phrase_ratio = sum(
            1 for phrase in expected_phrases if phrase.lower() in all_text
        ) / len(expected_phrases)
        answer_ratio = (completeness_ratio + phrase_ratio) / 2
    else:
        answer_ratio = completeness_ratio

    retrieval_ratio = retrieval_metrics["retrieval_accuracy"]
    if retrieval_ratio is None:
        expected_min_commits = case.expected.get("min_commits")
        if expected_min_commits:
            retrieval_ratio = min(len(result["commits"]) / expected_min_commits, 1.0)
        else:
            retrieval_ratio = 1.0 if result["commits"] or result["resolved_target"] else 0.0

    citation_ratio = citation_metrics["citation_coverage"] or 0.0
    if citation_metrics["citation_validity"] is not None:
        citation_ratio = (citation_ratio + citation_metrics["citation_validity"]) / 2

    limitations_text = str(sections.get("limitations", ""))
    limitation_sentences = list(_iter_citable_sentences(limitations_text))
    if limitation_sentences:
        limitation_cited = sum(
            1 for sentence in limitation_sentences if _CITATION_TOKEN_RE.search(sentence)
        )
        limitations_ratio = limitation_cited / len(limitation_sentences)
    else:
        limitations_ratio = 0.0

    component_ratios = {
        "answer_completeness": answer_ratio,
        "retrieval_support": retrieval_ratio,
        "citation_grounding": citation_ratio,
        "scope_honesty": limitations_ratio,
    }
    overall_ratio = sum(component_ratios.values()) / len(component_ratios)

    return {
        "mode": "proxy",
        "overall": _ratio_to_rubric(overall_ratio),
        "components": {
            name: _ratio_to_rubric(ratio)
            for name, ratio in component_ratios.items()
        },
    }


_LLM_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator of Git history explanations. "
    "Your job is to rate whether an agent's explanation faithfully reflects ONLY "
    "the evidence it was shown (pull request titles/bodies, issue titles/bodies, "
    "and commit messages). You MUST NOT rely on outside knowledge about the "
    "repository, project, or author.\n\n"
    "Return exactly one JSON object — no markdown fences, no prose before or "
    "after — with these keys:\n"
    "  \"rating\": one of \"accurate\", \"partially accurate\", \"hallucinated\"\n"
    "  \"reasoning\": a single-sentence justification grounded in the evidence\n"
    "  \"contradictions\": a list of strings naming specific contradictions "
    "(empty list if none)\n\n"
    "Rating definitions:\n"
    "  - \"accurate\": The explanation does not contradict the evidence. Claims "
    "unsupported by the evidence are acceptable ONLY when the agent clearly "
    "states uncertainty or says no evidence was found. Abstention is fine.\n"
    "  - \"partially accurate\": The explanation has a partial contradiction, "
    "or omits an important detail that the evidence clearly establishes.\n"
    "  - \"hallucinated\": The explanation makes confident factual claims that "
    "the evidence does not support — for example, attributing intent to a PR "
    "whose body does not support it, citing an issue that does not appear in "
    "the evidence, or fabricating a reason for the change.\n\n"
    "Respond with the JSON object only."
)


_LLM_JUDGE_EVIDENCE_CHAR_LIMIT = 2000


def _truncate(text: str, limit: int = _LLM_JUDGE_EVIDENCE_CHAR_LIMIT) -> str:
    """Truncate long free-form text so the judge prompt stays bounded."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _build_llm_judge_prompt(result: ExplanationResult, case: BenchmarkCase) -> str:
    """Build the user prompt presented to the judge."""
    query_lines: list[str] = []
    query_lines.append(f"Repo: {case.repo_url}")
    if case.owner and case.repo_name:
        query_lines.append(f"Repo slug: {case.owner}/{case.repo_name}")
    if case.file_path:
        query_lines.append(f"File: {case.file_path}")
    if case.start_line is not None and case.end_line is not None:
        query_lines.append(f"Line range: {case.start_line}-{case.end_line}")
    if case.question:
        query_lines.append(f"Question: {case.question}")

    pr_blocks: list[str] = []
    for pr in result.get("pull_requests", []) or []:
        number = pr.get("number")
        title = pr.get("title", "")
        body = _truncate(pr.get("body", ""))
        pr_blocks.append(
            f"PR #{number}: {title}\n"
            f"Body:\n{body if body else '(empty)'}"
        )

    issue_blocks: list[str] = []
    for issue in result.get("issues", []) or []:
        number = issue.get("number")
        title = issue.get("title", "")
        body = _truncate(issue.get("body", ""))
        issue_blocks.append(
            f"Issue #{number}: {title}\n"
            f"Body:\n{body if body else '(empty)'}"
        )

    commit_blocks: list[str] = []
    for commit in result.get("commits", []) or []:
        sha = commit.get("sha") or commit.get("full_sha", "")
        message = _truncate(commit.get("message", ""), limit=800)
        commit_blocks.append(f"Commit {sha}: {message}")

    sections = result.get("explanation", {}) or {}
    explanation_text = (
        "what_changed:\n" + str(sections.get("what_changed", "")) + "\n\n"
        "why:\n" + str(sections.get("why", "")) + "\n\n"
        "tradeoffs:\n" + str(sections.get("tradeoffs", "")) + "\n\n"
        "limitations:\n" + str(sections.get("limitations", "")) + "\n\n"
        "summary:\n" + str(sections.get("summary", ""))
    )

    def _fmt(blocks: list[str], label: str) -> str:
        if not blocks:
            return f"{label}: (none)"
        return f"{label}:\n" + "\n\n".join(blocks)

    return (
        "Query:\n" + "\n".join(query_lines) + "\n\n"
        + _fmt(pr_blocks, "Pull requests") + "\n\n"
        + _fmt(issue_blocks, "Issues") + "\n\n"
        + _fmt(commit_blocks, "Commits") + "\n\n"
        + "Agent explanation:\n" + explanation_text + "\n\n"
        + "Rate the explanation and return the JSON object described in the system prompt."
    )


_LLM_JUDGE_VALID_RATINGS = {"accurate", "partially accurate", "hallucinated"}
_LLM_JUDGE_PASS_RATINGS = {"accurate", "partially accurate"}
_LLM_JUDGE_RATING_ORDER = {
    "hallucinated": 0,
    "partially accurate": 1,
    "accurate": 2,
}


def _parse_llm_judge_response(response: str) -> dict[str, Any] | None:
    """Parse the judge's JSON response. Returns None on failure."""
    if not response:
        return None
    text = response.strip()
    # Strip a surrounding markdown fence if one slipped through.
    if text.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
    # If there is prose around the JSON, find the first {...} block.
    if not text.startswith("{"):
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
            return None
        text = text[brace_start : brace_end + 1]
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    rating = str(parsed.get("rating", "")).strip().lower()
    if rating not in _LLM_JUDGE_VALID_RATINGS:
        return None
    contradictions = parsed.get("contradictions", []) or []
    if not isinstance(contradictions, list):
        contradictions = [str(contradictions)]
    return {
        "rating": rating,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "contradictions": [str(c) for c in contradictions],
    }


def _llm_judge_meets_min_rating(rating: str, min_rating: str) -> bool:
    """Return True if `rating` is at least as good as `min_rating`."""
    actual = _LLM_JUDGE_RATING_ORDER.get(str(rating).lower())
    threshold = _LLM_JUDGE_RATING_ORDER.get(str(min_rating).lower())
    if actual is None or threshold is None:
        return False
    return actual >= threshold


def _compute_llm_judge_faithfulness(
    result: ExplanationResult,
    case: BenchmarkCase,
) -> dict[str, Any]:
    """Rate an agent explanation on the 3-point faithfulness rubric via an LLM judge."""
    if not llm.is_available():
        return {
            "rating": "skipped",
            "reasoning": "llm unavailable",
            "contradictions": [],
            "passes": False,
            "raw_response": None,
        }

    prompt = _build_llm_judge_prompt(result, case)
    raw_response: str | None = None

    for attempt in range(2):
        if attempt == 0:
            user_content = prompt
        else:
            user_content = (
                "Your previous response was not valid JSON. Return a SINGLE "
                "JSON object ONLY, with keys rating, reasoning, contradictions. "
                "No prose. No markdown fences.\n\n" + prompt
            )
        try:
            raw_response = llm.chat(
                user_content,
                system_prompt=_LLM_JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 — judge failure is non-fatal
            return {
                "rating": "unscored",
                "reasoning": f"judge call failed: {type(exc).__name__}: {exc}",
                "contradictions": [],
                "passes": False,
                "raw_response": None,
            }

        parsed = _parse_llm_judge_response(raw_response)
        if parsed is not None:
            return {
                "rating": parsed["rating"],
                "reasoning": parsed["reasoning"],
                "contradictions": parsed["contradictions"],
                "passes": parsed["rating"] in _LLM_JUDGE_PASS_RATINGS,
                "raw_response": None,
            }

    return {
        "rating": "unscored",
        "reasoning": "failed to parse judge response as JSON after retry",
        "contradictions": [],
        "passes": False,
        "raw_response": raw_response,
    }


def run_case(
    case: BenchmarkCase,
    repo_path: str,
    *,
    no_llm: bool = False,
    use_llm_judge: bool = False,
) -> CaseScore:
    """Run a single benchmark case and return its score."""
    if not repo_path:
        return CaseScore(
            case_id=case.id,
            passed=False,
            checks={},
            elapsed_seconds=0.0,
            error="Repository clone failed",
            metrics={},
        )

    expects_error = case.expected.get("expects_error", False)
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
            enforce_public_repo=case.enforce_public_repo,
        )
        elapsed = time.monotonic() - t0
        if expects_error:
            return score_error_case(case, None, elapsed)
        return score_case(case, result, elapsed, use_llm_judge=use_llm_judge)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        error_msg = f"{type(exc).__name__}: {exc}"
        if expects_error:
            return score_error_case(case, error_msg, elapsed)
        return CaseScore(
            case_id=case.id,
            passed=False,
            checks={},
            elapsed_seconds=round(elapsed, 3),
            error=error_msg,
            metrics={},
        )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def _ratio_to_rubric(ratio: float) -> float:
    bounded = max(0.0, min(1.0, ratio))
    return round(1.0 + 4.0 * bounded, 3)


def _iter_citable_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in _SENTENCE_RE.split(text.strip())
        if sentence.strip() and _sentence_needs_citation(sentence)
    ]


def _sentence_needs_citation(sentence: str) -> bool:
    plain = _CITATION_TOKEN_RE.sub("", sentence)
    plain = re.sub(r"[\s\W_]+", "", plain)
    return bool(plain)


def _citation_is_valid(result: ExplanationResult, kind: str, value: str) -> bool:
    if kind == "commit":
        if value == "none":
            return not result["commits"]
        for commit in result["commits"]:
            short_sha = str(commit.get("sha", ""))
            full_sha = str(commit.get("full_sha", ""))
            if value == short_sha or (full_sha and full_sha.startswith(value)):
                return True
        return False

    number = int(value.lstrip("#"))
    if kind == "pr":
        return any(pr["number"] == number for pr in result["pull_requests"])
    if kind == "issue":
        return any(issue["number"] == number for issue in result["issues"])
    return False


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 3)


def summarize_scores(
    cases: list[BenchmarkCase],
    scores: list[CaseScore],
    total_elapsed: float,
) -> dict[str, Any]:
    """Aggregate benchmark-wide metrics from per-case scores."""
    passed_count = 0
    failed_count = 0
    error_count = 0
    skipped_count = 0

    retrieval_matched = 0
    retrieval_targets = 0
    commit_sha_matches = 0
    commit_sha_total = 0
    cited_sentences = 0
    citable_sentences = 0
    valid_citations = 0
    total_citations = 0
    faithfulness_scores: list[float] = []
    latencies = [score.elapsed_seconds for score in scores if not score.skipped]

    llm_judge_accurate = 0
    llm_judge_partial = 0
    llm_judge_hallucinated = 0
    llm_judge_unscored = 0
    llm_judge_skipped = 0
    llm_judge_any_present = False

    for score in scores:
        if score.skipped:
            skipped_count += 1
            continue
        if score.error:
            error_count += 1
            continue
        if score.passed:
            passed_count += 1
        else:
            failed_count += 1

        retrieval_matched += int(score.metrics.get("retrieval_matched_count", 0))
        retrieval_targets += int(score.metrics.get("retrieval_target_count", 0))
        commit_sha_matches += int(score.metrics.get("commit_sha_matches", 0))
        commit_sha_total += int(score.metrics.get("commit_sha_total", 0))
        cited_sentences += int(score.metrics.get("cited_sentence_count", 0))
        citable_sentences += int(score.metrics.get("citable_sentence_count", 0))
        valid_citations += int(score.metrics.get("valid_citation_count", 0))
        total_citations += int(score.metrics.get("citation_count", 0))

        faithfulness = score.metrics.get("faithfulness_rubric", {}).get("overall")
        if faithfulness is not None:
            faithfulness_scores.append(float(faithfulness))

        judge = score.metrics.get("llm_judge")
        if judge is not None:
            llm_judge_any_present = True
            rating = judge.get("rating")
            if rating == "accurate":
                llm_judge_accurate += 1
            elif rating == "partially accurate":
                llm_judge_partial += 1
            elif rating == "hallucinated":
                llm_judge_hallucinated += 1
            elif rating == "skipped":
                llm_judge_skipped += 1
            else:
                llm_judge_unscored += 1

    total = len(scores)
    ran_total = passed_count + failed_count + error_count
    pass_rate = _safe_ratio(passed_count, ran_total)
    average_latency = round(sum(latencies) / len(latencies), 3) if latencies else None
    average_faithfulness = (
        round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
        if faithfulness_scores
        else None
    )

    llm_judge_scored = llm_judge_accurate + llm_judge_partial + llm_judge_hallucinated
    llm_judge_pass_rate = _safe_ratio(
        llm_judge_accurate + llm_judge_partial,
        llm_judge_scored,
    )
    llm_judge_summary: dict[str, Any] | None = None
    if llm_judge_any_present:
        llm_judge_summary = {
            "accurate_count": llm_judge_accurate,
            "partially_accurate_count": llm_judge_partial,
            "hallucinated_count": llm_judge_hallucinated,
            "unscored_count": llm_judge_unscored,
            "skipped_count": llm_judge_skipped,
            "scored_count": llm_judge_scored,
            "pass_rate": llm_judge_pass_rate,
        }

    return {
        "benchmark": {
            "case_count": len(cases),
            "repo_count": len({case.repo_url for case in cases}),
        },
        "counts": {
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "skipped": skipped_count,
            "total": total,
        },
        "pass_rate": pass_rate,
        "retrieval": {
            "matched_count": retrieval_matched,
            "target_count": retrieval_targets,
            "accuracy": _safe_ratio(retrieval_matched, retrieval_targets),
            "commit_sha_matches": commit_sha_matches,
            "commit_sha_total": commit_sha_total,
            "commit_sha_accuracy": _safe_ratio(commit_sha_matches, commit_sha_total),
        },
        "citation": {
            "cited_sentence_count": cited_sentences,
            "citable_sentence_count": citable_sentences,
            "coverage": _safe_ratio(cited_sentences, citable_sentences),
            "valid_citation_count": valid_citations,
            "citation_count": total_citations,
            "validity": _safe_ratio(valid_citations, total_citations),
        },
        "faithfulness_rubric": {
            "mode": "proxy",
            "average": average_faithfulness,
            "case_count": len(faithfulness_scores),
        },
        "llm_judge": llm_judge_summary,
        "latency": {
            "total_seconds": round(total_elapsed, 3),
            "average_seconds": average_latency,
            "p50_seconds": _percentile(latencies, 0.50),
            "p95_seconds": _percentile(latencies, 0.95),
        },
    }


def print_report(scores: list[CaseScore], summary: dict[str, Any]) -> None:
    """Print a human-readable summary report to stdout."""
    print()
    print("=== Git Explainer Evaluation Report ===")
    benchmark = summary["benchmark"]
    counts = summary["counts"]
    latency = summary["latency"]
    retrieval = summary["retrieval"]
    citation = summary["citation"]
    faithfulness = summary["faithfulness_rubric"]

    print(
        f"Ran {benchmark['case_count']} cases across {benchmark['repo_count']} repos "
        f"in {latency['total_seconds']:.1f}s"
    )
    print()

    for s in scores:
        if s.skipped:
            print(f"SKIPPED {s.case_id}  ({s.skip_reason or 'skipped'})")
        elif s.error:
            print(f"ERROR   {s.case_id}  ({s.elapsed_seconds:.1f}s)  {s.error}")
        elif s.passed:
            num_checks = len(s.checks)
            print(f"PASSED  {s.case_id}  ({s.elapsed_seconds:.1f}s)  [all {num_checks} checks passed]")
        else:
            failed_checks = [name for name, ok in s.checks.items() if not ok]
            print(f"FAILED  {s.case_id}  ({s.elapsed_seconds:.1f}s)  [{', '.join(failed_checks)}]")

    print()
    skipped_count = counts.get("skipped", 0)
    skipped_suffix = f", {skipped_count} skipped" if skipped_count else ""
    print(
        "Summary: "
        f"{counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['errors']} errors{skipped_suffix} out of {counts['total']} total"
    )
    print(f"Pass rate: {(summary['pass_rate'] or 0.0) * 100:.1f}%")
    if retrieval["accuracy"] is None:
        print("Retrieval accuracy: n/a (no gold retrieval targets were specified)")
    else:
        print(
            f"Retrieval accuracy: {retrieval['accuracy'] * 100:.1f}% "
            f"({retrieval['matched_count']}/{retrieval['target_count']} expected targets)"
        )
    if retrieval.get("commit_sha_accuracy") is not None:
        print(
            f"Commit SHA match rate: {retrieval['commit_sha_accuracy'] * 100:.1f}% "
            f"({retrieval['commit_sha_matches']}/{retrieval['commit_sha_total']} expected SHAs)"
        )
    if citation["coverage"] is None:
        print("Citation coverage: n/a (no citable explanation sentences)")
    else:
        print(
            f"Citation coverage: {citation['coverage'] * 100:.1f}% "
            f"({citation['cited_sentence_count']}/{citation['citable_sentence_count']} sentences)"
        )
    if citation["validity"] is None:
        print("Citation validity: n/a (no citations present)")
    else:
        print(
            f"Citation validity: {citation['validity'] * 100:.1f}% "
            f"({citation['valid_citation_count']}/{citation['citation_count']} citations)"
        )
    if faithfulness["average"] is not None:
        print(
            f"Faithfulness rubric ({faithfulness['mode']}): "
            f"{faithfulness['average']:.2f}/5.00"
        )
    llm_judge = summary.get("llm_judge")
    if llm_judge is not None:
        print()
        print("LLM-as-judge faithfulness:")
        print(
            f"  accurate: {llm_judge['accurate_count']}  "
            f"partially accurate: {llm_judge['partially_accurate_count']}  "
            f"hallucinated: {llm_judge['hallucinated_count']}"
        )
        print(
            f"  unscored: {llm_judge['unscored_count']}  "
            f"skipped: {llm_judge['skipped_count']}"
        )
        if llm_judge["pass_rate"] is None:
            print(
                "  Pass rate: n/a "
                f"(0/{llm_judge['scored_count']} scored; "
                f"{llm_judge['skipped_count']} skipped, "
                f"{llm_judge['unscored_count']} unscored)"
            )
        else:
            passing = llm_judge["accurate_count"] + llm_judge["partially_accurate_count"]
            print(
                f"  Pass rate: {llm_judge['pass_rate'] * 100:.1f}% "
                f"({passing}/{llm_judge['scored_count']} scored cases "
                f"rated accurate or partially accurate)"
            )
    if latency["average_seconds"] is not None:
        print(
            f"Latency: avg {latency['average_seconds']:.2f}s | "
            f"p50 {latency['p50_seconds']:.2f}s | "
            f"p95 {latency['p95_seconds']:.2f}s"
        )


def save_results(scores: list[CaseScore], summary: dict[str, Any], path: Path) -> None:
    """Write full scoring results to a JSON file."""
    path.write_text(
        json.dumps(
            {
                "summary": summary,
                "cases": [asdict(score) for score in scores],
            },
            indent=2,
        ) + "\n",
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
        "--use-llm-judge",
        action="store_true",
        default=False,
        help=(
            "Run an LLM-as-judge faithfulness scorer per case. Requires a "
            "configured LLM (e.g., GROQ_API_KEY). Adds one API call per case."
        ),
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
        if args.no_llm and case.use_llm:
            print(f"[{i}/{len(cases)}] Skipping {case.id} (use_llm=true under --no-llm)")
            scores.append(
                CaseScore(
                    case_id=case.id,
                    passed=False,
                    checks={},
                    elapsed_seconds=0.0,
                    skipped=True,
                    skip_reason="use_llm=true case skipped under --no-llm",
                )
            )
            continue
        print(f"[{i}/{len(cases)}] Running {case.id}...")
        repo_path = repo_map.get(_repo_cache_key(case), "")
        score = run_case(
            case,
            repo_path,
            no_llm=args.no_llm,
            use_llm_judge=args.use_llm_judge,
        )
        scores.append(score)

    total_elapsed = time.monotonic() - total_start

    summary = summarize_scores(cases, scores, total_elapsed)
    print_report(scores, summary)
    save_results(scores, summary, args.results_file)
    print(f"\nResults written to {args.results_file}")


if __name__ == "__main__":
    main()

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

# Eval-only Anthropic judge adapter. Imported by name (not via wildcard) so
# the agent's runtime LLM path stays on Groq and is untouched.
try:
    from eval import judge_anthropic
except ImportError:  # pragma: no cover — when running this file as a script
    import judge_anthropic  # type: ignore[no-redef]


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

# Stopwords used when computing sentence<->evidence keyword overlap for
# citation_support. Kept small and domain-neutral so we neither overfit to
# git prose nor accidentally drop content words.
_CITATION_SUPPORT_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "in",
        "on", "at", "to", "for", "by", "with", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "has", "have", "had", "do", "does", "did",
        "this", "that", "these", "those", "it", "its", "into", "over", "about",
        "which", "who", "whom", "what", "when", "where", "why", "how", "we",
        "they", "their", "our", "not", "no", "yes", "so", "than", "also", "more",
        "most", "some", "any", "all", "such", "via", "per", "out", "up", "down",
        "should", "would", "could", "may", "might", "can", "will", "shall",
        "because", "due", "been", "added", "add", "change", "changed", "changes",
        "new", "old", "made", "make", "used", "use", "using",
    }
)

# Tokens that signal the agent flagged limits or abstained from unsupported
# claims. Used to make scope_honesty reflect actual behavior on thin cases
# rather than just counting whether the limitations block has citations.
_ABSTENTION_MARKERS = (
    "no linked pull request",
    "no associated issue",
    "no associated pull request",
    "no linked issue",
    "no linked issues",
    "no pull request",
    "no issue found",
    "no issues found",
    "not found",
    "no evidence",
    "limited evidence",
    "insufficient evidence",
    "uncertain",
    "unknown",
    "cannot determine",
    "without additional",
)

# A check is "trivial" if it only asserts plumbing we could pass by echoing
# boilerplate (the fallback flag, a filename mention, the resolved file path,
# or the literal citation prefix). Used to separate pass-rate honesty from
# pass-rate-by-plumbing.
_TRIVIAL_CHECK_NAMES = frozenset(
    {
        "used_fallback",
        "explanation_contains",
        "resolved_file_path",
        "resolved_preview_contains",
        "resolved_matched_terms",
    }
)


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
        overall = faithfulness_metrics["overall"]
        checks["faithfulness_rubric_min"] = (
            overall is not None and overall >= expected["faithfulness_rubric_min"]
        )

    if "citation_support_min" in expected:
        support = citation_metrics.get("citation_support")
        checks["citation_support_min"] = (
            support is not None and support >= expected["citation_support_min"]
        )

    if expected.get("must_abstain"):
        checks["must_abstain"] = (
            retrieval_metrics.get("retrieval_precision") == 1.0
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
    """Compute benchmark retrieval recall + must-abstain precision against gold targets.

    Recall side: count hits against hand-authored positive targets.
    Precision side: only measured on must-abstain cases (`expects_no_*: true`
    or an explicit empty `pr_numbers: []` / `issue_numbers: []`). For
    positive-target cases we do NOT penalize extra PRs/issues because the
    gold lists are a lower bound, not an exhaustive enumeration.
    """
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

    # Must-abstain precision: on cases that assert "return none", any
    # returned PR/issue is noise. This is the only axis where we can
    # measure precision honestly without per-case exhaustive gold lists.
    must_abstain_prs = bool(expected.get("expects_no_pull_requests")) or (
        "expects_no_pull_requests" not in expected
        and "pr_numbers" in expected
        and expected["pr_numbers"] == []
    )
    must_abstain_issues = bool(expected.get("expects_no_issues")) or (
        "expects_no_issues" not in expected
        and "issue_numbers" in expected
        and expected["issue_numbers"] == []
    )
    unexpected_prs = 0
    unexpected_issues = 0
    precision_applicable = False
    if must_abstain_prs:
        precision_applicable = True
        unexpected_prs = len(result.get("pull_requests", []) or [])
    if must_abstain_issues:
        precision_applicable = True
        unexpected_issues = len(result.get("issues", []) or [])

    precision: float | None = None
    if precision_applicable:
        noise = unexpected_prs + unexpected_issues
        precision = 1.0 if noise == 0 else 0.0

    return {
        "retrieval_matched_count": matched,
        "retrieval_target_count": targets,
        "retrieval_accuracy": accuracy,
        "retrieval_breakdown": breakdown,
        "commit_sha_matches": sha_breakdown["matched"],
        "commit_sha_total": sha_breakdown["targets"],
        "unexpected_prs": unexpected_prs,
        "unexpected_issues": unexpected_issues,
        "precision_applicable": precision_applicable,
        "retrieval_precision": precision,
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
    """Measure citation coverage, validity (ID-exists), and support (semantic overlap).

    - coverage: fraction of citable sentences that contain any citation. This
      is largely a format-compliance metric because the prompt and fallback
      both require it, so it is reported but not headlined.
    - validity: fraction of citation tokens whose referenced ID actually
      appears in the returned evidence. Necessary but not sufficient: a
      sentence can cite a real commit that does not support its claim.
    - support: fraction of citation tokens where the cited artifact's
      text shares at least one content word with the citing sentence.
      Intended to catch citations that look real but are glued onto
      unrelated prose. This is a weak signal, not ground truth.
    """
    sections = result["explanation"]
    citable_sentences = 0
    cited_sentences = 0
    total_citations = 0
    valid_citations = 0
    supported_citations = 0

    for text in sections.values():
        for sentence in _iter_citable_sentences(str(text)):
            citable_sentences += 1
            citations = list(_CITATION_TOKEN_RE.finditer(sentence))
            if citations:
                cited_sentences += 1
            total_citations += len(citations)
            sentence_tokens = _content_tokens(sentence)
            for match in citations:
                kind = match.group("kind")
                value = match.group("value")
                if _citation_is_valid(result, kind, value):
                    valid_citations += 1
                    if _citation_is_supported(result, kind, value, sentence_tokens):
                        supported_citations += 1

    return {
        "citable_sentence_count": citable_sentences,
        "cited_sentence_count": cited_sentences,
        "citation_coverage": _safe_ratio(cited_sentences, citable_sentences),
        "citation_count": total_citations,
        "valid_citation_count": valid_citations,
        "citation_validity": _safe_ratio(valid_citations, total_citations),
        "supported_citation_count": supported_citations,
        "citation_support": _safe_ratio(supported_citations, total_citations),
    }


def _content_tokens(text: str) -> set[str]:
    """Tokenize free text into lowercase content words for overlap checks."""
    stripped = _CITATION_TOKEN_RE.sub(" ", text.lower())
    raw = re.findall(r"[a-z][a-z0-9_]{2,}", stripped)
    return {token for token in raw if token not in _CITATION_SUPPORT_STOPWORDS}


def _citation_is_supported(
    result: ExplanationResult,
    kind: str,
    value: str,
    sentence_tokens: set[str],
) -> bool:
    """Return True if the cited artifact's text shares a content word with the sentence.

    Falls back to True for the sentinel `[commit:none]` token when the agent
    legitimately asserts no commit evidence, so abstention language is not
    punished.
    """
    if kind == "commit" and value == "none":
        return True
    if not sentence_tokens:
        # Pure citation-only sentence (all words stripped). Treat as supported
        # rather than mark every such citation as unsupported noise.
        return True

    artifact_text = ""
    if kind == "commit":
        for commit in result.get("commits", []) or []:
            short = str(commit.get("sha", ""))
            full = str(commit.get("full_sha", ""))
            if value == short or (full and full.startswith(value)):
                artifact_text = str(commit.get("message", ""))
                break
    else:
        number = int(value.lstrip("#"))
        collection = (
            result.get("pull_requests", []) if kind == "pr" else result.get("issues", [])
        ) or []
        for artifact in collection:
            if artifact.get("number") == number:
                artifact_text = " ".join(
                    [
                        str(artifact.get("title", "")),
                        str(artifact.get("body", "")),
                    ]
                )
                break

    if not artifact_text:
        return False

    artifact_tokens = _content_tokens(artifact_text)
    return bool(sentence_tokens & artifact_tokens)


def _compute_faithfulness_metrics(
    case: BenchmarkCase,
    result: ExplanationResult,
    retrieval_metrics: dict[str, Any],
    citation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Estimate rubric-style faithfulness from retrieval, citations, and answer completeness.

    This is a PROXY. It is not a ground-truth honesty score. Design goals:

    - Components that cannot be honestly measured for a given case return
      ``None`` and are excluded from the overall average, so "no signal"
      does not masquerade as a perfect score.
    - ``citation_grounding`` uses the semantic support rate when available,
      falling back to validity — never to coverage, which is near-100% by
      construction and inflates the number.
    - ``abstention_correctness`` replaces the old ``scope_honesty`` hack:
      cases with evidence score full marks; thin-evidence cases score by
      whether the explanation actually flags the limit, not by whether the
      limitations section happens to contain a citation token.
    """
    sections = result["explanation"]
    all_text = " ".join(str(sections[key]) for key in sections).lower()

    # answer_completeness: only count what we can actually verify. If the
    # case ships expected phrases, use those. If not, we do NOT reward the
    # agent for simply having non-empty sections — return None.
    expected_phrases = case.expected.get("explanation_contains", [])
    if expected_phrases:
        non_trivial = [p for p in expected_phrases if not _is_trivial_phrase(p)]
        if non_trivial:
            hits = sum(1 for phrase in non_trivial if phrase.lower() in all_text)
            answer_ratio: float | None = hits / len(non_trivial)
        else:
            answer_ratio = None
    else:
        answer_ratio = None

    # retrieval_support: strictly the measured recall against gold targets.
    # If there are no positive gold targets, we have no signal — return
    # None rather than defaulting to 1.0 or to a min_commits heuristic,
    # which previously let external-repo cases with weak gold data score
    # full marks on retrieval.
    retrieval_ratio: float | None = retrieval_metrics["retrieval_accuracy"]

    # citation_grounding: prefer semantic support over ID-validity; never
    # use coverage alone (near-100% by construction).
    if citation_metrics["citation_count"] == 0:
        citation_ratio: float | None = None
    elif citation_metrics["citation_support"] is not None:
        citation_ratio = citation_metrics["citation_support"]
    else:
        citation_ratio = citation_metrics["citation_validity"]

    # abstention_correctness: measure actual abstention behavior. "Thin"
    # cases (no PRs and no issues in the returned evidence) must surface
    # an abstention marker to score; rich-evidence cases score full marks.
    has_prs = bool(result.get("pull_requests"))
    has_issues = bool(result.get("issues"))
    if has_prs and has_issues:
        abstention_ratio: float | None = 1.0
    else:
        abstention_ratio = 1.0 if _mentions_abstention(all_text) else 0.0

    component_ratios = {
        "answer_completeness": answer_ratio,
        "retrieval_support": retrieval_ratio,
        "citation_grounding": citation_ratio,
        "abstention_correctness": abstention_ratio,
    }
    measured = {k: v for k, v in component_ratios.items() if v is not None}
    if measured:
        overall_ratio = sum(measured.values()) / len(measured)
        overall = _ratio_to_rubric(overall_ratio)
    else:
        overall = None

    return {
        "mode": "proxy",
        "overall": overall,
        "measured_component_count": len(measured),
        "total_component_count": len(component_ratios),
        "components": {
            name: (_ratio_to_rubric(ratio) if ratio is not None else None)
            for name, ratio in component_ratios.items()
        },
    }


def _is_trivial_phrase(phrase: str) -> bool:
    """Return True for `explanation_contains` entries that only test formatting.

    ``"[commit:"``, ``"[pr:"``, ``"[issue:"`` all pass the moment the agent
    emits a single citation token, regardless of what it actually said.
    """
    lowered = phrase.strip().lower()
    return lowered in {"[commit:", "[pr:", "[issue:"}


def _mentions_abstention(text: str) -> bool:
    """Return True if the explanation surfaces any abstention/uncertainty marker."""
    lowered = text.lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


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


def _judge_backend() -> tuple[str, Any] | None:
    """Pick the LLM-judge backend, preferring Anthropic for independence from the Groq agent.

    Returns ``(name, callable)`` where ``callable(user_content, system_prompt,
    temperature)`` returns the judge's response text, or ``None`` if no
    judge LLM is configured.
    """
    if judge_anthropic.is_available():
        return ("anthropic-" + judge_anthropic.model_id(), judge_anthropic.chat)
    if llm.is_available():
        return ("groq-" + str(llm.config.GROQ_MODEL), llm.chat)
    return None


def _compute_llm_judge_faithfulness(
    result: ExplanationResult,
    case: BenchmarkCase,
) -> dict[str, Any]:
    """Rate an agent explanation on the 3-point faithfulness rubric via an LLM judge."""
    backend = _judge_backend()
    if backend is None:
        return {
            "rating": "skipped",
            "reasoning": "no judge LLM configured (set ANTHROPIC_API_KEY or GROQ_API_KEY)",
            "contradictions": [],
            "passes": False,
            "raw_response": None,
            "judge_model": None,
        }
    judge_name, judge_chat = backend

    prompt = _build_llm_judge_prompt(result, case)
    raw_response: str | None = None
    last_exc_repr: str | None = None

    # Up to 3 attempts: attempt 0 is the real prompt; attempt 1 is a retry
    # that also asks for strict JSON (covers parse failures); attempt 2 is
    # an extra rate-limit cushion. Transient 429s get a back-off sleep so
    # a single rate-limit burst does not silently disqualify 12+ cases.
    for attempt in range(3):
        if attempt == 0:
            user_content = prompt
        else:
            user_content = (
                "Your previous response was not valid JSON. Return a SINGLE "
                "JSON object ONLY, with keys rating, reasoning, contradictions. "
                "No prose. No markdown fences.\n\n" + prompt
            )
        try:
            raw_response = judge_chat(
                user_content,
                system_prompt=_LLM_JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 — judge failure is non-fatal
            last_exc_repr = f"{type(exc).__name__}: {exc}"
            msg = str(exc).lower()
            is_rate_limit = (
                "ratelimit" in type(exc).__name__.lower()
                or "429" in msg
                or "rate_limit" in msg
                or "rate limit" in msg
                or "quota" in msg
            )
            if is_rate_limit and attempt < 2:
                sleep_seconds = 30.0 if attempt == 0 else 60.0
                time.sleep(sleep_seconds)
                continue
            return {
                "rating": "unscored",
                "reasoning": f"judge call failed: {last_exc_repr}",
                "contradictions": [],
                "passes": False,
                "raw_response": None,
                "judge_model": judge_name,
            }

        parsed = _parse_llm_judge_response(raw_response)
        if parsed is not None:
            return {
                "rating": parsed["rating"],
                "reasoning": parsed["reasoning"],
                "contradictions": parsed["contradictions"],
                "passes": parsed["rating"] in _LLM_JUDGE_PASS_RATINGS,
                "raw_response": None,
                "judge_model": judge_name,
            }

    return {
        "rating": "unscored",
        "reasoning": (
            f"failed to parse judge response as JSON after retries "
            f"(last_error={last_exc_repr})"
            if last_exc_repr
            else "failed to parse judge response as JSON after retries"
        ),
        "contradictions": [],
        "passes": False,
        "raw_response": raw_response,
        "judge_model": judge_name,
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
        # When the case requires the LLM (use_llm=True) and the agent fell
        # back specifically because the LLM call raised (rate limit, daily
        # quota, transport error), this is an environmental failure -- not
        # an agent regression. Mark the case skipped instead of failed so
        # the harness's pass-rate denominator is not contaminated by transient
        # upstream outages. The orchestrator surfaces the structured reason
        # via ``result["fallback_reason"]``.
        if (
            use_llm
            and result.get("used_fallback")
            and result.get("fallback_reason") == "llm_error"
        ):
            return CaseScore(
                case_id=case.id,
                passed=False,
                checks={},
                elapsed_seconds=round(elapsed, 3),
                skipped=True,
                skip_reason=(
                    "LLM call failed at agent runtime (transient upstream "
                    "error -- rate limit, quota, transport, or 5xx). The "
                    "agent fell back deterministically; this is not a code "
                    "regression."
                ),
            )
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
    """Aggregate benchmark-wide metrics from per-case scores.

    Every aggregate reports a denominator so downstream consumers cannot mistake
    "no cases contributed" for "the metric was at ceiling." The pass rate is
    also reported twice: overall, and restricted to non-trivial cases (cases
    with at least one non-plumbing check), so headline pass rate cannot be
    inflated by cases whose only checks are ``used_fallback`` + a filename
    mention.
    """
    passed_count = 0
    failed_count = 0
    error_count = 0
    skipped_count = 0

    non_trivial_passed = 0
    non_trivial_failed = 0

    retrieval_matched = 0
    retrieval_targets = 0
    commit_sha_matches = 0
    commit_sha_total = 0
    retrieval_cases_with_targets = 0

    precision_applicable_cases = 0
    precision_passing_cases = 0
    total_unexpected_prs = 0
    total_unexpected_issues = 0

    cited_sentences = 0
    citable_sentences = 0
    valid_citations = 0
    supported_citations = 0
    total_citations = 0

    faithfulness_scores: list[float] = []
    faithfulness_unscored_cases = 0
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

        has_non_trivial_check = any(
            name not in _TRIVIAL_CHECK_NAMES for name in score.checks
        )
        if score.passed:
            passed_count += 1
            if has_non_trivial_check:
                non_trivial_passed += 1
        else:
            failed_count += 1
            if has_non_trivial_check:
                non_trivial_failed += 1

        targets = int(score.metrics.get("retrieval_target_count", 0))
        retrieval_matched += int(score.metrics.get("retrieval_matched_count", 0))
        retrieval_targets += targets
        if targets > 0:
            retrieval_cases_with_targets += 1
        commit_sha_matches += int(score.metrics.get("commit_sha_matches", 0))
        commit_sha_total += int(score.metrics.get("commit_sha_total", 0))

        if score.metrics.get("precision_applicable"):
            precision_applicable_cases += 1
            if score.metrics.get("retrieval_precision") == 1.0:
                precision_passing_cases += 1
            total_unexpected_prs += int(score.metrics.get("unexpected_prs", 0))
            total_unexpected_issues += int(score.metrics.get("unexpected_issues", 0))

        cited_sentences += int(score.metrics.get("cited_sentence_count", 0))
        citable_sentences += int(score.metrics.get("citable_sentence_count", 0))
        valid_citations += int(score.metrics.get("valid_citation_count", 0))
        supported_citations += int(score.metrics.get("supported_citation_count", 0))
        total_citations += int(score.metrics.get("citation_count", 0))

        faithfulness = score.metrics.get("faithfulness_rubric", {}).get("overall")
        if faithfulness is not None:
            faithfulness_scores.append(float(faithfulness))
        else:
            faithfulness_unscored_cases += 1

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
    non_trivial_ran = non_trivial_passed + non_trivial_failed
    non_trivial_pass_rate = _safe_ratio(non_trivial_passed, non_trivial_ran)
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
    llm_judge_strict_pass_rate = _safe_ratio(llm_judge_accurate, llm_judge_scored)
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
            "strict_pass_rate": llm_judge_strict_pass_rate,
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
            "non_trivial_passed": non_trivial_passed,
            "non_trivial_ran": non_trivial_ran,
        },
        "pass_rate": pass_rate,
        "non_trivial_pass_rate": non_trivial_pass_rate,
        "retrieval": {
            "matched_count": retrieval_matched,
            "target_count": retrieval_targets,
            "accuracy": _safe_ratio(retrieval_matched, retrieval_targets),
            "cases_with_targets": retrieval_cases_with_targets,
            "commit_sha_matches": commit_sha_matches,
            "commit_sha_total": commit_sha_total,
            "commit_sha_accuracy": _safe_ratio(commit_sha_matches, commit_sha_total),
            "must_abstain_cases": precision_applicable_cases,
            "must_abstain_passed": precision_passing_cases,
            "must_abstain_precision": _safe_ratio(
                precision_passing_cases, precision_applicable_cases
            ),
            "unexpected_prs_total": total_unexpected_prs,
            "unexpected_issues_total": total_unexpected_issues,
        },
        "citation": {
            "cited_sentence_count": cited_sentences,
            "citable_sentence_count": citable_sentences,
            "coverage": _safe_ratio(cited_sentences, citable_sentences),
            "valid_citation_count": valid_citations,
            "citation_count": total_citations,
            "validity": _safe_ratio(valid_citations, total_citations),
            "supported_citation_count": supported_citations,
            "support_rate": _safe_ratio(supported_citations, total_citations),
        },
        "faithfulness_rubric": {
            "mode": "proxy",
            "average": average_faithfulness,
            "case_count": len(faithfulness_scores),
            "unscored_case_count": faithfulness_unscored_cases,
            "warning": (
                "This is a proxy score averaged from measured components only. "
                "It does not verify claim-level correctness. Prefer llm_judge "
                "when reporting a headline faithfulness number."
            ),
        },
        "llm_judge": llm_judge_summary,
        "latency": {
            "total_seconds": round(total_elapsed, 3),
            "average_seconds": average_latency,
            "p50_seconds": _percentile(latencies, 0.50),
            "p95_seconds": _percentile(latencies, 0.95),
        },
        "evaluation_honesty": _build_evaluation_honesty(
            cases,
            scores,
            llm_judge_any_present=llm_judge_any_present,
            llm_judge_scored=llm_judge_scored,
        ),
    }


def _build_evaluation_honesty(
    cases: list[BenchmarkCase],
    scores: list[CaseScore],
    *,
    llm_judge_any_present: bool,
    llm_judge_scored: int,
) -> dict[str, Any]:
    """Machine-readable self-audit of what the numbers in this report do and do not prove."""
    ran_cases = [
        case
        for case, score in zip(cases, scores, strict=False)
        if not score.skipped and not score.error
    ]
    ran_scores = [score for score in scores if not score.skipped and not score.error]

    trivial_only_ids = [
        score.case_id
        for score in ran_scores
        if score.checks and all(name in _TRIVIAL_CHECK_NAMES for name in score.checks)
    ]
    no_retrieval_target_ids = [
        score.case_id
        for score in ran_scores
        if int(score.metrics.get("retrieval_target_count", 0)) == 0
    ]
    no_citation_ids = [
        score.case_id
        for score in ran_scores
        if int(score.metrics.get("citation_count", 0)) == 0
    ]

    caveats: list[str] = [
        "Pass rate counts a case as passed iff every configured check is true. "
        "Cases whose only checks are plumbing (used_fallback, filename mention, "
        "resolved_file_path, resolved_matched_terms) are surfaced in "
        "`trivial_check_case_ids` so they can be excluded when judging honesty.",
        "Retrieval accuracy is recall against hand-picked gold targets. Cases "
        "without gold targets are not counted; see `no_retrieval_target_case_ids`.",
        "Retrieval precision is only measured on must-abstain cases (see "
        "`summary.retrieval.must_abstain_*`); for positive-target cases the "
        "gold list is a lower bound, so extra evidence is not penalized here.",
        "Citation coverage is a format-compliance metric — the prompt and "
        "fallback both require a citation in every citable sentence.",
        "Citation validity only checks that the cited ID exists in the "
        "returned evidence. It does not verify the artifact supports the claim.",
        "Citation support is a weak semantic-overlap check: at least one "
        "content word shared between the sentence and the cited artifact's "
        "text. It detects glued-on citations; it does not prove entailment.",
        "Faithfulness rubric is a PROXY, averaged over only the components "
        "with signal for each case (answer_completeness, retrieval_support, "
        "citation_grounding, abstention_correctness). Components with no "
        "signal return null rather than defaulting to 1.0.",
        "abstention_correctness scores thin-evidence cases by whether the "
        "explanation mentions an abstention marker; rich-evidence cases "
        "score full marks. It is not a test of claim-level honesty.",
    ]

    headline: dict[str, Any] = {
        "preferred_faithfulness_metric": (
            "llm_judge" if llm_judge_any_present and llm_judge_scored > 0 else "none"
        ),
        "proxy_faithfulness_is_ground_truth": False,
        "citation_coverage_is_format_compliance": True,
    }

    return {
        "total_cases": len(cases),
        "ran_cases": len(ran_cases),
        "trivial_check_case_ids": trivial_only_ids,
        "no_retrieval_target_case_ids": no_retrieval_target_ids,
        "no_citation_case_ids": no_citation_ids,
        "headline": headline,
        "caveats": caveats,
    }


def print_report(scores: list[CaseScore], summary: dict[str, Any]) -> None:
    """Print a human-readable summary report to stdout.

    The report is split into four sections so format-compliance metrics
    cannot be mistaken for honesty metrics:

      1. Pass/fail roll-up (with a separate non-trivial pass rate)
      2. Evidence retrieval (recall, must-abstain precision, SHA hits)
      3. Format compliance (citation coverage + validity — reported, not celebrated)
      4. Explanation honesty (LLM judge is the headline; proxy rubric is clearly labeled)
    """
    print()
    print("=== Git Explainer Evaluation Report ===")
    benchmark = summary["benchmark"]
    counts = summary["counts"]
    latency = summary["latency"]
    retrieval = summary["retrieval"]
    citation = summary["citation"]
    faithfulness = summary["faithfulness_rubric"]
    honesty = summary.get("evaluation_honesty", {}) or {}

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
            non_trivial = sum(1 for name in s.checks if name not in _TRIVIAL_CHECK_NAMES)
            note = "" if non_trivial else " [TRIVIAL CHECKS ONLY]"
            print(
                f"PASSED  {s.case_id}  ({s.elapsed_seconds:.1f}s)  "
                f"[all {num_checks} checks passed]{note}"
            )
        else:
            failed_checks = [name for name, ok in s.checks.items() if not ok]
            print(f"FAILED  {s.case_id}  ({s.elapsed_seconds:.1f}s)  [{', '.join(failed_checks)}]")

    print()
    print("--- Pass/fail ---")
    skipped_count = counts.get("skipped", 0)
    skipped_suffix = f", {skipped_count} skipped" if skipped_count else ""
    print(
        "Summary: "
        f"{counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['errors']} errors{skipped_suffix} out of {counts['total']} total"
    )
    pass_rate = summary.get("pass_rate")
    if pass_rate is None:
        print("Pass rate: n/a")
    else:
        print(f"Pass rate: {pass_rate * 100:.1f}%")
    non_trivial = summary.get("non_trivial_pass_rate")
    nt_passed = counts.get("non_trivial_passed", 0)
    nt_ran = counts.get("non_trivial_ran", 0)
    if non_trivial is None:
        print("Non-trivial pass rate: n/a (no cases carried non-plumbing checks)")
    else:
        print(
            f"Non-trivial pass rate: {non_trivial * 100:.1f}% "
            f"({nt_passed}/{nt_ran} cases with at least one non-plumbing check)"
        )
    trivial_ids = honesty.get("trivial_check_case_ids") or []
    if trivial_ids:
        print(f"Trivial-checks-only cases: {len(trivial_ids)} -> {', '.join(trivial_ids)}")

    print()
    print("--- Evidence retrieval ---")
    cases_with_targets = retrieval.get("cases_with_targets", 0)
    if retrieval["accuracy"] is None:
        print(
            "Recall: n/a (no gold retrieval targets were specified — "
            f"0/{counts.get('total', 0)} ran cases contributed)"
        )
    else:
        print(
            f"Recall: {retrieval['accuracy'] * 100:.1f}% "
            f"({retrieval['matched_count']}/{retrieval['target_count']} targets, "
            f"{cases_with_targets} cases contributed)"
        )
    if retrieval.get("commit_sha_accuracy") is not None:
        print(
            f"Commit SHA match rate: {retrieval['commit_sha_accuracy'] * 100:.1f}% "
            f"({retrieval['commit_sha_matches']}/{retrieval['commit_sha_total']} expected SHAs)"
        )
    must_cases = retrieval.get("must_abstain_cases", 0)
    must_passed = retrieval.get("must_abstain_passed", 0)
    must_precision = retrieval.get("must_abstain_precision")
    if must_precision is None:
        print("Must-abstain precision: n/a (no must-abstain cases)")
    else:
        print(
            f"Must-abstain precision: {must_precision * 100:.1f}% "
            f"({must_passed}/{must_cases} cases returned zero PR/issue noise, "
            f"total stray PRs={retrieval.get('unexpected_prs_total', 0)}, "
            f"stray issues={retrieval.get('unexpected_issues_total', 0)})"
        )

    print()
    print("--- Format compliance (NOT honesty) ---")
    if citation["coverage"] is None:
        print("Citation coverage: n/a (no citable explanation sentences)")
    else:
        print(
            f"Citation coverage: {citation['coverage'] * 100:.1f}% "
            f"({citation['cited_sentence_count']}/{citation['citable_sentence_count']} sentences) "
            "-- near-100% by construction, the prompt enforces it."
        )
    if citation["validity"] is None:
        print("Citation ID validity: n/a (no citations present)")
    else:
        print(
            f"Citation ID validity: {citation['validity'] * 100:.1f}% "
            f"({citation['valid_citation_count']}/{citation['citation_count']} IDs "
            "exist in returned evidence)."
        )

    print()
    print("--- Explanation honesty ---")
    support = citation.get("support_rate")
    if support is None:
        print("Citation semantic support: n/a (no citations present)")
    else:
        print(
            f"Citation semantic support: {support * 100:.1f}% "
            f"({citation['supported_citation_count']}/{citation['citation_count']} "
            "citations share a content word with their sentence — weak signal, not entailment)."
        )
    llm_judge = summary.get("llm_judge")
    if llm_judge is not None and llm_judge.get("scored_count", 0) > 0:
        passing = llm_judge["accurate_count"] + llm_judge["partially_accurate_count"]
        print(
            f"LLM-judge faithfulness (HEADLINE): "
            f"{llm_judge['pass_rate'] * 100:.1f}% pass "
            f"({passing}/{llm_judge['scored_count']} accurate-or-partial), "
            f"strict {llm_judge['strict_pass_rate'] * 100:.1f}% "
            f"({llm_judge['accurate_count']} fully accurate)."
        )
        print(
            f"  breakdown: accurate={llm_judge['accurate_count']}  "
            f"partially_accurate={llm_judge['partially_accurate_count']}  "
            f"hallucinated={llm_judge['hallucinated_count']}  "
            f"unscored={llm_judge['unscored_count']}  skipped={llm_judge['skipped_count']}"
        )
    elif llm_judge is not None:
        # Judge was invoked but every call failed (e.g., daily quota burned).
        # Surface that honestly so the absence of a headline number is not
        # misread as "we did not check".
        print(
            "LLM-judge faithfulness: ATTEMPTED BUT ALL CALLS FAILED. "
            f"Unscored {llm_judge['unscored_count']}, "
            f"skipped {llm_judge['skipped_count']}, "
            f"scored 0. The proxy rubric below is NOT a substitute; re-run "
            "after the judge model's quota resets (or use a different judge)."
        )
    else:
        print(
            "LLM-judge faithfulness: not run. "
            "Re-run with --use-llm-judge to get a headline honesty number; "
            "the proxy rubric below is NOT a substitute."
        )
    if faithfulness["average"] is not None:
        print(
            f"Faithfulness proxy score (PROXY — not ground truth): "
            f"{faithfulness['average']:.2f}/5.00 "
            f"({faithfulness['case_count']}/{faithfulness['case_count'] + faithfulness['unscored_case_count']} cases scored)"
        )
    else:
        print("Faithfulness proxy score: n/a (no case produced a measurable proxy).")

    if latency["average_seconds"] is not None:
        print()
        print("--- Latency ---")
        print(
            f"Latency: avg {latency['average_seconds']:.2f}s | "
            f"p50 {latency['p50_seconds']:.2f}s | "
            f"p95 {latency['p95_seconds']:.2f}s"
        )

    caveats = honesty.get("caveats") or []
    if caveats:
        print()
        print("--- What these numbers do NOT prove ---")
        for caveat in caveats:
            print(f"  - {caveat}")


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
        "--no-llm-judge",
        action="store_true",
        default=False,
        help=(
            "Opt out of the LLM-as-judge pass even if an LLM is available. "
            "Overrides the default-on behavior and implies proxy-only honesty."
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

    # The LLM judge is the headline honesty metric. Run it by default whenever
    # any judge LLM is configured, unless --no-llm-judge or --no-llm is passed.
    # Anthropic Claude is preferred (different model family from the Groq
    # agent); Groq is the fallback. See _judge_backend.
    judge_available = judge_anthropic.is_available() or llm.is_available()
    use_llm_judge = args.use_llm_judge
    if not args.no_llm_judge and not args.no_llm and judge_available:
        use_llm_judge = True
    if args.no_llm_judge:
        use_llm_judge = False
    if use_llm_judge and not judge_available:
        print(
            "WARNING: --use-llm-judge requested but no judge LLM is configured. "
            "Set ANTHROPIC_API_KEY (preferred) or GROQ_API_KEY. "
            "Skipping LLM-judge pass; only the proxy rubric will be reported."
        )
        use_llm_judge = False
    if use_llm_judge:
        if judge_anthropic.is_available():
            print(f"LLM-judge backend: anthropic ({judge_anthropic.model_id()})")
        else:
            print(f"LLM-judge backend: groq ({llm.config.GROQ_MODEL})")

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
            use_llm_judge=use_llm_judge,
        )
        scores.append(score)

    total_elapsed = time.monotonic() - total_start

    summary = summarize_scores(cases, scores, total_elapsed)
    print_report(scores, summary)
    save_results(scores, summary, args.results_file)
    print(f"\nResults written to {args.results_file}")


if __name__ == "__main__":
    main()

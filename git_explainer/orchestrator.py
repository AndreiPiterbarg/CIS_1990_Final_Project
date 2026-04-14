"""Top-level orchestration for the Git explainer agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict

from git_explainer.guardrails import (
    ExplainerQuery,
    should_fetch_file_context,
    validate_query,
)
from git_explainer.llm import LLMUnavailableError, chat, is_available
from git_explainer.memory import ExplainerMemory
from git_explainer.prompts import SYSTEM_PROMPT, build_synthesis_prompt
from git_explainer.tools.file_context_reader import read_file_at_revision
from git_explainer.tools.git_blame_trace import trace_line_history
from git_explainer.tools.github_issue_lookup import (
    extract_issue_refs,
    fetch_issue,
    fetch_issue_comments,
)
from git_explainer.tools.github_pr_lookup import (
    fetch_pr,
    fetch_pr_comments,
    find_prs_for_commit,
)


class ExplanationSections(TypedDict):
    what_changed: str
    why: str
    tradeoffs: str
    limitations: str
    summary: str


class ExplanationResult(TypedDict):
    query: dict[str, Any]
    explanation: ExplanationSections
    commits: list[dict[str, Any]]
    pull_requests: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    file_contexts: list[dict[str, Any]]
    cache_stats: dict[str, int]
    used_fallback: bool


@dataclass(slots=True)
class GitExplainerAgent:
    """Coordinate git tracing, GitHub enrichment, caching, and synthesis."""

    use_llm: bool = True

    def explain(self, query: ExplainerQuery) -> ExplanationResult:
        normalized = validate_query(query)
        memory = ExplainerMemory(normalized.repo_path)

        commits = trace_line_history(
            normalized.repo_path,
            normalized.file_path,
            normalized.start_line,
            normalized.end_line,
            max_count=normalized.max_commits,
        )

        pull_requests, issues, contexts = self._collect_evidence(normalized, commits, memory)
        evidence = {
            "commits": commits,
            "pull_requests": pull_requests,
            "issues": issues,
            "file_contexts": contexts,
        }
        explanation, used_fallback = self._synthesize(normalized, evidence)
        memory.flush()

        return ExplanationResult(
            query=normalized.to_dict(),
            explanation=explanation,
            commits=commits,
            pull_requests=pull_requests,
            issues=issues,
            file_contexts=contexts,
            cache_stats=memory.stats(),
            used_fallback=used_fallback,
        )

    def _collect_evidence(
        self,
        query: ExplainerQuery,
        commits: list[dict[str, Any]],
        memory: ExplainerMemory,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        pull_requests: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        contexts: list[dict[str, Any]] = []
        seen_prs: set[int] = set()
        seen_issues: set[int] = set()

        for commit in commits:
            lookup_sha = commit.get("full_sha", commit["sha"])
            pr_numbers = memory.get_commit_prs(lookup_sha)
            if pr_numbers is None:
                if query.owner and query.repo_name:
                    pr_numbers = find_prs_for_commit(query.owner, query.repo_name, lookup_sha)
                else:
                    pr_numbers = []
                memory.set_commit_prs(lookup_sha, pr_numbers)

            commit_issue_refs = set(extract_issue_refs(commit["message"]))
            pr_bodies: list[str] = []

            for pr_number in pr_numbers:
                if pr_number in seen_prs:
                    cached_pr = memory.get_pr(pr_number)
                    if cached_pr is not None:
                        pr_bodies.append(cached_pr.get("body", ""))
                    continue

                pr_data = memory.get_pr(pr_number)
                if pr_data is None:
                    if query.owner and query.repo_name:
                        pr_data = fetch_pr(query.owner, query.repo_name, pr_number)
                    if pr_data is None:
                        continue
                    memory.set_pr(pr_number, pr_data)

                comments = memory.get_pr_comments(pr_number)
                if comments is None:
                    if query.owner and query.repo_name:
                        comments = fetch_pr_comments(query.owner, query.repo_name, pr_number)
                    else:
                        comments = []
                    memory.set_pr_comments(pr_number, comments)

                pr_record = dict(pr_data)
                pr_record["review_comments"] = comments
                pull_requests.append(pr_record)
                seen_prs.add(pr_number)
                pr_bodies.append(pr_record.get("body", ""))
                commit_issue_refs.update(extract_issue_refs(pr_record.get("body", "")))

            if should_fetch_file_context(commit["message"], "\n".join(pr_bodies)):
                key = self._context_cache_key(query, lookup_sha)
                context_text = memory.get_context(key)
                if context_text is None:
                    start_line = max(1, query.start_line - query.context_radius)
                    end_line = query.end_line + query.context_radius
                    context_text = read_file_at_revision(
                        query.repo_path,
                        query.file_path,
                        revision=lookup_sha,
                        start_line=start_line,
                        end_line=end_line,
                    ) or ""
                    memory.set_context(key, context_text)
                if context_text:
                    contexts.append({
                        "commit_sha": commit["sha"],
                        "file_path": query.file_path,
                        "start_line": max(1, query.start_line - query.context_radius),
                        "end_line": query.end_line + query.context_radius,
                        "content": context_text,
                    })

            for issue_number in sorted(commit_issue_refs):
                if issue_number in seen_issues or not query.owner or not query.repo_name:
                    continue
                issue_data = memory.get_issue(issue_number)
                if issue_data is None:
                    issue_data = fetch_issue(query.owner, query.repo_name, issue_number)
                    if issue_data is None:
                        continue
                    memory.set_issue(issue_number, issue_data)

                comments = memory.get_issue_comments(issue_number)
                if comments is None:
                    comments = fetch_issue_comments(query.owner, query.repo_name, issue_number)
                    memory.set_issue_comments(issue_number, comments)

                issue_record = dict(issue_data)
                issue_record["comments"] = comments
                issues.append(issue_record)
                seen_issues.add(issue_number)

        return pull_requests, issues, contexts

    def _synthesize(
        self,
        query: ExplainerQuery,
        evidence: dict[str, Any],
    ) -> tuple[ExplanationSections, bool]:
        if self.use_llm and is_available():
            try:
                response = chat(
                    build_synthesis_prompt(query.to_dict(), evidence),
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.1,
                )
                parsed = json.loads(response)
                return self._normalize_sections(parsed), False
            except (LLMUnavailableError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        return self._fallback_summary(query, evidence), True

    def _fallback_summary(
        self,
        query: ExplainerQuery,
        evidence: dict[str, Any],
    ) -> ExplanationSections:
        commits = evidence["commits"]
        pull_requests = evidence["pull_requests"]
        issues = evidence["issues"]
        contexts = evidence["file_contexts"]

        commit_citations = " ".join(f"[commit:{c['sha']}]" for c in commits[:3]) or "[commit:none]"
        pr_citations = " ".join(f"[pr:#{pr['number']}]" for pr in pull_requests[:2])
        issue_citations = " ".join(f"[issue:#{issue['number']}]" for issue in issues[:2])

        if commits:
            what_changed = (
                f"The selected lines in {query.file_path}:{query.start_line}-{query.end_line} "
                f"were most recently shaped by {len(commits)} traced commit(s): "
                + "; ".join(f"{c['sha']} ({c['message']})" for c in commits[:3])
                + f". {commit_citations}"
            )
        else:
            what_changed = (
                f"No line-history commits were found for {query.file_path}:{query.start_line}-{query.end_line}. "
                "[commit:none]"
            )

        if pull_requests or issues:
            why_parts: list[str] = []
            if pull_requests:
                why_parts.append(
                    "Related pull requests suggest the intent was "
                    + "; ".join(f"#{pr['number']} ({pr['title']})" for pr in pull_requests[:2])
                )
            if issues:
                why_parts.append(
                    "Linked issues add context from "
                    + "; ".join(f"#{issue['number']} ({issue['title']})" for issue in issues[:2])
                )
            why = ". ".join(why_parts) + ". " + " ".join(x for x in [pr_citations, issue_citations] if x)
        else:
            why = (
                "No linked pull request or issue metadata was found, so the intent can only be inferred "
                f"from commit messages and surrounding code. {commit_citations}"
            )

        if contexts:
            tradeoffs = (
                f"Surrounding file context was fetched for {len(contexts)} commit(s), which suggests the change "
                "needed additional local context beyond the commit message. Explicit trade-offs were not clearly "
                f"documented in the fetched metadata. {commit_citations}"
            )
        else:
            tradeoffs = (
                "The retrieved PRs and issues did not spell out clear trade-offs, so no stronger claim is made "
                f"beyond the available commit and PR evidence. {' '.join(x for x in [commit_citations, pr_citations] if x)}"
            )

        limitations = (
            "This explanation is limited to the traced commits, associated pull requests, linked issues, "
            "and any fetched file context. If a change was discussed elsewhere, it will not appear here. "
            f"{' '.join(x for x in [commit_citations, pr_citations, issue_citations] if x)}"
        )

        summary = f"{what_changed} {why}"
        return ExplanationSections(
            what_changed=what_changed,
            why=why,
            tradeoffs=tradeoffs,
            limitations=limitations,
            summary=summary,
        )

    def _normalize_sections(self, data: dict[str, Any]) -> ExplanationSections:
        return ExplanationSections(
            what_changed=str(data["what_changed"]).strip(),
            why=str(data["why"]).strip(),
            tradeoffs=str(data["tradeoffs"]).strip(),
            limitations=str(data["limitations"]).strip(),
            summary=str(data["summary"]).strip(),
        )

    def _context_cache_key(self, query: ExplainerQuery, sha: str) -> str:
        return (
            f"{sha}:{query.file_path}:{query.start_line}:{query.end_line}:"
            f"{query.context_radius}"
        )


def explain_code_history(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    *,
    owner: str | None = None,
    repo_name: str | None = None,
    max_commits: int = 5,
    context_radius: int = 30,
    enforce_public_repo: bool = False,
    use_llm: bool = True,
) -> ExplanationResult:
    """Convenience wrapper around :class:`GitExplainerAgent`."""
    agent = GitExplainerAgent(use_llm=use_llm)
    query = ExplainerQuery(
        repo_path=repo_path,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        owner=owner,
        repo_name=repo_name,
        max_commits=max_commits,
        context_radius=context_radius,
        enforce_public_repo=enforce_public_repo,
    )
    return agent.explain(query)

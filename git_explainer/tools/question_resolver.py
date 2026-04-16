"""Resolve natural-language questions to relevant code spans."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from git_explainer import config
from git_explainer.guardrails import normalize_file_path
from git_explainer.tools.file_context_reader import read_file_at_revision
from git_explainer.tools.git_utils import run_git

_ALLOWED_SUFFIXES = {
    "",
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
}
_MAX_FILE_LINES = 1200
_MODULE_HEAD_LINES = 40
_FALLBACK_WINDOW = 12
_PREVIEW_LINE_LIMIT = 20
_LIBRARY_HINT_TERMS = {"library", "package", "dependency", "module", "import", "imports"}
_TEST_HINT_TERMS = {"test", "tests", "pytest", "unittest"}
_KEEP_SHORT_TOKENS = {"ai", "api", "db", "id", "llm", "pr", "ui"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "does",
    "explain",
    "for",
    "from",
    "how",
    "i",
    "if",
    "implement",
    "implementation",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "used",
    "using",
    "want",
    "was",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_+-]*")


@dataclass(slots=True)
class ResolvedCodeSpan:
    """A concrete file/line selection derived from a natural-language question."""

    file_path: str
    start_line: int
    end_line: int
    score: float
    matched_terms: list[str]
    preview: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class _QuestionFeatures:
    terms: list[str]
    phrases: list[str]
    wants_library_explanation: bool
    mentions_test_files: bool


@dataclass(slots=True)
class _CandidateSpan:
    file_path: str
    start_line: int
    end_line: int
    text: str
    kind: str
    label: str


def resolve_question_to_code(
    repo_path: str,
    question: str,
    *,
    file_path_hint: str | None = None,
) -> ResolvedCodeSpan:
    """Return the most relevant code span for a natural-language question."""
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir() or not (repo / ".git").exists():
        raise ValueError(f"Not a git repository: {repo}")

    question = question.strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    features = _extract_question_features(question)
    if not features.terms:
        raise ValueError("question must include at least one specific search term")

    file_candidates = _list_files(str(repo), file_path_hint=file_path_hint)

    prioritized_candidates = file_candidates
    fallback_candidates: list[str] = []
    if file_path_hint is None and not features.mentions_test_files:
        prioritized_candidates = [path for path in file_candidates if not _is_test_path(path)]
        fallback_candidates = [path for path in file_candidates if _is_test_path(path)]

    best_span = _find_best_span(str(repo), prioritized_candidates, features)
    if best_span is None and fallback_candidates:
        best_span = _find_best_span(str(repo), fallback_candidates, features)

    if best_span is not None:
        return best_span

    if file_path_hint is not None:
        hint_path = normalize_file_path(str(repo), file_path_hint)
        content = _read_question_file(str(repo), hint_path)
        if content not in (None, "[binary file]"):
            lines = content.splitlines()
            end_line = min(len(lines), _MODULE_HEAD_LINES)
            return ResolvedCodeSpan(
                file_path=hint_path,
                start_line=1,
                end_line=max(1, end_line),
                score=0.0,
                matched_terms=[],
                preview="\n".join(lines[:_PREVIEW_LINE_LIMIT]),
            )

    raise ValueError(f"Could not map question to code: {question!r}")


def _list_files(repo_path: str, *, file_path_hint: str | None = None) -> list[str]:
    if file_path_hint is not None:
        return [normalize_file_path(repo_path, file_path_hint)]

    output = run_git(repo_path, ["ls-files"])
    files: list[str] = []
    for raw_path in output.splitlines():
        path = raw_path.strip()
        if not path:
            continue
        suffix = Path(path).suffix.lower()
        if suffix in _ALLOWED_SUFFIXES:
            files.append(path)
    return files


def _extract_question_features(question: str) -> _QuestionFeatures:
    tokens = [match.group(0).lower() for match in _TOKEN_RE.finditer(question)]

    terms: list[str] = []
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if len(token) <= 2 and token not in _KEEP_SHORT_TOKENS:
            continue
        if token not in terms:
            terms.append(token)

    phrases: list[str] = []
    quoted = re.findall(r"[`'\"]([^`'\"]{3,})[`'\"]", question)
    for phrase in quoted:
        normalized = " ".join(_TOKEN_RE.findall(phrase.lower()))
        if normalized and normalized not in phrases:
            phrases.append(normalized)

    filtered = [token for token in tokens if token not in _STOPWORDS]
    for size in (3, 2):
        for idx in range(len(filtered) - size + 1):
            phrase = " ".join(filtered[idx:idx + size])
            if len(phrase) >= 8 and phrase not in phrases:
                phrases.append(phrase)

    return _QuestionFeatures(
        terms=terms,
        phrases=phrases,
        wants_library_explanation=any(token in _LIBRARY_HINT_TERMS for token in tokens),
        mentions_test_files=any(token in _TEST_HINT_TERMS for token in tokens),
    )


def _find_best_span(
    repo_path: str,
    file_candidates: list[str],
    features: _QuestionFeatures,
) -> ResolvedCodeSpan | None:
    best_span: ResolvedCodeSpan | None = None
    best_key: tuple[float, int, int, str] | None = None

    for file_path in file_candidates:
        content = _read_question_file(repo_path, file_path)
        if content in (None, "[binary file]"):
            continue

        lines = content.splitlines()
        if not lines or len(lines) > _MAX_FILE_LINES:
            continue

        for candidate in _build_candidates(file_path, content, lines):
            score, matched_terms = _score_candidate(candidate, features)
            if score <= 0:
                continue

            preview = "\n".join(lines[candidate.start_line - 1:candidate.end_line][: _PREVIEW_LINE_LIMIT])
            resolved = ResolvedCodeSpan(
                file_path=file_path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                score=round(score, 3),
                matched_terms=matched_terms,
                preview=preview,
            )
            key = (
                resolved.score,
                len(resolved.matched_terms),
                -(resolved.end_line - resolved.start_line),
                resolved.file_path,
            )
            if best_key is None or key > best_key:
                best_span = resolved
                best_key = key

    return best_span


def _read_question_file(repo_path: str, file_path: str) -> str | None:
    """Prefer HEAD so resolved spans remain valid for history tracing."""
    content = read_file_at_revision(repo_path, file_path, revision="HEAD")
    if content is None:
        content = read_file_at_revision(repo_path, file_path)
    return content


def _is_test_path(file_path: str) -> bool:
    path = Path(file_path)
    return "tests" in path.parts or path.name.startswith("test_") or path.stem.endswith("_test")


def _build_candidates(file_path: str, content: str, lines: list[str]) -> list[_CandidateSpan]:
    candidates: list[_CandidateSpan] = []
    seen: set[tuple[int, int, str]] = set()

    def add_candidate(start_line: int, end_line: int, kind: str, label: str) -> None:
        if not lines:
            return
        start_line = max(1, start_line)
        end_line = min(len(lines), end_line)
        if end_line < start_line:
            return
        if end_line - start_line + 1 > config.DEFAULT_MAX_LINE_SPAN:
            end_line = start_line + config.DEFAULT_MAX_LINE_SPAN - 1
        key = (start_line, end_line, kind)
        if key in seen:
            return
        seen.add(key)
        text = "\n".join(lines[start_line - 1:end_line])
        candidates.append(_CandidateSpan(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            text=text,
            kind=kind,
            label=label,
        ))

    add_candidate(1, min(len(lines), _MODULE_HEAD_LINES), "module_head", Path(file_path).name)

    if Path(file_path).suffix == ".py":
        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    end_line = getattr(node, "end_lineno", node.lineno)
                    add_candidate(node.lineno, end_line, type(node).__name__.lower(), node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    end_line = getattr(node, "end_lineno", node.lineno)
                    names: list[str] = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    else:
                        if node.module:
                            names.append(node.module)
                        names.extend(alias.name for alias in node.names)
                    add_candidate(node.lineno, end_line, "import", " ".join(names))

    for start in range(1, len(lines) + 1, _FALLBACK_WINDOW):
        add_candidate(start, start + _FALLBACK_WINDOW - 1, "window", f"window:{start}")

    return candidates


def _score_candidate(candidate: _CandidateSpan, features: _QuestionFeatures) -> tuple[float, list[str]]:
    lowered_text = candidate.text.lower()
    lowered_label = candidate.label.lower()
    lowered_path = candidate.file_path.lower()

    score = 0.0
    matched_terms: list[str] = []

    for phrase in features.phrases:
        phrase_score = 0.0
        if phrase in lowered_text:
            phrase_score += 7.0
        if phrase in lowered_label:
            phrase_score += 5.0
        if phrase in lowered_path:
            phrase_score += 4.0
        if phrase_score:
            score += phrase_score
            for term in phrase.split():
                if term not in matched_terms:
                    matched_terms.append(term)

    for term in features.terms:
        token_score = 0.0
        whole_word = re.findall(rf"\b{re.escape(term)}\b", lowered_text)
        if whole_word:
            token_score += min(len(whole_word), 3) * 3.0
        elif term in lowered_text:
            token_score += 1.5

        if re.search(rf"\b{re.escape(term)}\b", lowered_label):
            token_score += 4.0
        if re.search(rf"\b{re.escape(term)}\b", lowered_path):
            token_score += 3.0

        if token_score:
            score += token_score
            if term not in matched_terms:
                matched_terms.append(term)

    if not matched_terms:
        return 0.0, []

    if candidate.kind == "import" and features.wants_library_explanation:
        score += 5.0
    if candidate.kind == "module_head" and features.wants_library_explanation:
        score += 3.0
    if candidate.kind in {"functiondef", "asyncfunctiondef", "classdef"}:
        score += 2.0

    score += len(matched_terms) * 0.5
    span_length = candidate.end_line - candidate.start_line + 1
    if span_length <= 20:
        score += 1.0
    return score, matched_terms

# Git Explainer Agent

A CLI agent that answers: why does this code exist?

## Overview

Given a local repository and either a line range or a natural-language question, the agent:

- traces relevant commits using `git log -L`
- looks up associated pull requests and linked issues on GitHub
- fetches PR review comments and issue comments
- reads surrounding file context when commit metadata is ambiguous
- resolves a natural-language question (e.g., "Why is requests used here?") into a concrete file and line span before tracing history
- synthesizes a cited explanation using an LLM, with a deterministic fallback when no LLM is configured

Output is a single JSON object containing commits, pull requests, issues, and a structured explanation.

## Prerequisites

- Python 3.11 or later
- `git` installed and on `PATH`

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `openai>=1.12.0`, `requests>=2.31.0`, `python-dotenv>=1.0.0`, `pytest>=7.4.0`.

## Configuration

The agent reads configuration from environment variables. Create a `.env` file in the project root and the agent will load it automatically via `python-dotenv`.

Example `.env`:

```
GROQ_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Required for LLM mode | — | API key for the Groq inference endpoint. If absent, the agent falls back to a deterministic summary. |
| `GITHUB_TOKEN` | Optional | — | Personal access token for the GitHub API. Increases the rate limit from 60 to 5,000 requests/hour. |
| `GROQ_BASE_URL` | Optional | `https://api.groq.com/openai/v1` | Base URL for the OpenAI-compatible inference endpoint. |
| `GROQ_MODEL` | Optional | `llama-3.3-70b-versatile` | Model identifier passed to the inference endpoint. |
| `GIT_EXPLAINER_MAX_LINE_SPAN` | Optional | `200` | Maximum number of lines in a traced span. |
| `GIT_EXPLAINER_MAX_COMMITS` | Optional | `5` | Maximum number of commits to trace. |
| `GIT_EXPLAINER_CONTEXT_RADIUS` | Optional | `30` | Lines of surrounding code to fetch when extra context is needed. |
| `GIT_EXPLAINER_CACHE_FILENAME` | Optional | `.git_explainer_cache.json` | Filename for the local request cache. |
| `GITHUB_RATE_LIMIT_MIN_REMAINING` | Optional | `10` | Threshold for `X-RateLimit-Remaining` below which the client sleeps preemptively until the reset timestamp. |
| `GITHUB_RATE_LIMIT_SLEEP_CAP` | Optional | `60` | Upper bound (in seconds) on any single preemptive rate-limit sleep, so runs cannot stall indefinitely. |
| `EVIDENCE_CHAR_BUDGET` | Optional | `30000` | Serialized-evidence character budget that triggers pre-synthesis condensation when exceeded. |
| `EVIDENCE_FIELD_MAX_CHARS` | Optional | `3000` | Per-field length above which a PR/issue body or comment is eligible for condensation. |
| `EVIDENCE_SUMMARY_TARGET_CHARS` | Optional | `800` | Target length for each LLM-produced per-field summary during condensation. |

## Usage

All modes print a JSON object to stdout.

### Mode 1: Line range

Explain a specific range of lines in a file.

```bash
python main.py /path/to/repo src/app.py 10 25 \
    --owner AndreiPiterbarg \
    --repo-name CIS_1990_Final_Project
```

### Mode 2: Natural-language question

Let the agent locate the relevant code span itself.

```bash
python main.py /path/to/repo \
    --question "Why does the orchestrator fall back to a deterministic summary?" \
    --owner AndreiPiterbarg \
    --repo-name CIS_1990_Final_Project
```

### Mode 3: Question with file hint

Provide a file path to narrow the search when the question is scoped to a specific file.

```bash
python main.py /path/to/repo git_explainer/orchestrator.py \
    --question "Why is the LLM call retried?" \
    --owner AndreiPiterbarg \
    --repo-name CIS_1990_Final_Project
```

### Optional flags

| Flag | Description |
|---|---|
| `--no-llm` | Skip the LLM and use the deterministic fallback summary. |
| `--allow-private-repo` | Opt out of the default public-repo guardrail. With the guardrail enabled (the default), the agent refuses queries against private or unreachable GitHub repositories. |
| `--max-commits N` | Override the maximum number of commits to trace (default: 5). |
| `--context-radius N` | Override the lines of surrounding context to fetch (default: 30). |

#### Public-repo guardrail (default)

The `enforce_public_repo` guardrail is on by default and refuses queries
against private, missing, or otherwise unreachable GitHub repositories
via `ensure_public_github_repo` in `git_explainer/guardrails.py`. To
query a private repository (or an offline/local clone that lacks a
reachable GitHub API endpoint), pass `--allow-private-repo`.

## Output Format

The agent prints a single JSON object with the following top-level fields:

```json
{
  "commits": [...],
  "pull_requests": [...],
  "issues": [...],
  "explanation": {
    "what_changed": "...",
    "why": "...",
    "tradeoffs": "...",
    "limitations": "...",
    "summary": "..."
  },
  "used_fallback": false,
  "resolved_target": { ... }
}
```

| Field | Description |
|---|---|
| `commits` | Array of commit objects. Each has `sha`, `message`, `author`, `date`, and optionally `diff`. |
| `pull_requests` | Array of PR objects. Each has `number`, `title`, `body`, `state`, and `comments`. |
| `issues` | Array of issue objects. Each has `number`, `title`, `body`, `state`, and `comments`. |
| `explanation.what_changed` | Description of the code changes found in the traced commits. |
| `explanation.why` | Reasoning behind the changes, drawn from PR/issue metadata. |
| `explanation.tradeoffs` | Design tradeoffs noted in the discussion or commit history. |
| `explanation.limitations` | Known gaps, TODOs, or caveats mentioned in the history. |
| `explanation.summary` | One- or two-sentence synthesis of the above. |
| `used_fallback` | `true` if no LLM was available and the deterministic summary was used instead. |
| `resolved_target` | Present in question mode. Contains `file_path`, `start_line`, `end_line`, `matched_terms`, and `preview` showing how the question was resolved to a code span. |

## Running Tests

```bash
pytest tests/
```

## Running the Evaluation Harness

Run all benchmark cases:

```bash
python eval/evaluate.py
```

Run only cases that match specific tags:

```bash
python eval/evaluate.py --tags smoke llm
```

Run only specific cases by ID:

```bash
python eval/evaluate.py --ids case_001 case_003
```

Force deterministic mode for all cases (no LLM calls):

```bash
python eval/evaluate.py --no-llm
```

Specify custom benchmark and results file paths:

```bash
python eval/evaluate.py \
    --benchmark-file eval/benchmark.json \
    --results-file eval/results.json
```

The harness prints a pass/fail report plus aggregate benchmark metrics, including retrieval accuracy, citation coverage, citation validity, a rubric-style faithfulness score, and latency percentiles. It writes both the aggregate summary and full per-case scoring details to `eval/results.json`.

#### `--use-llm-judge` flag

Opt-in LLM-as-judge faithfulness scorer. When enabled, the harness sends
each case's explanation plus the evidence that backed it to the
configured LLM and asks for a 3-point rubric rating:
`accurate` / `partially accurate` / `hallucinated`. Usage:

```bash
python eval/evaluate.py --use-llm-judge
```

Cost implications:

- One additional LLM call per scored case (on top of any synthesis
  calls made by the cases with `use_llm: true`).
- Requires an available LLM (for example, a configured `GROQ_API_KEY`).
  When no LLM is available, cases are recorded as `skipped` rather
  than failing the run.
- Deterministic mode: the judge prompt is sent with `temperature=0.0`
  and a retry on parse failure (one retry).

The rubric results are aggregated in the report under `LLM-as-judge
faithfulness` and persisted to `eval/results.json` per case.

#### `expected_commit_shas` benchmark field

Each benchmark case may declare an `expected_commit_shas` list in its
`expected` block. Each entry is a full 40-character SHA treated as
ground truth for retrieval:

```json
"expected": {
  "expected_commit_shas": [
    "0ec7f713d679ceed2c605e62ac5d38d579f29fa0"
  ]
}
```

Matching is strict equality, with a prefix match on either side so
short SHAs surfaced by the agent (7 hex chars) still score against a
full ground-truth SHA. Each matched SHA contributes to the overall
retrieval-accuracy metric alongside `pr_numbers` and
`commit_message_contains` targets. A dedicated `commit_sha_accuracy`
line is also printed in the report when any SHAs are declared.

### GitHub rate-limit handling and ETag caching

GitHub lookups go through a shared client that parses the
`X-RateLimit-Remaining` header on each response. When the remaining
quota drops below `GITHUB_RATE_LIMIT_MIN_REMAINING` (default `10`), the
client sleeps preemptively until the reset timestamp reported by the
API, capped by `GITHUB_RATE_LIMIT_SLEEP_CAP` (default `60` seconds) so
interactive runs and tests cannot hang indefinitely. After the sleep
the request retries once.

Conditional requests are cached by ETag: successful PR and issue
fetches store the response's `ETag` in `ExplainerMemory` (the
JSON-backed cache at `.git_explainer_cache.json`). Subsequent fetches
send `If-None-Match` and, on a `304 Not Modified`, reuse the cached
body without counting against the rate limit. Cache entries are keyed
by shape (commit-PRs, PRs, PR comments, issues, issue comments, file
context, diffs) and flushed to disk at the end of each explain run.

### Evidence pre-summarization

Long pull-request threads and issue discussions can push the synthesis
prompt past the model's context window. Before the explanation is
synthesized, the orchestrator measures the serialized evidence payload
and — when it exceeds `EVIDENCE_CHAR_BUDGET` (default `30000`) —
condenses the largest eligible free-form fields using a two-tier
strategy:

1. Tier 1 (preferred): ask the configured LLM to summarize the field,
   preserving citation anchors (commit SHAs, PR/issue numbers, file
   paths, titles) and stated intent. Target length is
   `EVIDENCE_SUMMARY_TARGET_CHARS` (default `800` chars).
2. Tier 2 (fallback): deterministic head+tail truncation with a
   visible elision marker. Used when the LLM is unavailable or its
   response is empty.

Only PR bodies, PR review comments, issue bodies, and issue comments
are eligible, and only when their length exceeds
`EVIDENCE_FIELD_MAX_CHARS` (default `3000` chars). Citation-relevant
metadata — commit SHAs, PR/issue numbers, titles, labels, URLs, file
contexts, and diffs — is always preserved verbatim.

Important: the `ExplanationResult` returned to the caller still
contains the un-condensed originals. Only the synthesis LLM sees the
condensed view. The result includes a `condensation` field describing
what happened:

```json
"condensation": {
  "original_size": 48123,
  "condensed_size": 22041,
  "fields_condensed": ["pr#42.body", "issue#7.comments[2].body"],
  "method_used": "mixed"
}
```

`method_used` is one of `"none"`, `"llm"`, `"heuristic"`, or `"mixed"`.

## Notebook Workbench

For an interactive notebook under `eval/`, install the optional notebook dependencies and register the project-local kernel:

```bash
.venv/bin/python eval/setup_notebook.py
```

Then open:

```bash
.venv/bin/jupyter notebook eval/git_explainer_eval_workbench.ipynb
```

The notebook includes:

- a single-query agent sandbox for trying out `explain_code_history`
- a configurable benchmark runner with an offline-friendly smoke setup
- plots for pass/fail counts, retrieval accuracy, citation coverage, citation validity, faithfulness, and latency

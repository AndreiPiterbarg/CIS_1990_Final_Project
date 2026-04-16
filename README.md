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
| `--enforce-public-repo` | Abort if the GitHub repository is private or unreachable. |
| `--max-commits N` | Override the maximum number of commits to trace (default: 5). |
| `--context-radius N` | Override the lines of surrounding context to fetch (default: 30). |

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

The harness prints a pass/fail report to stdout and writes full scoring details to `eval/results.json`.

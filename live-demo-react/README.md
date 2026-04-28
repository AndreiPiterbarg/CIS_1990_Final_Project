# React Public-Repo Live Demo

This is a separate presentation sandbox for running Git Explainer against the public [`facebook/react`](https://github.com/facebook/react) repository. It does not modify or import the existing `live-demo/` kit.

The runner keeps generated files inside this folder:

- `live-demo-react/repos/react/` for the React clone
- `live-demo-react/.cache/` for Git Explainer cache files
- `live-demo-react/logs/` for captured demo output

## Quick Start

From the project root:

```bash
python3 live-demo-react/run_react_demo.py check
python3 live-demo-react/run_react_demo.py prepare
python3 live-demo-react/run_react_demo.py no-llm hooks-use-transition
```

For the full agentic path with live model calls:

```bash
python3 live-demo-react/run_react_demo.py live hooks-use-transition
```

`prepare` clones `https://github.com/facebook/react.git` into `live-demo-react/repos/react` if needed. If you already have a local React checkout, use:

```bash
python3 live-demo-react/run_react_demo.py --repo-path /path/to/react check
python3 live-demo-react/run_react_demo.py --repo-path /path/to/react live hooks-use-transition
```

## Useful Commands

List the curated presets:

```bash
python3 live-demo-react/run_react_demo.py list
```

Print the exact `main.py` command without running it:

```bash
python3 live-demo-react/run_react_demo.py command hooks-use-transition
```

Run without LLM synthesis, while still using the public React repository and GitHub metadata:

```bash
python3 live-demo-react/run_react_demo.py no-llm hooks-use-state
```

Run the planner and critic path:

```bash
python3 live-demo-react/run_react_demo.py live hooks-use-state
```

## Environment

The `no-llm` mode does not require model keys. The `live` mode uses the same environment variables as the main project:

```bash
GROQ_API_KEY=...
ANTHROPIC_API_KEY=...
GITHUB_TOKEN=...
```

`GITHUB_TOKEN` is optional, but strongly recommended for the React repository because public unauthenticated GitHub API limits are low.

The runner loads `.env` from the project root when `python-dotenv` is installed and sets:

```bash
GIT_EXPLAINER_CACHE_FILENAME=<absolute path inside live-demo-react/.cache>
GITHUB_RATE_LIMIT_SLEEP_CAP=5
```

## How It Stays Current

The query presets use stable file paths plus an anchor string such as `export function useTransition`. At runtime the runner reads the current React checkout, finds the matching line, and passes the resolved span to `main.py`. That keeps the demo from breaking just because upstream React moved a function up or down in the file.

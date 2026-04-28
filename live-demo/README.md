# Live Demo Kit

This folder is a presentation sandbox for the Git Explainer Agent. It is designed to leave the existing codebase untouched: demo logs and cache files are written under `live-demo/`.

## Quick Start

From the project root:

```bash
python3 live-demo/run_demo.py check
python3 live-demo/run_demo.py scripted --scenario both
```

The scripted demo is the safest presentation path. It runs the production planner, dispatcher, synthesizer validation, critic loop, git tools, and evidence merging, but uses recorded LLM replies and mocked GitHub responses so the presentation is not dependent on API keys, network latency, or rate limits.

When run in an interactive terminal, the scripted walkthrough pauses after each visible step. Press Enter to move to the next planner/tool/synthesizer/critic step. Add `--no-pause` if you want it to run straight through.

For a more technical version of the old demo's prompt tracing, add `--verbose-prompts`. The default view stays compact for live narration.

## Best Demo Flow

1. Run the setup check:

```bash
python3 live-demo/run_demo.py check
```

2. Run the step-by-step agentic demo:

```bash
python3 live-demo/run_demo.py scripted --scenario both
```

Non-pausing version for smoke tests or log generation:

```bash
python3 live-demo/run_demo.py scripted --scenario both --no-pause
```

Verbose technical walkthrough:

```bash
python3 live-demo/run_demo.py scripted --scenario 1 --verbose-prompts
```

3. If you want a real API moment, use only the critic live:

```bash
python3 live-demo/run_demo.py scripted --scenario 2 --live-critic
```

4. If you want a fully deterministic CLI example after the main demo:

```bash
python3 live-demo/run_demo.py safe natural-config
```

5. If you want the risky full live path:

```bash
python3 live-demo/run_demo.py agentic config-switch
```

The full live path uses the real Groq planner/synthesizer and Anthropic critic when keys are configured. It may fail or fall back if keys, quotas, or network access are unavailable.

## Files

- `run_demo.py`: presentation runner, with cache/log isolation.
- `agent_walkthrough.py`: step-by-step walkthrough used by `run_demo.py scripted`.
- `demo_fixtures.py`: scripted LLM replies, mocked GitHub data, and printing helpers.
- `sample_queries.json`: curated line-range and natural-language queries.
- `prompts.md`: prompt bank with what each prompt demonstrates.
- `presenter_notes.md`: short talk track for a 5-8 minute presentation.
- `.gitignore`: keeps generated demo logs/cache out of version control.

## Environment

The safe scripted demo does not require API keys. The optional live modes use:

```bash
GROQ_API_KEY=...
ANTHROPIC_API_KEY=...
GITHUB_TOKEN=...
```

`GITHUB_TOKEN` is optional but helps avoid low unauthenticated rate limits. The runner also sets:

```bash
GIT_EXPLAINER_CACHE_FILENAME=<absolute path inside live-demo/.cache>
```

so generated cache data stays inside this folder.

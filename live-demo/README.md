# Live Demo Kit

This folder is a presentation sandbox for the Git Explainer Agent. It is designed to leave the existing codebase untouched: demo logs and cache files are written under `live-demo/`.

## Quick Start

From the project root:

```bash
python3 live-demo/run_demo.py check
python3 live-demo/run_demo.py live
```

The step-by-step walkthrough uses one natural-language question: "How does the agent recover when the first explanation is missing evidence?" It resolves that question to the Planner/Critic orchestration code, then uses real Groq LLM responses for the planner and synthesizer while preserving the presentation formatting and Enter-to-continue flow. GitHub responses stay on demo fixtures so the history story is stable; the Anthropic critic runs when configured and is skipped by production code when unavailable. The old `scripted` command name is still accepted as an alias.

When run in an interactive terminal, the walkthrough pauses after each visible step. Press Enter to move to the next planner/tool/synthesizer/critic step. Add `--no-pause` if you want it to run straight through.

Planner turns show the `Evidence collected so far` section in full without printing the bulky tool schema around it. Critic prompts are shown in full, and the critic step also prints the raw reply or a structured skipped report when Anthropic is unavailable. For a more technical version of the rest of the prompt tracing, add `--verbose-prompts`.

## Best Demo Flow

1. Run the setup check:

```bash
python3 live-demo/run_demo.py check
```

2. Run the step-by-step agentic demo:

```bash
python3 live-demo/run_demo.py live
```

Non-pausing version for smoke tests or log generation:

```bash
python3 live-demo/run_demo.py live --no-pause
```

Verbose technical walkthrough:

```bash
python3 live-demo/run_demo.py live --verbose-prompts
```

3. The old `--live-critic` flag is still accepted for older commands, but the critic is already live whenever Anthropic is configured:

```bash
python3 live-demo/run_demo.py live --live-critic
```

4. If you want a fully deterministic CLI example after the main demo:

```bash
python3 live-demo/run_demo.py safe agent-recovery-question
```

5. If you want the full live CLI path, including live GitHub calls:

```bash
python3 live-demo/run_demo.py agentic config-switch
```

The full live CLI path uses the real Groq planner/synthesizer, real GitHub calls, and Anthropic critic when keys are configured. It may fail or fall back if keys, quotas, or network access are unavailable.

## Files

- `run_demo.py`: presentation runner, with cache/log isolation.
- `agent_walkthrough.py`: step-by-step walkthrough used by `run_demo.py live`.
- `demo_fixtures.py`: mocked GitHub data, printing helpers, and legacy scripted LLM replies kept for reference.
- `sample_queries.json`: curated line-range and natural-language queries.
- `prompts.md`: prompt bank with what each prompt demonstrates.
- `presenter_notes.md`: short talk track for a 5-8 minute presentation.
- `.gitignore`: keeps generated demo logs/cache out of version control.

## Environment

The safe deterministic demos do not require API keys. The step-by-step live walkthrough uses:

```bash
GROQ_API_KEY=...
ANTHROPIC_API_KEY=...
GITHUB_TOKEN=...
GROQ_MODEL=openai/gpt-oss-120b
```

`GITHUB_TOKEN` is optional but helps avoid low unauthenticated rate limits. The runner also sets:

```bash
GIT_EXPLAINER_CACHE_FILENAME=<absolute path inside live-demo/.cache>
```

so generated cache data stays inside this folder.

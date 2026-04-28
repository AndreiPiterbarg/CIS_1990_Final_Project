# Prompt Bank

Use these prompts in this order for a clean presentation. The first command is the best live-demo path because it shows the whole agent step by step without relying on external services.

## 1. Main Step-by-Step Demo

```bash
python3 live-demo/run_demo.py scripted --scenario both
```

What it demonstrates:

- Pauses after each visible step so you can narrate before continuing.
- Keeps prompts compact by default, with `--verbose-prompts` available for a deeper technical walkthrough.
- Planner chooses deterministic tools instead of directly answering.
- Tool dispatcher gathers commits, PRs, diffs, and file context.
- Synthesizer writes cited explanation sections.
- Critic can accept a good answer or request more evidence.
- Re-plan/re-synthesize loop improves a thin answer.

Good thing to say:

> The LLM is not allowed to execute arbitrary commands. It only chooses from a small registry of tested tools, and every tool result becomes auditable evidence.

## 2. Natural-Language Query

```bash
python3 live-demo/run_demo.py safe natural-config
```

Prompt:

```text
Why does config use os.getenv for GROQ_API_KEY and GITHUB_TOKEN instead of requiring API keys at import time?
```

What it demonstrates:

- Users do not need exact line numbers.
- The question resolver maps terms like `os.getenv`, `GROQ_API_KEY`, and `GITHUB_TOKEN` to `git_explainer/config.py`.
- The final JSON includes `resolved_target`, so the mapping is visible.

## 3. Critic-Friendly Bug-Fix Case

```bash
python3 live-demo/run_demo.py safe file-reader-revision-fix
```

Prompt:

```text
Why does _read_from_revision return None when git show fails instead of raising an exception?
```

What it demonstrates:

- Sparse metadata case with no linked PR or issue.
- The agent still returns useful local evidence from commits and diffs.
- Good transition into the critic story from the scripted demo.

## 4. Small Bug-Fix Backup

```bash
python3 live-demo/run_demo.py safe line-range-slice-fix
```

Prompt:

```text
Why did the file context reader need a line range slicing fix?
```

What it demonstrates:

- Short and fast direct line-range history.
- Good fallback if you are low on time.
- Easy to explain because the bug is concrete.

## 5. Tool Design Prompt

```bash
python3 live-demo/run_demo.py safe issue-regex
```

Prompt:

```text
Why does the GitHub issue lookup tool use a regex to extract issue references?
```

What it demonstrates:

- Small deterministic tools are easier to validate than broad agent powers.
- The agent can explain helper code, not just application-facing code.

## Optional Live API Prompts

Use this if you have API keys configured and want one real model call in the room:

```bash
python3 live-demo/run_demo.py scripted --scenario 2 --live-critic
```

Use `--no-pause` on any `scripted` command if you want to generate a complete log without pressing Enter.

Use this if someone asks to see more of the actual prompts being sent through the agent:

```bash
python3 live-demo/run_demo.py scripted --scenario 1 --verbose-prompts
```

Use this only if you are comfortable with network/rate-limit risk:

```bash
python3 live-demo/run_demo.py agentic config-switch --skip-public-check
```

The scripted demo is the one to trust during the presentation. The optional live paths are good for Q&A or a backup recording, not for the core flow.

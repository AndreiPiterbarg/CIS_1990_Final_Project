# Prompt Bank

Use these prompts in this order for a clean presentation. The first command is the best live-demo path because it shows the whole agent step by step with real Groq planner/synthesizer responses while keeping GitHub evidence on stable demo fixtures.

## 1. Main Step-by-Step Demo

```bash
python3 live-demo/run_demo.py live
```

What it demonstrates:

- Pauses after each visible step so you can narrate before continuing.
- Starts from a single natural-language question instead of hand-picked scenario choices.
- Displays the original user query before the agent begins.
- Shows the resolved target in `git_explainer/orchestrator.py` before history tracing continues.
- Shows each planner turn's evidence section in full, plus the full critic prompt and critic reply or skipped report.
- Planner chooses deterministic tools instead of directly answering.
- Tool dispatcher gathers commits, PRs, diffs, and file context.
- Synthesizer writes cited explanation sections.
- Critic can accept a good answer or request more evidence.
- Re-plan/re-synthesize loop improves a thin answer.

Good thing to say:

> The LLM is not allowed to execute arbitrary commands. It only chooses from a small registry of tested tools, and every tool result becomes auditable evidence.

## 2. Deterministic Question Backup

```bash
python3 live-demo/run_demo.py safe agent-recovery-question
```

Prompt:

```text
How does the agent recover when the first explanation is missing evidence?
```

What it demonstrates:

- Users do not need exact line numbers.
- The question resolver maps the prompt to the Planner/Critic control flow in `git_explainer/orchestrator.py`.
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
- Good transition into the critic story from the step-by-step demo.

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

## Optional Variants

The old `--live-critic` flag is still accepted for older commands; the critic is already live whenever Anthropic is configured:

```bash
python3 live-demo/run_demo.py live --live-critic
```

Use `--no-pause` on any `live` command if you want to generate a complete log without pressing Enter.

Use this if someone asks to see more of the actual prompts being sent through the agent:

```bash
python3 live-demo/run_demo.py live --verbose-prompts
```

Use this only if you are comfortable with network/rate-limit risk:

```bash
python3 live-demo/run_demo.py agentic config-switch --skip-public-check
```

Use the `safe` commands when you need deterministic no-API backups. Use the full `agentic` command only when you are comfortable with live GitHub/network/rate-limit risk.

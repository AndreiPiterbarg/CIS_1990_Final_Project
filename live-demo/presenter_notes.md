# Presenter Notes

This is a compact 5-8 minute talk track.

## Before Presenting

Run:

```bash
python3 live-demo/run_demo.py check
python3 live-demo/run_demo.py live
```

Keep a terminal open at the project root. The step-by-step walkthrough pauses after each visible step; press Enter when you are ready to continue. Planner and synthesizer replies are live Groq responses, while GitHub data is fixed to demo fixtures. If the room network is unstable, use the `safe` commands as deterministic backup.

If someone asks what the prompts look like under the hood, run `python3 live-demo/run_demo.py live --verbose-prompts`.

## Opening

"The agent answers a very specific developer question: why does this code exist? The hard part is that the answer usually lives across local git history, PRs, issues, review comments, and the surrounding code. So the agent is designed to retrieve evidence first, then write a cited explanation."

## Main Demo: Question To Evidence

Command:

```bash
python3 live-demo/run_demo.py live
```

Point out:

- The user asks: "How does the agent recover when the first explanation is missing evidence?"
- The agent resolves the question to a concrete span in `git_explainer/orchestrator.py`.
- The demo prints the original user query before the agent begins.
- The planner starts with commits and chooses GitHub/git tool lookups.
- The planner display keeps the evidence section complete without dumping the full tool schema every turn.
- The tool results are not hidden; they become returned evidence.
- The critic prompt and response are printed as their own visible step after synthesis.
- The final answer has sections: `what_changed`, `why`, `tradeoffs`, `limitations`, `summary`.
- Claims cite `[commit:...]`, and the limitations section is explicit when PR or issue evidence is absent.

Good line:

"The critic is not there to make the prose prettier. It is there to ask whether the evidence is strong enough for the claims."

## Deterministic Backup

Command:

```bash
python3 live-demo/run_demo.py safe agent-recovery-question
```

Point out:

- This is deterministic and local.
- It uses the same question as the live walkthrough.
- `resolved_target` shows exactly what span the agent chose.

## Close

"The design tradeoff is deliberate: narrow deterministic tools, local cache, guardrails before retrieval, and fallbacks when models or metadata are unavailable. The result may be less magical than a general chatbot, but it is much easier to audit."

## If Something Fails

- If live API keys fail, use `python3 live-demo/run_demo.py safe agent-recovery-question`.
- If output is too long, run `python3 live-demo/run_demo.py live --no-pause` once beforehand and narrate from the saved log.
- If asked for a deterministic proof of question mode, run `python3 live-demo/run_demo.py safe agent-recovery-question --json`.
- Generated files live under `live-demo/logs` and `live-demo/.cache`.

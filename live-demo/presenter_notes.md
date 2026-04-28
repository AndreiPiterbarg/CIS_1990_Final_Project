# Presenter Notes

This is a compact 5-8 minute talk track.

## Before Presenting

Run:

```bash
python3 live-demo/run_demo.py check
python3 live-demo/run_demo.py scripted --scenario both
```

Keep a terminal open at the project root. The scripted walkthrough pauses after each visible step; press Enter when you are ready to continue. If the room network is unstable, use only `scripted` and `safe` commands.

If someone asks what the prompts look like under the hood, run `python3 live-demo/run_demo.py scripted --scenario 1 --verbose-prompts`.

## Opening

"The agent answers a very specific developer question: why does this code exist? The hard part is that the answer usually lives across local git history, PRs, issues, review comments, and the surrounding code. So the agent is designed to retrieve evidence first, then write a cited explanation."

## Demo 1: Happy Path

Command:

```bash
python3 live-demo/run_demo.py scripted --scenario 1
```

Point out:

- The selected code is `git_explainer/config.py:13-19`.
- The planner starts with commits and chooses GitHub/tool lookups.
- The tool results are not hidden; they become returned evidence.
- The final answer has sections: `what_changed`, `why`, `tradeoffs`, `limitations`, `summary`.
- Claims cite `[commit:...]` and `[pr:#1]`.

Good line:

"The model is not guessing from vibes. It has to ground its explanation in returned evidence."

## Demo 2: Critic Loop

Command:

```bash
python3 live-demo/run_demo.py scripted --scenario 2
```

Point out:

- This case has no PR, so the first draft is weaker.
- The critic says the answer is too thin and asks for surrounding file context.
- The planner calls `read_file_at_revision`.
- The second synthesis becomes more specific.

Good line:

"The critic is not there to make the prose prettier. It is there to ask whether the evidence is strong enough for the claims."

## Natural-Language Follow-Up

Command:

```bash
python3 live-demo/run_demo.py safe natural-config
```

Point out:

- This is deterministic and local.
- The user gives a question instead of line numbers.
- `resolved_target` shows exactly what span the agent chose.

## Close

"The design tradeoff is deliberate: narrow deterministic tools, local cache, guardrails before retrieval, and fallbacks when models or metadata are unavailable. The result may be less magical than a general chatbot, but it is much easier to audit."

## If Something Fails

- If live API keys fail, use `python3 live-demo/run_demo.py scripted --scenario both`.
- If output is too long, run only `--scenario 1`.
- If asked for a deterministic proof, run `python3 live-demo/run_demo.py safe natural-config --json`.
- Generated files live under `live-demo/logs` and `live-demo/.cache`.

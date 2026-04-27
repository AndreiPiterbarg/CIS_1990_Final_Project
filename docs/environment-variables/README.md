# Environment Variables

The Git Explainer Agent reads configuration from environment variables. Put
these values in a `.env` file at the project root, or export them in your shell
before running the CLI.

## `.env` template

```bash
# Required for LLM mode. Leave unset to use deterministic fallback summaries.
GROQ_API_KEY=

# Optional, but recommended for GitHub API rate limits.
GITHUB_TOKEN=

# Optional, only needed for --critic.
ANTHROPIC_API_KEY=
```

## Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Required for LLM mode | - | API key for the Groq inference endpoint. If absent, the agent falls back to a deterministic summary. |
| `GITHUB_TOKEN` | Optional | - | Personal access token for the GitHub API. Increases the rate limit from 60 to 5,000 requests/hour. |
| `GROQ_BASE_URL` | Optional | `https://api.groq.com/openai/v1` | Base URL for the OpenAI-compatible inference endpoint. |
| `GROQ_MODEL` | Optional | `llama-3.3-70b-versatile` | Model identifier passed to the inference endpoint. |
| `GROQ_MAX_TOKENS` | Optional | `4096` | Maximum tokens for the synthesis LLM response. |
| `GITHUB_API_BASE` | Optional | `https://api.github.com` | GitHub API base URL. Override only for compatible GitHub API deployments. |
| `GIT_EXPLAINER_MAX_LINE_SPAN` | Optional | `200` | Maximum number of lines in a traced span. |
| `GIT_EXPLAINER_MAX_COMMITS` | Optional | `5` | Maximum number of commits to trace. |
| `GIT_EXPLAINER_CONTEXT_RADIUS` | Optional | `30` | Lines of surrounding code to fetch when extra context is needed. |
| `GIT_EXPLAINER_CACHE_FILENAME` | Optional | `.git_explainer_cache.json` | Filename for the local request cache. |
| `GITHUB_RATE_LIMIT_MIN_REMAINING` | Optional | `10` | Threshold for `X-RateLimit-Remaining` below which the client sleeps preemptively until the reset timestamp. |
| `GITHUB_RATE_LIMIT_SLEEP_CAP` | Optional | `60` | Upper bound in seconds on any single preemptive rate-limit sleep, so runs cannot stall indefinitely. |
| `EVIDENCE_CHAR_BUDGET` | Optional | `30000` | Serialized-evidence character budget that triggers pre-synthesis condensation when exceeded. |
| `EVIDENCE_FIELD_MAX_CHARS` | Optional | `3000` | Per-field length above which a PR/issue body or comment is eligible for condensation. |
| `EVIDENCE_SUMMARY_TARGET_CHARS` | Optional | `800` | Target length for each LLM-produced per-field summary during condensation. |
| `PLANNER_MAX_ITERATIONS` | Optional | `10` | Maximum Planner LLM tool-routing turns when `--planner` is enabled. |
| `PLANNER_MODEL` | Optional | same as `GROQ_MODEL` | Model identifier for Planner LLM calls. |
| `PLANNER_MAX_TOKENS` | Optional | `1024` | Maximum tokens for each Planner response. |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_KEY` | Required for `--critic` | - | Anthropic API key for the optional Critic LLM. The critic is skipped if the key or SDK is unavailable. |
| `CRITIC_MODEL` | Optional | `claude-haiku-4-5` | Model identifier for the optional Critic LLM. |
| `CRITIC_MAX_TOKENS` | Optional | `600` | Maximum tokens for each Critic response. |

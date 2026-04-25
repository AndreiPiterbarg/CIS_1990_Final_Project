"""Configuration helpers for the Git explainer agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Environment-backed credentials. We keep module import side effects light so
# tests and offline workflows can still import the package.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq / OpenAI-compatible endpoint
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "4096"))

# GitHub API
GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")
# When the observed X-RateLimit-Remaining drops below this threshold we sleep
# preemptively until the reset timestamp. GitHub resets hourly, but each
# individual preemptive sleep is also clamped (see GITHUB_RATE_LIMIT_SLEEP_CAP).
GITHUB_RATE_LIMIT_MIN_REMAINING = int(
    os.getenv("GITHUB_RATE_LIMIT_MIN_REMAINING", "10")
)
# Upper bound (seconds) on any single preemptive rate-limit sleep. Kept small
# so tests and interactive runs don't hang indefinitely.
GITHUB_RATE_LIMIT_SLEEP_CAP = int(
    os.getenv("GITHUB_RATE_LIMIT_SLEEP_CAP", "60")
)

# Agent defaults from the PRD
DEFAULT_MAX_LINE_SPAN = int(os.getenv("GIT_EXPLAINER_MAX_LINE_SPAN", "200"))
DEFAULT_MAX_COMMITS = int(os.getenv("GIT_EXPLAINER_MAX_COMMITS", "5"))
DEFAULT_CONTEXT_RADIUS = int(os.getenv("GIT_EXPLAINER_CONTEXT_RADIUS", "30"))
CACHE_FILENAME = os.getenv("GIT_EXPLAINER_CACHE_FILENAME", ".git_explainer_cache.json")

# Evidence condensation thresholds. When the serialized evidence exceeds
# EVIDENCE_CHAR_BUDGET we try to shrink long-form fields (PR/issue bodies and
# their comment threads) so they fit in the synthesis model's context. Short
# fields are left alone. See git_explainer/evidence_condenser.py.
EVIDENCE_CHAR_BUDGET = int(os.getenv("EVIDENCE_CHAR_BUDGET", "30000"))
EVIDENCE_FIELD_MAX_CHARS = int(os.getenv("EVIDENCE_FIELD_MAX_CHARS", "3000"))
EVIDENCE_SUMMARY_TARGET_CHARS = int(os.getenv("EVIDENCE_SUMMARY_TARGET_CHARS", "800"))


# ---------------------------------------------------------------------------
# Planner + Critic (agentic LLM layers)
# ---------------------------------------------------------------------------
# When the Planner is enabled, an LLM decides which deterministic GitHub /
# git tools to invoke (and in what order) instead of the orchestrator's
# fixed sequence. The Critic is an independent LLM that grades the
# synthesized explanation against the gathered evidence; if it flags gaps
# the Planner is invoked once more before re-synthesis.
#
# Both layers are opt-in: callers must explicitly enable them via the
# ``GitExplainerAgent`` constructor or the ``--planner`` / ``--critic`` CLI
# flags. The defaults below only set ceilings, not policy.
PLANNER_MAX_ITERATIONS = int(os.getenv("PLANNER_MAX_ITERATIONS", "10"))
PLANNER_MODEL = os.getenv("PLANNER_MODEL", GROQ_MODEL)
PLANNER_MAX_TOKENS = int(os.getenv("PLANNER_MAX_TOKENS", "1024"))

CRITIC_MAX_ROUNDS = int(os.getenv("CRITIC_MAX_ROUNDS", "1"))
# We use a Claude family model for the critic so the second LLM in the
# pipeline is genuinely independent from the synthesis LLM (Groq llama).
# This mirrors the eval harness's judge_anthropic.py choice.
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "claude-haiku-4-5")
CRITIC_MAX_TOKENS = int(os.getenv("CRITIC_MAX_TOKENS", "600"))


def github_headers(*, accept: str = "application/vnd.github.v3+json") -> dict[str, str]:
    """Return GitHub headers with the configured token, if any."""
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": accept,
    }


def has_github_token() -> bool:
    return bool(GITHUB_TOKEN.strip())


def has_groq_api_key() -> bool:
    return bool(GROQ_API_KEY.strip())

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

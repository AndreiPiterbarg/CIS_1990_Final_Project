"""Shared HTTP helper for GitHub REST API calls.

Centralizes:
- Exponential backoff on 429 / 5xx responses (retained from the previous
  per-module ``_get`` helpers).
- Rate-limit awareness via ``X-RateLimit-*`` headers. We track the most recent
  observed values in a module-level :data:`_RATE_LIMIT_STATE` so callers can
  preemptively sleep before we hit the limit.
- ETag-based conditional requests: when an :class:`ExplainerMemory` instance
  is provided, we store the response ``ETag`` per URL and send
  ``If-None-Match`` on subsequent calls. A 304 response returns the cached
  data without re-parsing.

The public entry point is :func:`github_get_json`, which returns a
:class:`GitHubResponse` dataclass.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from git_explainer.config import (
    GITHUB_RATE_LIMIT_MIN_REMAINING,
    GITHUB_RATE_LIMIT_SLEEP_CAP,
    github_headers,
)

# Module-level rate-limit state. Populated from X-RateLimit-* response headers
# on every successful GitHub call. Callers check this via get_rate_limit_state()
# or indirectly: _maybe_preemptive_sleep() does it for us before each request.
_RATE_LIMIT_STATE: dict[str, Any] = {
    "limit": None,       # int | None
    "remaining": None,   # int | None
    "reset": None,       # unix ts (int) | None
    "last_updated": None,
}

# Buffer added on top of the computed (reset - now) sleep to avoid racing the
# server clock.
_RATE_LIMIT_SLEEP_BUFFER = 1.0


@dataclass(slots=True)
class GitHubResponse:
    """Normalized result from :func:`github_get_json`.

    - ``data``: parsed JSON body (any JSON type) or ``None`` on 404.
    - ``status_code``: the HTTP status code actually returned (or 200 for a
      synthesized "304-served-from-cache" response so downstream code that
      only cares about status sees success).
    - ``headers``: the response headers dict (empty for cache hits with no
      fresh response).
    - ``from_cache``: True when the data was served from the ETag cache via
      a 304 Not Modified response.
    """

    data: Any
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    from_cache: bool = False


def _safe_headers_dict(headers: Any) -> dict[str, str]:
    """Best-effort conversion of a response.headers attribute to a plain dict.

    Real ``requests.Response`` objects expose a CaseInsensitiveDict here, but
    in tests callers frequently pass plain :class:`MagicMock` which can't be
    iterated. Fall back to an empty dict in that case.
    """
    if not headers:
        return {}
    try:
        return dict(headers)
    except (TypeError, ValueError):
        return {}


def get_rate_limit_state() -> dict[str, Any]:
    """Return a shallow copy of the current observed rate-limit state.

    Exposed for testability and for callers that want to surface
    rate-limit info to users/logs.
    """
    return dict(_RATE_LIMIT_STATE)


def _reset_rate_limit_state_for_tests() -> None:
    """Test helper: clear observed rate-limit state."""
    _RATE_LIMIT_STATE.update(
        {"limit": None, "remaining": None, "reset": None, "last_updated": None}
    )


def _parse_rate_limit_headers(headers: dict[str, str] | Any) -> None:
    """Update :data:`_RATE_LIMIT_STATE` from response headers, if present."""
    if not headers:
        return
    # requests' CaseInsensitiveDict supports .get; a plain dict does too.
    getter = headers.get if hasattr(headers, "get") else lambda k, d=None: d

    def _as_int(val: Any) -> int | None:
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    limit = _as_int(getter("X-RateLimit-Limit"))
    remaining = _as_int(getter("X-RateLimit-Remaining"))
    reset = _as_int(getter("X-RateLimit-Reset"))

    # Only update fields we actually saw so a 304 without rate-limit headers
    # doesn't wipe known state.
    if limit is not None:
        _RATE_LIMIT_STATE["limit"] = limit
    if remaining is not None:
        _RATE_LIMIT_STATE["remaining"] = remaining
    if reset is not None:
        _RATE_LIMIT_STATE["reset"] = reset
    if limit is not None or remaining is not None or reset is not None:
        _RATE_LIMIT_STATE["last_updated"] = time.time()


def _maybe_preemptive_sleep() -> None:
    """Sleep until reset if we're below the configured remaining threshold.

    Capped by ``GITHUB_RATE_LIMIT_SLEEP_CAP`` so a single sleep never hangs
    indefinitely. GitHub resets hourly so in theory the wait could be up to
    ~3600s, but we deliberately bound it for test-friendliness — the next
    request after the cap will simply observe state and sleep again if
    still throttled.
    """
    remaining = _RATE_LIMIT_STATE.get("remaining")
    reset = _RATE_LIMIT_STATE.get("reset")
    if remaining is None or reset is None:
        return
    if remaining >= GITHUB_RATE_LIMIT_MIN_REMAINING:
        return

    now = time.time()
    wait = max(0.0, float(reset) - now) + _RATE_LIMIT_SLEEP_BUFFER
    wait = min(wait, float(GITHUB_RATE_LIMIT_SLEEP_CAP))
    if wait <= 0:
        return
    print(
        f"[github_http] preemptive rate-limit sleep "
        f"remaining={remaining} reset={reset} sleeping={wait:.1f}s",
        file=sys.stderr,
    )
    time.sleep(wait)


def _handle_rate_limit_403(resp: requests.Response) -> bool:
    """If a 403 indicates rate-limit exhaustion, update state and sleep.

    Returns True iff we treated the response as a rate-limit hit and slept
    (so the caller should retry).
    """
    if resp.status_code != 403:
        return False
    remaining_hdr = resp.headers.get("X-RateLimit-Remaining") if resp.headers else None
    try:
        remaining = int(remaining_hdr) if remaining_hdr is not None else None
    except (TypeError, ValueError):
        remaining = None
    if remaining != 0:
        return False
    _parse_rate_limit_headers(resp.headers)
    _maybe_preemptive_sleep()
    return True


def github_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    memory: Any = None,
    timeout: int = 10,
) -> GitHubResponse:
    """GET a GitHub URL, returning parsed JSON plus metadata.

    Parameters
    ----------
    url:
        Full URL to request.
    headers:
        Optional override headers. Defaults to the configured
        :func:`github_headers` (Authorization + Accept).
    retries:
        Number of attempts on 429/5xx before giving up. Matches the legacy
        behaviour of ``_get``.
    memory:
        Optional :class:`ExplainerMemory` used to store/retrieve ETag entries
        for conditional requests. When ``None``, no caching happens.
    timeout:
        Per-request timeout in seconds.
    """
    base_headers = dict(headers) if headers is not None else github_headers()

    # Attach If-None-Match when we have a cached ETag for this URL.
    cached_entry: dict | None = None
    if memory is not None:
        try:
            cached_entry = memory.get_etag_cache(url)
        except AttributeError:
            cached_entry = None
        if cached_entry and cached_entry.get("etag"):
            base_headers["If-None-Match"] = cached_entry["etag"]

    # Preemptive sleep based on the most recently observed rate-limit state.
    _maybe_preemptive_sleep()

    response: requests.Response | None = None
    for attempt in range(retries):
        response = requests.get(url, headers=base_headers, timeout=timeout)

        # A 403 with Remaining=0 is a rate-limit hit; sleep and retry rather
        # than giving up immediately.
        if _handle_rate_limit_403(response) and attempt < retries - 1:
            continue

        # Retry on 429 / 5xx with exponential backoff (preserved behaviour).
        if (
            response.status_code == 429
            or 500 <= response.status_code < 600
        ) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue

        break

    assert response is not None  # retries >= 1 guarantees at least one attempt

    _parse_rate_limit_headers(response.headers)

    # 304 Not Modified → serve from the ETag cache.
    if response.status_code == 304 and cached_entry is not None:
        return GitHubResponse(
            data=cached_entry.get("data"),
            status_code=304,
            headers=_safe_headers_dict(response.headers),
            from_cache=True,
        )

    # 404 → no data, but not an error.
    if response.status_code == 404:
        return GitHubResponse(
            data=None,
            status_code=404,
            headers=_safe_headers_dict(response.headers),
            from_cache=False,
        )

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            data = None
        # Update the ETag cache if the server provided one.
        etag = response.headers.get("ETag") if response.headers else None
        if memory is not None and etag and data is not None:
            try:
                memory.set_etag_cache(url, etag, data)
            except AttributeError:
                pass
        return GitHubResponse(
            data=data,
            status_code=200,
            headers=_safe_headers_dict(response.headers),
            from_cache=False,
        )

    # Any other status → surface it unchanged so callers can apply their own
    # error mapping (auth failure, 429 after exhausted retries, etc).
    return GitHubResponse(
        data=None,
        status_code=response.status_code,
        headers=dict(response.headers or {}),
        from_cache=False,
    )

"""Tests for the JSON-backed explainer cache."""

from __future__ import annotations

from pathlib import Path

from git_explainer.memory import ExplainerMemory


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def test_memory_round_trip(tmp_path):
    repo = _make_repo(tmp_path)
    memory = ExplainerMemory(str(repo))

    memory.set_commit_prs("abc1234", [42])
    memory.set_pr(42, {"number": 42, "title": "Explain history"})
    memory.set_issue(7, {"number": 7, "title": "Bug"})
    memory.set_context("abc1234:ctx", "sample context")
    memory.flush()

    reloaded = ExplainerMemory(str(repo))
    assert reloaded.get_commit_prs("abc1234") == [42]
    assert reloaded.get_pr(42) == {"number": 42, "title": "Explain history"}
    assert reloaded.get_issue(7) == {"number": 7, "title": "Bug"}
    assert reloaded.get_context("abc1234:ctx") == "sample context"


def test_memory_tracks_hits_and_misses(tmp_path):
    repo = _make_repo(tmp_path)
    memory = ExplainerMemory(str(repo))

    assert memory.get_pr(99) is None
    memory.set_pr(99, {"number": 99})
    assert memory.get_pr(99) == {"number": 99}
    assert memory.stats()["misses"] == 1
    assert memory.stats()["hits"] == 1


def test_etag_cache_round_trip(tmp_path):
    repo = _make_repo(tmp_path)
    memory = ExplainerMemory(str(repo))

    url = "https://api.github.com/repos/o/r/issues/1"
    assert memory.get_etag_cache(url) is None

    memory.set_etag_cache(url, 'W/"tag-v1"', {"number": 1, "title": "t"})
    entry = memory.get_etag_cache(url)
    assert entry is not None
    assert entry["etag"] == 'W/"tag-v1"'
    assert entry["data"] == {"number": 1, "title": "t"}
    assert "last_fetched" in entry

    memory.flush()
    reloaded = ExplainerMemory(str(repo))
    persisted = reloaded.get_etag_cache(url)
    assert persisted is not None
    assert persisted["etag"] == 'W/"tag-v1"'


def test_etag_cache_backward_compat_with_old_payload(tmp_path):
    """Old cache files without an 'etags' section should still load cleanly."""
    repo = _make_repo(tmp_path)
    import json
    old_payload = {
        "version": 1,
        "commit_prs": {"abc": [1]},
        "prs": {},
        "pr_comments": {},
        "issues": {},
        "issue_comments": {},
        "contexts": {},
        "diffs": {},
        # NOTE: no 'etags' key
    }
    (repo / ".git_explainer_cache.json").write_text(
        json.dumps(old_payload), encoding="utf-8"
    )
    memory = ExplainerMemory(str(repo))
    assert memory.get_commit_prs("abc") == [1]
    assert memory.get_etag_cache("https://example.com/any") is None
    # New entries should still be writable.
    memory.set_etag_cache("https://example.com/any", "etag", {"data": 1})
    assert memory.get_etag_cache("https://example.com/any")["etag"] == "etag"

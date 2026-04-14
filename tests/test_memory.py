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

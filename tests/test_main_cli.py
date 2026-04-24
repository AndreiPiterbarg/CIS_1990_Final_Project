"""Tests for the CLI entry point in main.py."""

from __future__ import annotations

from main import build_parser


def test_default_enforces_public_repo():
    parser = build_parser()
    args = parser.parse_args(["/tmp/repo", "src/app.py", "1", "2"])
    assert args.enforce_public_repo is True


def test_allow_private_repo_flag_disables_enforcement():
    parser = build_parser()
    args = parser.parse_args(
        ["/tmp/repo", "src/app.py", "1", "2", "--allow-private-repo"]
    )
    assert args.enforce_public_repo is False

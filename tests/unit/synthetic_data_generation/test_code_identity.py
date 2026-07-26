"""Tests for synthetic-data publisher code identity."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.synthetic_data_generation.code_identity import (
    FULL_SCALE_RELEVANT_FILES,
    compute_code_identity,
)


def test_code_identity_ignores_unrelated_worktree_changes(tmp_path: Path) -> None:
    for relative in FULL_SCALE_RELEVANT_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen: {relative}\n", encoding="utf-8")
    unrelated = tmp_path / "README.md"
    unrelated.write_text("initial\n", encoding="utf-8")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "initial")

    baseline = compute_code_identity(tmp_path)
    unrelated.write_text("unrelated change\n", encoding="utf-8")
    after_unrelated_change = compute_code_identity(tmp_path)
    relevant = tmp_path / FULL_SCALE_RELEVANT_FILES[0]
    relevant.write_text("relevant change\n", encoding="utf-8")
    after_relevant_change = compute_code_identity(tmp_path)

    assert after_unrelated_change == baseline
    assert after_relevant_change != baseline


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

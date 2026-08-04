"""Command-level tests for the repository's :mod:`spin` workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_spin(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "spin", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_help_lists_project_workflows() -> None:
    result = _run_spin("--help")

    assert result.returncode == 0, result.stderr
    for command in ("ci", "doctor", "lint", "setup", "test", "typecheck"):
        assert command in result.stdout


def test_lint_can_check_an_explicit_path() -> None:
    result = _run_spin("lint", ".spin/cmds.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed" in result.stdout


def test_typecheck_can_check_an_explicit_path() -> None:
    result = _run_spin("typecheck", ".spin/cmds.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Success: no issues found" in result.stdout


def test_changed_mode_rejects_an_unknown_git_base() -> None:
    result = _run_spin(
        "lint",
        "--changed",
        "--base",
        "refs/heads/definitely-not-a-real-branch",
    )

    assert result.returncode != 0
    assert "was not found" in result.stderr


def test_test_command_forwards_pytest_arguments() -> None:
    result = _run_spin(
        "test",
        "--all",
        "--serial",
        "tests/unit/utils/geometry/test_angles.py",
        "-q",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout

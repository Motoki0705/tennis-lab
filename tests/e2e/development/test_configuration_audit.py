"""Command-level tests for the canonical configuration audit entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_audit_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/audit_configuration.py", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_root_configuration_audit_script_is_the_operational_entrypoint() -> None:
    result = _run_audit_script("--help")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "source_root" in result.stdout
    assert "--show-discovered-boundaries" in result.stdout


def test_generated_inventory_update_requires_an_explicit_revision() -> None:
    result = _run_audit_script("src", "--write-generated-data")

    assert result.returncode == 2
    assert "--write-generated-data requires --source-revision" in result.stderr

"""Explicit asset acceptance cannot be green when a test did not execute."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "source,expected",
    [
        ("def test_ok(): pass", 0),
        ("import pytest\ndef test_skip(): pytest.skip('no asset')", 1),
        ("import pytest\npytest.skip('no asset', allow_module_level=True)", 1),
        ("def helper(): pass", 1),
    ],
)
def test_required_execution_reports_missing_acceptance(
    tmp_path: Path, source: str, expected: int
) -> None:
    path = tmp_path / "test_acceptance.py"
    path.write_text(source)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "scripts.ci_metrics.require_execution",
            str(path),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTEST_ADDOPTS": ""},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr

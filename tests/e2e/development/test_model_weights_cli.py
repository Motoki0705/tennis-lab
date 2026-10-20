"""The published model-card command remains executable after package changes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_model_weights_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.utils.model_weights", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for option in ("--checkpoint-root", "--model", "--revision", "--verify-only"):
        assert option in result.stdout

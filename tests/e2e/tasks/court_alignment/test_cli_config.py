"""Command-level config publication checks for court-alignment scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _run_module(module: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[4]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, "-m", module, "--cfg", "job", *arguments],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("sigma_px", (0.75, 1.0, 1.5, 2.0))
def test_train_cli_publishes_sigma_override(sigma_px: float) -> None:
    result = _run_module(
        "src.tasks.court_alignment.scripts.train",
        f"data.sigma_px={sigma_px}",
    )

    assert result.returncode == 0, result.stderr
    assert f"sigma_px: {sigma_px}" in result.stdout
    assert "augmentations:" in result.stdout
    assert "_target_: src.tasks.court_alignment.models.cnn.CourtAlignmentCNN" in result.stdout


def test_evaluate_cli_publishes_explicit_checkpoint_contract() -> None:
    result = _run_module("src.tasks.court_alignment.scripts.evaluate")

    assert result.returncode == 0, result.stderr
    assert "evaluation:" in result.stdout
    assert "checkpoint_path: null" in result.stdout

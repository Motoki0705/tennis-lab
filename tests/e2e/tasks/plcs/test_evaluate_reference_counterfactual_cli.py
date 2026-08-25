"""Asset-free Hydra CLI checks for PLCS paired-reference evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from src.tasks.plcs.scripts import evaluate_reference_counterfactual as cli_module
from src.utils.paths import PROJECT_ROOT


def _environment(repro_dir: Path) -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": "0",
        "TENNIS_REPRO_DIR": str(repro_dir),
    }


def test_plcs_counterfactual_cli_declares_standard_queue_artifacts() -> None:
    documentation = cli_module.__doc__ or ""
    assert "pred_test.npz" in documentation
    assert "flat `metrics.json`" in documentation
    assert "reference_counterfactual.json" in documentation


def test_plcs_counterfactual_cli_composes_explicit_checkpoint_only_contract(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.plcs.scripts.evaluate_reference_counterfactual",
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=PROJECT_ROOT,
        env=_environment(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )
    config = yaml.safe_load(completed.stdout)
    assert config["evaluation"] == {
        "task": "plcs",
        "checkpoint_path": "???",
        "output_dir": f"{tmp_path}/predictions",
        "passes": ["same_side", "opposite_side"],
        "trainer": {
            "accelerator": "auto",
            "devices": 1,
            "precision": "32-true",
            "deterministic": True,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    }
    assert config["data"]["seq_len_range"] == [128, 128]
    assert config["data"]["num_views_range"] == [6, 6]
    assert config["data"]["camera_mode"] == "first"
    assert config["run"]["seed"] == 42
    assert config["training"]["compile"]["enabled"] is False


def test_plcs_counterfactual_cli_rejects_unknown_evaluation_key(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.touch()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.plcs.scripts.evaluate_reference_counterfactual",
            f"evaluation.checkpoint_path={checkpoint}",
            "+evaluation.unknown=true",
        ],
        cwd=PROJECT_ROOT,
        env=_environment(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Unknown configuration key" in completed.stderr


def test_plcs_counterfactual_cli_requires_checkpoint_path(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.plcs.scripts.evaluate_reference_counterfactual",
        ],
        cwd=PROJECT_ROOT,
        env=_environment(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "checkpoint_path" in completed.stderr

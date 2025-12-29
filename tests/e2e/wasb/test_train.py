"""E2E tests for WASB training scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.fixtures.wasb_fixtures import create_minimal_wasb_dataset


@pytest.mark.e2e
def test_wasb_ball_detection_train_dry_run(tmp_path: Path) -> None:
    """Test WASB ball detection training in dry run mode.

    This test verifies that:
    1. The training script runs in dry run mode (validates dataloader only)
    2. Dry run directory is created

    """
    # Create minimal dataset
    dataset_dir = tmp_path / "tennis"
    create_minimal_wasb_dataset(dataset_dir)

    output_dir = tmp_path / "wasb_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.train.ball_detection",
            f"run.output_dir={output_dir}",
            f"data.root_dir={dataset_dir}",
            "data.train_matches=[game1]",
            "data.val_matches=[game1]",
            "run.dry_run=true",
            "run.gpus=0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Dry run failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check dry run directory exists
    dry_run_dir = output_dir / "dry_run"
    assert dry_run_dir.exists(), "Dry run directory was not created"


@pytest.mark.e2e
def test_wasb_ball_detection_train_fast_dev(tmp_path: Path) -> None:
    """Test WASB ball detection training with fast_dev_run.

    This test verifies that:
    1. The training script runs with fast_dev_run
    2. Checkpoint files are created

    """
    dataset_dir = tmp_path / "tennis"
    create_minimal_wasb_dataset(dataset_dir)

    output_dir = tmp_path / "wasb_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.train.ball_detection",
            f"run.output_dir={output_dir}",
            f"data.root_dir={dataset_dir}",
            "data.train_matches=[game1]",
            "data.val_matches=[game1]",
            "run.fast_dev_run=true",
            "run.gpus=0",
            "training.max_epochs=1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check that checkpoint directory exists
    checkpoint_pattern = output_dir / "logs" / "version_*" / "checkpoints"
    checkpoint_dirs = list(output_dir.glob("logs/version_*/checkpoints"))

    assert len(checkpoint_dirs) > 0, "No checkpoint directory was created"

    # Check that checkpoints exist
    checkpoint_files = list(checkpoint_dirs[0].glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint files were created"


@pytest.mark.e2e
def test_wasb_trajectory_train_dry_run(tmp_path: Path) -> None:
    """Test WASB trajectory training in dry run mode.

    This test verifies the trajectory completion model training script.

    """
    dataset_dir = tmp_path / "tennis"
    create_minimal_wasb_dataset(dataset_dir)

    output_dir = tmp_path / "wasb_trajectory_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.train.trajectory",
            f"run.output_dir={output_dir}",
            f"data.root_dir={dataset_dir}",
            "data.train_matches=[game1]",
            "data.val_matches=[game1]",
            "run.dry_run=true",
            "run.gpus=0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Dry run failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"


@pytest.mark.e2e
def test_wasb_event_detection_train_dry_run(tmp_path: Path) -> None:
    """Test WASB event detection training in dry run mode.

    This test verifies the event detection model training script.

    """
    dataset_dir = tmp_path / "tennis"
    create_minimal_wasb_dataset(dataset_dir)

    output_dir = tmp_path / "wasb_event_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.train.event_detection",
            f"run.output_dir={output_dir}",
            f"data.root_dir={dataset_dir}",
            "data.train_matches=[game1]",
            "data.val_matches=[game1]",
            "run.dry_run=true",
            "run.gpus=0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Dry run failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

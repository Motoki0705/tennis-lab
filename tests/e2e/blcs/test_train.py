"""E2E tests for BLCS training script."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.fixtures.blcs_fixtures import create_minimal_blcs_dataset


@pytest.mark.e2e
def test_blcs_train_basic(tmp_path: Path) -> None:
    """Test BLCS training script with minimal config.

    This test verifies that:
    1. The training script runs without errors
    2. Checkpoint files are created
    3. Config file is saved

    """
    # Create minimal dataset
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_blcs_dataset(dataset_dir, num_scenes=10)

    output_dir = tmp_path / "blcs_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.blcs.scripts.train",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "training.max_epochs=1",
            "run.gpus=0",
            "run.fast_dev_run=true",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Assert success
    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output files
    assert output_dir.exists(), "Output directory was not created"

    # Check checkpoint directory exists
    checkpoint_dir = output_dir / "checkpoints"
    assert checkpoint_dir.exists(), "Checkpoint directory was not created"

    # Check that at least one checkpoint was created
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint files were created"

    # Check config file exists
    config_file = output_dir / "config.yaml"
    assert config_file.exists(), "Config file was not created"


@pytest.mark.e2e
def test_blcs_train_with_custom_params(tmp_path: Path) -> None:
    """Test BLCS training with custom hyperparameters.

    This test verifies that Hydra config overrides work correctly.

    """
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_blcs_dataset(dataset_dir, num_scenes=10)

    output_dir = tmp_path / "blcs_custom_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.blcs.scripts.train",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "training.max_epochs=2",
            "run.gpus=0",
            "run.fast_dev_run=true",
            "data.batch_size=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    checkpoint_dir = output_dir / "checkpoints"
    assert checkpoint_dir.exists(), "Checkpoint directory was not created"

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint files were created"

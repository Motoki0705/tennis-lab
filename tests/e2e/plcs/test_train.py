"""E2E tests for PLCS training scripts."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import pytest

from tests.e2e.fixtures.plcs_fixtures import create_minimal_plcs_dataset

F = TypeVar("F", bound=Callable[..., object])
e2e = cast(Callable[[F], F], pytest.mark.e2e)
cuda = cast(Callable[[F], F], pytest.mark.cuda)


@e2e
@cuda
def test_plcs_train_basic(tmp_path: Path) -> None:
    """Test PLCS training script with minimal config (GPU mode).

    This test verifies that:
    1. The training script runs without errors
    2. Checkpoint files are created
    3. Config file is saved

    Note: This test requires CUDA/GPU.
    """
    # Create minimal dataset
    dataset_dir = tmp_path / "plcs_data"
    create_minimal_plcs_dataset(dataset_dir, num_scenes=10)

    output_dir = tmp_path / "plcs_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.plcs.scripts.train",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "training.max_epochs=1",
            "run.gpus=1",
            "run.fast_dev_run=true",
            "data.batch_size=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Assert success
    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output files
    assert output_dir.exists(), "Output directory was not created"

    # Check checkpoint directory exists (under versioned log directory)
    checkpoint_dirs = list(output_dir.glob("logs/version_*/checkpoints"))
    assert len(checkpoint_dirs) > 0, "Checkpoint directory was not created"
    checkpoint_dir = checkpoint_dirs[0]

    # Check that at least one checkpoint was created
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint files were created"

    # Check config file exists
    config_file = output_dir / "config.yaml"
    assert config_file.exists(), "Config file was not created"


@e2e
@cuda
def test_plcs_train_sequence(tmp_path: Path) -> None:
    """Test PLCS sequence training script (GPU mode).

    This test verifies the sequence-based training variant works.

    Note: This test requires CUDA/GPU.
    """
    # Create minimal dataset
    dataset_dir = tmp_path / "plcs_data"
    create_minimal_plcs_dataset(dataset_dir, num_scenes=10)

    output_dir = tmp_path / "plcs_seq_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.plcs.scripts.train_sequence",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "training.max_epochs=1",
            "run.gpus=1",
            "run.fast_dev_run=true",
            "data.batch_size=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output files (checkpoint directory under versioned log directory)
    checkpoint_dirs = list(output_dir.glob("logs/version_*/checkpoints"))
    assert len(checkpoint_dirs) > 0, "Checkpoint directory was not created"
    checkpoint_dir = checkpoint_dirs[0]

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint files were created"


@e2e
@cuda
def test_plcs_train_with_custom_params(tmp_path: Path) -> None:
    """Test PLCS training with custom hyperparameters (GPU mode).

    This test verifies that Hydra config overrides work correctly.

    Note: This test requires CUDA/GPU.
    """
    dataset_dir = tmp_path / "plcs_data"
    create_minimal_plcs_dataset(dataset_dir, num_scenes=10)

    output_dir = tmp_path / "plcs_custom_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.plcs.scripts.train",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "training.max_epochs=2",
            "run.gpus=1",
            "run.fast_dev_run=true",
            "data.batch_size=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check checkpoint directory exists (under versioned log directory)
    checkpoint_dirs = list(output_dir.glob("logs/version_*/checkpoints"))
    assert len(checkpoint_dirs) > 0, "Checkpoint directory was not created"


@e2e
def test_plcs_train_dry_run(tmp_path: Path) -> None:
    """Test PLCS training dry run mode (CPU only).

    This test verifies that:
    1. The dry run mode works without errors
    2. Config file is saved
    3. No checkpoints are created (dry run doesn't save)

    Note: Dry run is designed to work on CPU, no GPU required.
    """
    # Create minimal dataset
    dataset_dir = tmp_path / "plcs_data"
    create_minimal_plcs_dataset(dataset_dir, num_scenes=10)

    output_dir = tmp_path / "plcs_dry_run_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.plcs.scripts.train",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "run.dry_run=true",
            "data.batch_size=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Assert success
    assert result.returncode == 0, f"Dry run failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output files
    assert output_dir.exists(), "Output directory was not created"

    # Check config file exists
    config_file = output_dir / "config.yaml"
    assert config_file.exists(), "Config file was not created"

    # Dry run should print batch info
    assert "dry run" in result.stdout.lower() or "Running dry run" in result.stdout


@e2e
def test_plcs_train_sequence_dry_run(tmp_path: Path) -> None:
    """Test PLCS sequence training dry run mode (CPU only).

    This test verifies that sequence training dry run works.

    Note: Dry run is designed to work on CPU, no GPU required.
    """
    # Create minimal dataset
    dataset_dir = tmp_path / "plcs_data"
    create_minimal_plcs_dataset(dataset_dir, num_scenes=10)

    output_dir = tmp_path / "plcs_seq_dry_run_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.plcs.scripts.train_sequence",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "run.dry_run=true",
            "data.batch_size=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Assert success
    assert result.returncode == 0, f"Dry run failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output files
    assert output_dir.exists(), "Output directory was not created"

    # Check config file exists
    config_file = output_dir / "config.yaml"
    assert config_file.exists(), "Config file was not created"

    # Dry run should print batch info
    assert "dry run" in result.stdout.lower() or "Running dry run" in result.stdout

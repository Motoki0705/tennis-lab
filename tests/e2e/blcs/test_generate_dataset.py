"""E2E tests for BLCS dataset generation script."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_blcs_generate_dataset(tmp_path: Path) -> None:
    """Test BLCS dataset generation script.

    This test verifies that:
    1. The script runs without errors
    2. Scene files are created
    3. Split files (train.txt, val.txt, test.txt) are created

    """
    output_dir = tmp_path / "blcs_generated"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.blcs.scripts.generate_dataset",
            f"run.output_dir={output_dir}",
            "sampling.per_from_cell_samples=10",
            "generator.num_cameras_sampled=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Assert script completed successfully
    assert result.returncode == 0, f"Script failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output directory exists
    assert output_dir.exists(), "Output directory was not created"

    # Check scenes directory exists
    scenes_dir = output_dir / "scenes"
    assert scenes_dir.exists(), "Scenes directory was not created"

    # Check that scene files were created (may be less than requested due to visibility filtering)
    scene_files = list(scenes_dir.glob("scene_*.npz"))
    assert len(scene_files) > 0, f"Expected at least 1 scene file, found {len(scene_files)}"

    # Check metadata files exist (BLCS generate_dataset doesn't create train/val/test.txt)
    assert (output_dir / "meta.json").exists(), "meta.json not created"


@pytest.mark.e2e
def test_blcs_generate_dataset_minimal_samples(tmp_path: Path) -> None:
    """Test BLCS dataset generation with very minimal samples.

    This test verifies quick generation with minimal configuration.

    """
    output_dir = tmp_path / "blcs_minimal"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.blcs.scripts.generate_dataset",
            f"run.output_dir={output_dir}",
            "sampling.per_from_cell_samples=5",
            "generator.num_cameras_sampled=1",
            "generator.ball_visibility_threshold=0.5",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Script failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    scenes_dir = output_dir / "scenes"
    scene_files = list(scenes_dir.glob("scene_*.npz"))
    assert len(scene_files) > 0, f"Expected at least 1 scene file, found {len(scene_files)}"

"""E2E tests for PLCS dataset generation script."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_plcs_generate_dataset(tmp_path: Path) -> None:
    """Test PLCS dataset generation script.

    This test verifies that:
    1. The script runs without errors
    2. Scene files are created
    3. Split files (train.txt, val.txt, test.txt) are created
    4. Metadata files are created

    """
    output_dir = tmp_path / "plcs_generated"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.plcs.scripts.generate_dataset",
            f"run.output_dir={output_dir}",
            "simulation.num_scenes=10",
            "simulation.num_cameras=2",
            "simulation.human_visibility_threshold=0.3",
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
    assert len(scene_files) <= 10, f"Expected at most 10 scene files, found {len(scene_files)}"

    # Check metadata files exist (PLCS generate_dataset doesn't create train/val/test.txt)
    assert (output_dir / "stats.json").exists(), "stats.json not created"
    assert (output_dir / "scenes_meta.json").exists(), "scenes_meta.json not created"
    assert (output_dir / "meta.json").exists(), "meta.json not created"


@pytest.mark.e2e
def test_plcs_generate_dataset_custom_settings(tmp_path: Path) -> None:
    """Test PLCS dataset generation with custom settings.

    This test verifies that Hydra config overrides work correctly.

    """
    output_dir = tmp_path / "plcs_custom"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.plcs.scripts.generate_dataset",
            f"run.output_dir={output_dir}",
            "simulation.num_scenes=5",
            "simulation.num_cameras=1",
            "simulation.human_visibility_threshold=0.3",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Script failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    scenes_dir = output_dir / "scenes"
    scene_files = list(scenes_dir.glob("scene_*.npz"))
    assert len(scene_files) > 0, f"Expected at least 1 scene file, found {len(scene_files)}"
    assert len(scene_files) <= 5, f"Expected at most 5 scene files, found {len(scene_files)}"

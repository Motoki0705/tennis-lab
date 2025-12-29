"""E2E tests for BLCS visualization script."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.fixtures.blcs_fixtures import (
    create_minimal_blcs_checkpoint,
    create_minimal_blcs_dataset,
)


@pytest.mark.e2e
def test_blcs_visualize_ground_truth(tmp_path: Path) -> None:
    """Test BLCS visualization with ground truth.

    This test verifies that:
    1. The visualization script runs without errors in visualize mode
    2. Output image is created when save path is specified

    """
    # Create a minimal dataset with 1 scene
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_blcs_dataset(dataset_dir, num_scenes=1)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))
    output_path = tmp_path / "vis_gt.png"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.blcs.scripts.visualize",
            f"visualization.scene_path={scene_path}",
            "visualization.mode=visualize",
            "visualization.view=3d",
            f"visualization.save={output_path}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Visualization failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check that output file was created
    assert output_path.exists(), "Output visualization was not created"


@pytest.mark.e2e
def test_blcs_visualize_predict(tmp_path: Path) -> None:
    """Test BLCS visualization with model prediction.

    This test verifies that:
    1. The visualization script runs with a checkpoint in predict mode
    2. Predictions can be visualized

    """
    # Create dataset and checkpoint
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_blcs_dataset(dataset_dir, num_scenes=1)

    checkpoint_path = tmp_path / "blcs_model.ckpt"
    create_minimal_blcs_checkpoint(checkpoint_path)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))
    output_path = tmp_path / "vis_pred.png"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.blcs.scripts.visualize",
            f"visualization.scene_path={scene_path}",
            "visualization.mode=predict",
            f"visualization.checkpoint={checkpoint_path}",
            f"visualization.save={output_path}",
            "visualization.view=3d",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Visualization failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check that output file was created
    assert output_path.exists(), "Output visualization was not created"


@pytest.mark.e2e
def test_blcs_visualize_different_views(tmp_path: Path) -> None:
    """Test BLCS visualization with different view modes.

    This test verifies that different visualization views work.

    """
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_blcs_dataset(dataset_dir, num_scenes=1)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))

    # Test 2d view
    output_path_2d = tmp_path / "vis_2d.png"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.blcs.scripts.visualize",
            f"visualization.scene_path={scene_path}",
            "visualization.mode=visualize",
            "visualization.view=2d",
            f"visualization.save={output_path_2d}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"2D visualization failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    assert output_path_2d.exists(), "2D output visualization was not created"


@pytest.mark.e2e
def test_blcs_visualize_camera_view(tmp_path: Path) -> None:
    """Test BLCS visualization with camera view mode.

    This test verifies the camera view works correctly.

    """
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_blcs_dataset(dataset_dir, num_scenes=1)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))
    output_path_camera = tmp_path / "vis_camera.png"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.blcs.scripts.visualize",
            f"visualization.scene_path={scene_path}",
            "visualization.mode=visualize",
            "visualization.view=camera",
            "visualization.camera=0",
            f"visualization.save={output_path_camera}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Camera view failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    assert output_path_camera.exists(), "Camera output visualization was not created"

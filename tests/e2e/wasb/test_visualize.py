"""E2E tests for WASB visualization scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.fixtures.wasb_fixtures import (
    create_minimal_trajectory_checkpoint,
    create_minimal_video,
    create_minimal_wasb_checkpoint,
    create_minimal_wasb_dataset,
)


@pytest.mark.e2e
@pytest.mark.skip(
    reason="Requires real WASB HRNet checkpoint; mock checkpoint creation not feasible"
)
def test_wasb_ball_video(tmp_path: Path) -> None:
    """Test WASB ball detection video visualization.

    This test verifies that:
    1. The ball_video script runs without errors
    2. Output video is created

    """
    # Create minimal video and checkpoint
    video_path = tmp_path / "test_match.mp4"
    create_minimal_video(video_path, num_frames=50)

    checkpoint_path = tmp_path / "wasb_model.ckpt"
    create_minimal_wasb_checkpoint(checkpoint_path)

    output_path = tmp_path / "output_ball.mp4"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.visualize.ball_video",
            f"video_path={video_path}",
            f"checkpoint={checkpoint_path}",
            f"output_path={output_path}",
            "max_frames=50",
            "device=cpu",
            "completion.enabled=false",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Visualization failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check that output video was created
    assert output_path.exists(), "Output video was not created"


@pytest.mark.e2e
@pytest.mark.skip(
    reason="Requires real WASBLightningModule checkpoint; mock checkpoint creation not feasible"
)
def test_wasb_ball_video_ensemble(tmp_path: Path) -> None:
    """Test WASB ensemble ball detection video visualization.

    This test verifies the ensemble prediction script.

    """
    # Create minimal video and multiple checkpoints
    video_path = tmp_path / "test_match.mp4"
    create_minimal_video(video_path, num_frames=50)

    checkpoint1 = tmp_path / "wasb_model1.ckpt"
    checkpoint2 = tmp_path / "wasb_model2.ckpt"
    create_minimal_wasb_checkpoint(checkpoint1)
    create_minimal_wasb_checkpoint(checkpoint2)

    output_path = tmp_path / "output_ensemble.mp4"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.visualize.ball_video_ensemble",
            f"video_path={video_path}",
            f"ensemble.checkpoints=[{checkpoint1},{checkpoint2}]",
            f"output_path={output_path}",
            "max_frames=50",
            "device=cpu",
            "completion.enabled=false",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Ensemble visualization failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    assert output_path.exists(), "Output video was not created"


@pytest.mark.e2e
def test_wasb_trajectory_visualize(tmp_path: Path) -> None:
    """Test WASB trajectory visualization.

    This test verifies the trajectory completion visualization script.

    """
    # Create dataset and checkpoint
    dataset_dir = tmp_path / "tennis"
    create_minimal_wasb_dataset(dataset_dir)

    checkpoint_path = tmp_path / "trajectory_model.ckpt"
    create_minimal_trajectory_checkpoint(checkpoint_path)

    output_dir = tmp_path / "trajectory_vis"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.visualize.trajectory",
            f"visualization.checkpoint={checkpoint_path}",
            f"visualization.output_dir={output_dir}",
            f"data.root_dir={dataset_dir}",
            "data.test_matches=[game1]",
            "visualization.split=test",
            "visualization.num_samples=2",
            "run.gpus=0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Trajectory visualization failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output directory exists
    assert output_dir.exists(), "Output directory was not created"

    # Check that sample images were created
    sample_images = list(output_dir.glob("test_sample_*.png"))
    assert len(sample_images) > 0, "No sample images were created"


@pytest.mark.e2e
def test_wasb_save_one_sample_visuals(tmp_path: Path) -> None:
    """Test WASB save one sample visuals script.

    This test verifies the script that saves individual sample visualizations.

    """
    # Create dataset
    dataset_dir = tmp_path / "tennis"
    create_minimal_wasb_dataset(dataset_dir)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.visualize.save_one_sample_visuals",
            f"data.root_dir={dataset_dir}",
            "data.train_matches=[game1]",
            "data.val_matches=[game1]",
            "data.test_matches=[game1]",
            "split=train",
            "sample_index=0",
            "target_index=0",
            "num_samples=1",
            "data.batch_size=1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Save sample visuals failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

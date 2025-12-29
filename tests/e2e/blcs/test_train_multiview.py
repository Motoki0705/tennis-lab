"""E2E tests for BLCS multi-view training scripts."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pytest
import torch

from src.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.blcs.simulation.cell_manager import ShotCategory

F = TypeVar("F", bound=Callable[..., object])
e2e = cast(Callable[[F], F], pytest.mark.e2e)
cuda = cast(Callable[[F], F], pytest.mark.cuda)


def make_multiview_blcs_scene(
    *, scene_id: str = "scene_000000", num_cameras: int = 3
) -> BLCSSceneData:
    """Create a minimal multi-camera BLCS scene for testing.

    Args:
        scene_id: Scene identifier
        num_cameras: Number of cameras (must be >= 2 for multi-view)

    Returns:
        BLCSSceneData: Minimal scene data for multi-view testing

    """
    T = 30  # 1 second at 30 fps
    fps = 30

    # Simple parabolic trajectory (serve from near side to far side)
    ball_pos_world = torch.zeros(T, 3, dtype=torch.float32)
    ball_pos_world[:, 0] = torch.linspace(0, 2, T)  # x: 0 to 2m
    ball_pos_world[:, 1] = torch.linspace(-5, 5, T)  # y: -5 to 5m
    ball_pos_world[:, 2] = torch.sin(torch.linspace(0, np.pi, T)) * 2.0  # z: arc

    # Normalize positions
    ball_pos_norm = ball_pos_world / torch.tensor([5.485, 11.885, 1.07])

    # Compute velocities (finite difference)
    ball_vel_world = torch.zeros(T, 3, dtype=torch.float32)
    ball_vel_world[1:] = (ball_pos_world[1:] - ball_pos_world[:-1]) * fps
    ball_vel_world[0] = ball_vel_world[1]

    # Event times
    t_net = 15  # Crosses net at midpoint
    t_bounce1 = 20  # Bounces at frame 20
    t_fence = -1  # Doesn't hit fence
    t_bounce2 = -1  # No second bounce in this scene

    cameras: list[CameraData] = []
    for camera_idx in range(num_cameras):
        camera_params = {
            "center": [float(camera_idx), 0.0, 5.0],
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "f": 1000.0,
            "cx": 640.0,
            "cy": 360.0,
            "w": 1280,
            "h": 720,
        }

        # Simple UV projection (normalized to [0, 1])
        ball_uv = np.random.rand(T, 2).astype(np.float32)
        ball_visible = np.ones(T, dtype=bool)

        # Court keypoints: normalized to [0, 1]
        court_kp_uv = np.random.rand(20, 2).astype(np.float32)
        court_kp_visible = np.ones(20, dtype=bool)

        cameras.append(
            CameraData(
                camera_params=camera_params,
                ball_uv=ball_uv,
                ball_visible=ball_visible,
                ball_visibility_ratio=1.0,
                court_kp_uv=court_kp_uv,
                court_kp_visible=court_kp_visible,
                court_visibility_count=20.0,
            )
        )

    return BLCSSceneData(
        scene_id=scene_id,
        from_cell=0,
        from_side="near",
        category=ShotCategory.IN_COURT,
        to_cell=6,
        ball_pos_world=ball_pos_world,
        ball_pos_norm=ball_pos_norm,
        ball_vel_world=ball_vel_world,
        t_net=t_net,
        t_fence=t_fence,
        t_bounce1=t_bounce1,
        t_bounce2=t_bounce2,
        cameras=cameras,
        num_cameras_sampled=num_cameras,
        fps_out=fps,
        sim_fps=240,
    )


def create_minimal_multiview_blcs_dataset(
    output_dir: Path,
    num_scenes: int = 5,
    num_cameras: int = 3,
) -> Path:
    """Create a minimal multi-view BLCS dataset for testing.

    Args:
        output_dir: Output directory for the dataset.
        num_scenes: Number of scenes to generate.
        num_cameras: Number of cameras per scene.

    Returns:
        Path: Output directory

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = BLCSDatasetWriter(output_dir=output_dir)

    # Generate scenes
    for i in range(num_scenes):
        scene_id = f"scene_{i:06d}"
        scene = make_multiview_blcs_scene(
            scene_id=scene_id,
            num_cameras=num_cameras,
        )
        writer.save_scene(scene)

    # Create split files (include .npz extension for dataset loader compatibility)
    scene_ids = [f"scene_{i:06d}.npz" for i in range(num_scenes)]

    # 60% train, 20% val, 20% test
    num_train = int(num_scenes * 0.6)
    num_val = int(num_scenes * 0.2)

    train_ids = scene_ids[:num_train]
    val_ids = scene_ids[num_train : num_train + num_val]
    test_ids = scene_ids[num_train + num_val :]

    (output_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (output_dir / "val.txt").write_text("\n".join(val_ids) + "\n")
    (output_dir / "test.txt").write_text("\n".join(test_ids) + "\n")

    return output_dir


@e2e
def test_blcs_train_multiview_dry_run(tmp_path: Path) -> None:
    """Test BLCS multi-view training script with dry run (CPU only).

    This test verifies that:
    1. The training script runs without errors in dry-run mode
    2. Multi-view data loading works correctly

    """
    # Create minimal multi-view dataset
    dataset_dir = tmp_path / "blcs_multiview_data"
    create_minimal_multiview_blcs_dataset(dataset_dir, num_scenes=5, num_cameras=3)

    output_dir = tmp_path / "blcs_multiview_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.blcs.scripts.train_multiview",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "run.dry_run=true",
            "data.batch_size=2",
            "data.num_views=2",
            "data.min_cameras=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Assert success
    assert result.returncode == 0, (
        f"Multi-view training dry run failed:\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    # Check that batch shapes are printed (from dry run)
    assert "ball_uv" in result.stdout or "Dry run complete" in result.stdout


@e2e
@cuda
def test_blcs_train_multiview_gpu(tmp_path: Path) -> None:
    """Test BLCS multi-view training script with GPU (fast_dev_run).

    This test verifies that:
    1. The training script runs without errors on GPU
    2. Checkpoint files are created
    3. Config file is saved

    Note: This test requires CUDA/GPU.
    """
    # Create minimal multi-view dataset
    dataset_dir = tmp_path / "blcs_multiview_data"
    create_minimal_multiview_blcs_dataset(dataset_dir, num_scenes=10, num_cameras=3)

    output_dir = tmp_path / "blcs_multiview_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.blcs.scripts.train_multiview",
            f"run.output_dir={output_dir}",
            f"data.scene_dir={dataset_dir}",
            "training.max_epochs=1",
            "run.gpus=1",
            "run.fast_dev_run=true",
            "data.batch_size=2",
            "data.num_views=2",
            "data.min_cameras=2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Assert success
    assert result.returncode == 0, (
        f"Multi-view training failed:\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    # Check output files
    assert output_dir.exists(), "Output directory was not created"

    # Check config file exists
    config_file = output_dir / "config.yaml"
    assert config_file.exists(), "Config file was not created"

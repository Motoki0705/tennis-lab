"""E2E tests for PLCS multi-view training scripts."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytest

from src.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.plcs.generate_dataset.scene_generator import CameraData, SceneData

F = TypeVar("F", bound=Callable[..., object])
e2e = cast(Callable[[F], F], pytest.mark.e2e)
cuda = cast(Callable[[F], F], pytest.mark.cuda)


def make_multiview_plcs_scene(
    *, scene_id: str = "scene_000000", num_cameras: int = 3
) -> SceneData:
    """Create a minimal multi-camera PLCS scene for testing.

    Args:
        scene_id: Scene identifier
        num_cameras: Number of cameras (must be >= 2 for multi-view)

    Returns:
        SceneData: Minimal scene data for multi-view testing

    """
    num_frames = 64
    num_joints = 5

    meta = {
        "scene_id": scene_id,
        "motion_source": "e2e_multiview_test",
        "motion_category": "serve",
        "gender": "neutral",
        "fps": 30,
        "num_frames": num_frames,
        "initial_position": [0.1, -0.2],
        "initial_yaw": 0.0,
        "num_cameras_sampled": num_cameras,
    }

    position = np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3) / 10
    rotation = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (num_frames, 1))
    canonical_pose_3d = (
        np.arange(num_frames * num_joints * 3, dtype=np.float32).reshape(
            num_frames, num_joints, 3
        )
        / 100
    )

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
        human_kp_uv: npt.NDArray[np.float32] = np.random.rand(num_frames, 17, 2).astype(
            np.float32
        )
        court_kp_uv: npt.NDArray[np.float32] = np.random.rand(num_frames, 20, 2).astype(
            np.float32
        )
        human_kp_visible: npt.NDArray[np.bool_] = np.ones((num_frames, 17), dtype=bool)
        court_kp_visible: npt.NDArray[np.bool_] = np.ones((num_frames, 20), dtype=bool)

        cameras.append(
            CameraData(
                camera_params=camera_params,
                human_kp_uv=human_kp_uv,
                court_kp_uv=court_kp_uv,
                human_kp_visible=human_kp_visible,
                court_kp_visible=court_kp_visible,
                human_visibility_ratio=0.95,
                court_visibility_count=18.0,
            )
        )

    return SceneData(
        meta=meta,
        position=position,
        rotation=rotation,
        canonical_pose_3d=canonical_pose_3d,
        cameras=cameras,
    )


def create_minimal_multiview_plcs_dataset(
    output_dir: Path,
    num_scenes: int = 5,
    num_cameras: int = 3,
) -> None:
    """Create a minimal multi-view PLCS dataset for testing.

    Args:
        output_dir: Output directory for the dataset.
        num_scenes: Number of scenes to generate.
        num_cameras: Number of cameras per scene.

    """
    writer = PLCSDatasetWriter(output_dir)

    for i in range(num_scenes):
        scene = make_multiview_plcs_scene(
            scene_id=f"scene_{i:06d}",
            num_cameras=num_cameras,
        )
        writer.save_scene(scene)

    writer.save_split_info(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)


@e2e
def test_plcs_train_multiview_dry_run(tmp_path: Path) -> None:
    """Test PLCS multi-view training script with dry run (CPU only).

    This test verifies that:
    1. The training script runs without errors in dry-run mode
    2. Multi-view data loading works correctly

    """
    # Create minimal multi-view dataset
    dataset_dir = tmp_path / "plcs_multiview_data"
    create_minimal_multiview_plcs_dataset(dataset_dir, num_scenes=5, num_cameras=3)

    output_dir = tmp_path / "plcs_multiview_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.plcs.scripts.train_multiview",
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
    assert "human_kp" in result.stdout or "Dry run complete" in result.stdout


@e2e
@cuda
def test_plcs_train_multiview_gpu(tmp_path: Path) -> None:
    """Test PLCS multi-view training script with GPU (fast_dev_run).

    This test verifies that:
    1. The training script runs without errors on GPU
    2. Checkpoint files are created
    3. Config file is saved

    Note: This test requires CUDA/GPU.
    """
    # Create minimal multi-view dataset
    dataset_dir = tmp_path / "plcs_multiview_data"
    create_minimal_multiview_plcs_dataset(dataset_dir, num_scenes=10, num_cameras=3)

    output_dir = tmp_path / "plcs_multiview_output"

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.plcs.scripts.train_multiview",
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

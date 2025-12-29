"""E2E tests for PLCS multi-view visualization script."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import pytest
import torch

from src.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.plcs.training.multiview_lightning_module import PLCSMultiViewLightningModule

F = TypeVar("F", bound=Callable[..., object])
e2e = cast(Callable[[F], F], pytest.mark.e2e)
cuda = cast(Callable[[F], F], pytest.mark.cuda)


def make_multiview_plcs_scene(
    *, scene_id: str = "scene_000000", num_cameras: int = 3
) -> SceneData:
    """Create a minimal multi-camera PLCS scene for testing."""
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
        human_kp_uv: npt.NDArray[np.float32] = np.random.rand(
            num_frames, 17, 2
        ).astype(np.float32)
        court_kp_uv: npt.NDArray[np.float32] = np.random.rand(
            num_frames, 20, 2
        ).astype(np.float32)
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
    num_scenes: int = 1,
    num_cameras: int = 3,
) -> None:
    """Create a minimal multi-view PLCS dataset for testing."""
    writer = PLCSDatasetWriter(output_dir)

    for i in range(num_scenes):
        scene = make_multiview_plcs_scene(
            scene_id=f"scene_{i:06d}",
            num_cameras=num_cameras,
        )
        writer.save_scene(scene)

    writer.save_split_info(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)


def create_minimal_multiview_plcs_checkpoint(checkpoint_path: Path) -> Path:
    """Create a minimal PLCS multi-view checkpoint for testing."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create model with minimal config
    model = PLCSMultiViewModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )

    # Create Lightning module
    config = {
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
        },
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
    }

    lightning_module = PLCSMultiViewLightningModule(config=config)
    lightning_module.model = model

    # Create checkpoint
    checkpoint = {
        "state_dict": lightning_module.state_dict(),
        "hyper_parameters": config,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl.__version__,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


@e2e
def test_plcs_visualize_multiview_ground_truth(tmp_path: Path) -> None:
    """Test PLCS multi-view visualization with ground truth.

    This test verifies that:
    1. The visualization script runs without errors in visualize mode
    2. Output image is created when save path is specified

    """
    dataset_dir = tmp_path / "plcs_data"
    create_minimal_multiview_plcs_dataset(dataset_dir, num_scenes=1, num_cameras=3)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))
    output_path = tmp_path / "vis_multiview_gt.png"

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.plcs.scripts.visualize_multiview",
            f"visualization.scene_path={scene_path}",
            "visualization.mode=visualize",
            "visualization.view=3d",
            f"visualization.save={output_path}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"Visualization failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    assert output_path.exists(), "Output visualization was not created"


@e2e
def test_plcs_visualize_multiview_predict(tmp_path: Path) -> None:
    """Test PLCS multi-view visualization with model prediction.

    This test verifies that:
    1. The visualization script runs with a checkpoint in predict mode
    2. Multi-view predictions can be visualized

    """
    dataset_dir = tmp_path / "plcs_data"
    create_minimal_multiview_plcs_dataset(dataset_dir, num_scenes=1, num_cameras=3)

    checkpoint_path = tmp_path / "plcs_multiview_model.ckpt"
    create_minimal_multiview_plcs_checkpoint(checkpoint_path)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))
    output_path = tmp_path / "vis_multiview_pred.png"

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.plcs.scripts.visualize_multiview",
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

    assert result.returncode == 0, (
        f"Visualization failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    assert output_path.exists(), "Output visualization was not created"


@e2e
def test_plcs_visualize_multiview_info(tmp_path: Path) -> None:
    """Test PLCS multi-view visualization info mode.

    This test verifies that the info mode prints scene information.
    """
    dataset_dir = tmp_path / "plcs_data"
    create_minimal_multiview_plcs_dataset(dataset_dir, num_scenes=1, num_cameras=3)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.plcs.scripts.visualize_multiview",
            f"visualization.scene_path={scene_path}",
            "visualization.info=true",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"Info mode failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    # Check that scene info was printed
    assert "Scene Information" in result.stdout
    assert "Cameras available" in result.stdout

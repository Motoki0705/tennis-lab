"""E2E tests for BLCS multi-view visualization script."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pytorch_lightning as pl
import pytest
import torch

from src.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.blcs.models.blcs_multiview_model import BLCSMultiViewModel
from src.blcs.simulation.cell_manager import ShotCategory
from src.blcs.training.multiview_lightning_module import BLCSMultiViewLightningModule

F = TypeVar("F", bound=Callable[..., object])
e2e = cast(Callable[[F], F], pytest.mark.e2e)
cuda = cast(Callable[[F], F], pytest.mark.cuda)


def make_multiview_blcs_scene(
    *, scene_id: str = "scene_000000", num_cameras: int = 3
) -> BLCSSceneData:
    """Create a minimal multi-camera BLCS scene for testing."""
    T = 30  # 1 second at 30 fps
    fps = 30

    ball_pos_world = torch.zeros(T, 3, dtype=torch.float32)
    ball_pos_world[:, 0] = torch.linspace(0, 2, T)
    ball_pos_world[:, 1] = torch.linspace(-5, 5, T)
    ball_pos_world[:, 2] = torch.sin(torch.linspace(0, np.pi, T)) * 2.0

    ball_pos_norm = ball_pos_world / torch.tensor([5.485, 11.885, 1.07])

    ball_vel_world = torch.zeros(T, 3, dtype=torch.float32)
    ball_vel_world[1:] = (ball_pos_world[1:] - ball_pos_world[:-1]) * fps
    ball_vel_world[0] = ball_vel_world[1]

    t_net = 15
    t_bounce1 = 20
    t_fence = -1
    t_bounce2 = -1

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

        ball_uv = np.random.rand(T, 2).astype(np.float32)
        ball_visible = np.ones(T, dtype=bool)
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
    num_scenes: int = 1,
    num_cameras: int = 3,
) -> Path:
    """Create a minimal multi-view BLCS dataset for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = BLCSDatasetWriter(output_dir=output_dir)

    for i in range(num_scenes):
        scene_id = f"scene_{i:06d}"
        scene = make_multiview_blcs_scene(
            scene_id=scene_id,
            num_cameras=num_cameras,
        )
        writer.save_scene(scene)

    scene_ids = [f"scene_{i:06d}.npz" for i in range(num_scenes)]
    num_train = max(1, int(num_scenes * 0.6))
    num_val = max(1, int(num_scenes * 0.2))

    train_ids = scene_ids[:num_train]
    val_ids = scene_ids[num_train : num_train + num_val] or scene_ids[:1]
    test_ids = scene_ids[num_train + num_val :] or scene_ids[:1]

    (output_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (output_dir / "val.txt").write_text("\n".join(val_ids) + "\n")
    (output_dir / "test.txt").write_text("\n".join(test_ids) + "\n")

    return output_dir


def create_minimal_multiview_blcs_checkpoint(checkpoint_path: Path) -> Path:
    """Create a minimal BLCS multi-view checkpoint for testing."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = BLCSMultiViewModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_seq_len=120,
    )

    config = {
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "max_seq_len": 120,
        },
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
    }

    lightning_module = BLCSMultiViewLightningModule(config=config)
    lightning_module.model = model

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
def test_blcs_visualize_multiview_ground_truth(tmp_path: Path) -> None:
    """Test BLCS multi-view visualization with ground truth.

    This test verifies that:
    1. The visualization script runs without errors in visualize mode
    2. Output image is created when save path is specified

    """
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_multiview_blcs_dataset(dataset_dir, num_scenes=1, num_cameras=3)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))
    output_path = tmp_path / "vis_multiview_gt.png"

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.blcs.scripts.visualize_multiview",
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
def test_blcs_visualize_multiview_predict(tmp_path: Path) -> None:
    """Test BLCS multi-view visualization with model prediction.

    This test verifies that:
    1. The visualization script runs with a checkpoint in predict mode
    2. Multi-view predictions can be visualized

    """
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_multiview_blcs_dataset(dataset_dir, num_scenes=1, num_cameras=3)

    checkpoint_path = tmp_path / "blcs_multiview_model.ckpt"
    create_minimal_multiview_blcs_checkpoint(checkpoint_path)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))
    output_path = tmp_path / "vis_multiview_pred.png"

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.blcs.scripts.visualize_multiview",
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
def test_blcs_visualize_multiview_info(tmp_path: Path) -> None:
    """Test BLCS multi-view visualization info mode.

    This test verifies that the info mode prints scene information.
    """
    dataset_dir = tmp_path / "blcs_data"
    create_minimal_multiview_blcs_dataset(dataset_dir, num_scenes=1, num_cameras=3)

    scene_path = next((dataset_dir / "scenes").glob("scene_*.npz"))

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "src.blcs.scripts.visualize_multiview",
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
    assert "Scene" in result.stdout or "scene" in result.stdout

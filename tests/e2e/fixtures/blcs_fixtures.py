"""BLCS test fixtures for e2e tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from src.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.blcs.models.blcs_model import BLCSModel
from src.blcs.simulation.cell_manager import ShotCategory
from src.blcs.training.lightning_module import BLCSLightningModule


def make_minimal_blcs_scene(*, scene_id: str = "scene_000000") -> BLCSSceneData:
    """Create a minimal BLCS scene for testing.

    This function creates a minimal BLCSSceneData object with:
    - 30 frames (1 second at 30 fps)
    - Simple parabolic trajectory
    - 1 camera

    Args:
        scene_id: Scene identifier

    Returns:
        BLCSSceneData: Minimal scene data for testing

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

    # Create minimal camera
    camera_params = {
        "camera_idx": 0,
        "K": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
        "image_size": [1280, 720],
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t": [0.0, 0.0, 5.0],
    }

    # Simple UV projection (random for testing)
    ball_uv = np.random.rand(T, 2).astype(np.float32)
    ball_visible = np.ones(T, dtype=bool)

    court_kp_uv = np.random.rand(20, 2).astype(np.float32) * [1280, 720]
    court_kp_visible = np.ones(20, dtype=bool)

    camera = CameraData(
        camera_params=camera_params,
        ball_uv=ball_uv,
        ball_visible=ball_visible,
        ball_visibility_ratio=1.0,
        court_kp_uv=court_kp_uv,
        court_kp_visible=court_kp_visible,
        court_visibility_count=20.0,
    )

    return BLCSSceneData(
        scene_id=scene_id,
        from_cell=0,
        from_side="near",
        category=ShotCategory.IN_COURT,  # Use valid enum value (not NORMAL_RALLY)
        to_cell=6,
        ball_pos_world=ball_pos_world,
        ball_pos_norm=ball_pos_norm,
        ball_vel_world=ball_vel_world,
        t_net=t_net,
        t_fence=t_fence,
        t_bounce1=t_bounce1,
        t_bounce2=t_bounce2,
        cameras=[camera],
        num_cameras_sampled=1,
        fps_out=fps,
        sim_fps=240,
    )


def create_minimal_blcs_dataset(
    output_dir: Path | str,
    num_scenes: int = 10,
) -> Path:
    """Create a minimal BLCS dataset for testing.

    This function creates a dataset with:
    - `num_scenes` scene files (.npz)
    - train.txt, val.txt, test.txt split files
    - Dataset metadata files

    Args:
        output_dir: Output directory for the dataset
        num_scenes: Number of scenes to generate (default: 10)

    Returns:
        Path: Output directory containing the dataset

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = BLCSDatasetWriter(output_dir=output_dir)

    # Generate scenes
    for i in range(num_scenes):
        scene_id = f"scene_{i:06d}"
        scene = make_minimal_blcs_scene(scene_id=scene_id)
        writer.save_scene(scene)

    # Create split files
    scene_ids = [f"scene_{i:06d}" for i in range(num_scenes)]

    # 70% train, 15% val, 15% test
    num_train = int(num_scenes * 0.7)
    num_val = int(num_scenes * 0.15)

    train_ids = scene_ids[:num_train]
    val_ids = scene_ids[num_train : num_train + num_val]
    test_ids = scene_ids[num_train + num_val :]

    (output_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (output_dir / "val.txt").write_text("\n".join(val_ids) + "\n")
    (output_dir / "test.txt").write_text("\n".join(test_ids) + "\n")

    return output_dir


def create_minimal_blcs_checkpoint(checkpoint_path: Path | str) -> Path:
    """Create a minimal BLCS checkpoint for testing.

    This creates a minimal PyTorch Lightning checkpoint that can be loaded
    by BLCSPredictor.load_from_checkpoint().

    Args:
        checkpoint_path: Path where checkpoint will be saved

    Returns:
        Path: Checkpoint path

    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create minimal model
    model = BLCSModel(
        input_dim=42,  # 2 (ball_uv) + 20*2 (court_kp)
        hidden_dim=64,
        output_dim=3,  # 3D position
        num_layers=2,
        dropout=0.1,
    )

    # Create Lightning module
    config = {
        "model": {
            "input_dim": 42,
            "hidden_dim": 64,
            "output_dim": 3,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
    }

    lightning_module = BLCSLightningModule(config=config)
    lightning_module.model = model

    # Create checkpoint manually
    checkpoint = {
        "state_dict": lightning_module.state_dict(),
        "hyper_parameters": config,
        "epoch": 0,
        "global_step": 0,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path

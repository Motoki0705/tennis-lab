"""PLCS test fixtures for e2e tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch

from src.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.plcs.models.plcs_model import PLCSModel
from src.plcs.training.lightning_module import PLCSLightningModule
from src.plcs.models.plcs_sequence_model import PLCSSequenceModel
from src.plcs.training.sequence_lightning_module import PLCSSequenceLightningModule


def make_minimal_plcs_scene(*, scene_id: str = "scene_000000") -> SceneData:
    """Create a minimal PLCS scene for testing.

    This function creates a minimal SceneData object with:
    - 64 frames (enough for sequence model with seq_len=32)
    - 5 joints (reduced from full 17)
    - 1 camera

    Based on _make_dummy_scene() from tests/unit/plcs/test_dataset_io.py

    Args:
        scene_id: Scene identifier

    Returns:
        SceneData: Minimal scene data for testing

    """
    num_frames = 64  # Needs to be >= seq_len (32) for sequence models
    num_joints = 5
    num_cameras = 1

    meta = {
        "scene_id": scene_id,
        "motion_source": "e2e_test",
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
            "center": [0.0, 0.0, 5.0],
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "f": 1000.0,
            "cx": 640.0,
            "cy": 360.0,
            "w": 1280,
            "h": 720,
        }
        # Normalized keypoints (values in [0, 1])
        human_kp_uv: npt.NDArray[np.float32] = (
            np.random.rand(num_frames, 17, 2).astype(np.float32)
        )
        court_kp_uv: npt.NDArray[np.float32] = (
            np.random.rand(num_frames, 20, 2).astype(np.float32)
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
                human_visibility_ratio=1.0,
                court_visibility_count=20.0,
            )
        )

    return SceneData(
        meta=meta,
        position=position,
        rotation=rotation,
        canonical_pose_3d=canonical_pose_3d,
        cameras=cameras,
    )


def create_minimal_plcs_dataset(
    output_dir: Path | str,
    num_scenes: int = 10,
) -> Path:
    """Create a minimal PLCS dataset for testing.

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

    writer = PLCSDatasetWriter(output_dir=output_dir)

    # Generate scenes
    for i in range(num_scenes):
        scene_id = f"scene_{i:06d}"
        scene = make_minimal_plcs_scene(scene_id=scene_id)
        writer.save_scene(scene)

    # Create split files (include .npz extension for dataset loader compatibility)
    scene_ids = [f"scene_{i:06d}.npz" for i in range(num_scenes)]

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


def create_minimal_plcs_checkpoint(checkpoint_path: Path | str) -> Path:
    """Create a minimal PLCS checkpoint for testing.

    This creates a minimal PyTorch Lightning checkpoint that can be loaded
    by PLCSPredictor.load_from_checkpoint().

    Args:
        checkpoint_path: Path where checkpoint will be saved

    Returns:
        Path: Checkpoint path

    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create model matching default config
    # See: src/plcs/configs/model/frame.yaml
    model = PLCSModel(
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_transformer=True,
        use_combined_head=False,
    )

    # Create Lightning module with matching config
    config = {
        "model": {
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
            "use_transformer": True,
            "use_combined_head": False,
        },
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
    }

    lightning_module = PLCSLightningModule(config=config)
    lightning_module.model = model

    # Save checkpoint
    trainer = pl.Trainer(
        default_root_dir=str(checkpoint_path.parent),
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )

    # Create checkpoint with pytorch-lightning_version
    checkpoint = {
        "state_dict": lightning_module.state_dict(),
        "hyper_parameters": config,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl.__version__,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def create_minimal_plcs_sequence_checkpoint(checkpoint_path: Path | str) -> Path:
    """Create a minimal PLCS sequence model checkpoint for testing.

    This creates a minimal PyTorch Lightning checkpoint for the sequence model
    that can be loaded by PLCSSequencePredictor.load_from_checkpoint().

    Args:
        checkpoint_path: Path where checkpoint will be saved

    Returns:
        Path: Checkpoint path

    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create model matching sequence config
    # See: src/plcs/configs/model/sequence.yaml
    model = PLCSSequenceModel(
        hidden_dim=256,
        num_layers=4,  # Use smaller for faster test
        num_heads=8,
        dropout=0.1,
        max_seq_len=120,
    )

    # Create Lightning module with matching config
    config = {
        "model": {
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
            "max_seq_len": 120,
        },
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
    }

    lightning_module = PLCSSequenceLightningModule(config=config)
    lightning_module.model = model

    # Create checkpoint with pytorch-lightning_version
    checkpoint = {
        "state_dict": lightning_module.state_dict(),
        "hyper_parameters": config,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl.__version__,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path

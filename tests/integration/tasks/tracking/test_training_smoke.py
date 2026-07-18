"""One-step CPU training smoke tests for canonical track-query datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir

from src.tasks.blcs.data.tracking_datamodule import BLCSTrackingDataModule
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData
from src.tasks.blcs.generate_dataset.scene_generator import CameraData as BLCSCameraData
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.scene_generator import CameraData as PLCSCameraData
from src.tasks.plcs.generate_dataset.scene_generator import SceneData
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _write_splits(root: Path, names: dict[str, list[str]]) -> None:
    for split, scene_ids in names.items():
        (root / f"{split}.txt").write_text("\n".join(scene_ids) + "\n")


def _materialize_blcs(root: Path) -> None:
    writer = BLCSDatasetWriter(root)
    names: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    for split_index, split in enumerate(names):
        for index in range(4):
            scene_id = f"scene_{split}_{index:03d}"
            names[split].append(scene_id)
            frames, objects, cameras = 8, 3, 2
            positions = torch.rand(frames, objects, 3) + split_index / 100.0
            camera_rows = [
                BLCSCameraData(
                    camera_params={
                        "C": [0, 0, 5],
                        "R": np.eye(3).tolist(),
                        "f": 1,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 1,
                        "h": 1,
                    },
                    ball_uv=np.random.default_rng(index + camera).random(
                        (frames, objects, 2), dtype=np.float32
                    ),
                    ball_visible=np.ones((frames, objects), dtype=bool),
                    ball_visibility_ratio=1.0,
                    court_kp_uv=np.zeros((20, 2), dtype=np.float32),
                    court_kp_visible=np.ones(20, dtype=bool),
                    court_visibility_count=20.0,
                )
                for camera in range(cameras)
            ]
            writer.save_scene(
                BLCSSceneData(
                    scene_id=scene_id,
                    initial_from_cell=0,
                    initial_from_side="near",
                    rally_length=1,
                    end_reason="test",
                    winner_side=None,
                    shots=[],
                    ball_pos_world=positions,
                    ball_pos_norm=positions,
                    ball_vel_world=torch.zeros_like(positions),
                    cameras=camera_rows,
                    num_cameras_sampled=cameras,
                    fps_out=30,
                    sim_fps=120,
                    physics_config_dict={},
                    court_config_dict={},
                    ball_present=torch.ones(frames, objects, dtype=torch.bool),
                    num_balls=objects,
                )
            )
    _write_splits(root, names)


def _materialize_plcs(root: Path) -> None:
    writer = PLCSDatasetWriter(root)
    names: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    for split_index, split in enumerate(names):
        for index in range(4):
            scene_id = f"scene_{split}_{index:03d}"
            names[split].append(scene_id)
            frames, objects, cameras = 8, 3, 2
            position = np.random.default_rng(index).random(
                (frames, objects, 3), dtype=np.float32
            )
            rotation: np.ndarray = np.zeros((frames, objects, 2), dtype=np.float32)
            rotation[..., 0] = 1.0
            camera_rows = [
                PLCSCameraData(
                    camera_params={
                        "C": [0, 0, 5],
                        "R": np.eye(3).tolist(),
                        "f": 1,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 1,
                        "h": 1,
                    },
                    human_kp_uv=np.random.default_rng(index + camera).random(
                        (frames, objects, 17, 2), dtype=np.float32
                    ),
                    court_kp_uv=np.zeros((frames, 20, 2), dtype=np.float32),
                    human_kp_visible=np.ones((frames, objects, 17), dtype=bool),
                    court_kp_visible=np.ones((frames, 20), dtype=bool),
                    human_visibility_ratio=1.0,
                    court_visibility_count=20.0,
                )
                for camera in range(cameras)
            ]
            writer.save_scene(
                SceneData(
                    meta={
                        "scene_id": scene_id,
                        "motion_source": "test",
                        "motion_category": "test",
                        "gender": "neutral",
                        "fps": 30.0,
                        "num_frames": frames,
                        "initial_position": (0.0, 0.0),
                        "initial_yaw": 0.0,
                        "num_cameras_sampled": cameras,
                    },
                    position=position + split_index / 100.0,
                    rotation=rotation,
                    canonical_pose_3d=np.zeros(
                        (frames, objects, 17, 3), dtype=np.float32
                    ),
                    cameras=camera_rows,
                    human_kp_3d=np.zeros((frames, objects, 17, 3), dtype=np.float32),
                    person_present=np.ones((frames, objects), dtype=bool),
                    num_persons=objects,
                )
            )
    _write_splits(root, names)


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_chunked_tracking_reloads_train_dataloader_each_epoch(task: str) -> None:
    """Retired chunks must not remain referenced by Lightning's cached loader."""
    config_dir = Path(f"src/tasks/{task}/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="train_tracking_chunked")

    assert config.data.chunk.epochs_per_chunk == 20
    assert config.data.chunk.scenes_per_chunk == 1000
    assert config.data.chunk.prefetch_chunks == 5
    assert config.data.chunk.generation_workers == 16
    assert config.data.num_workers == 4
    assert list(config.data.seq_len_range) == [512, 1024]
    assert list(config.data.num_views_range) == [3, 5]
    assert config.training.trainer.max_epochs == 100
    assert config.training.trainer.check_val_every_n_epoch == 5
    assert config.training.early_stopping.enabled is False
    assert config.training.qualitative_logging.enabled is True
    assert config.training.qualitative_logging.every_n_epochs == 10
    assert config.training.trainer.reload_dataloaders_every_n_epochs == 1


@pytest.mark.parametrize(
    ("task", "datamodule_class", "module_class", "materialize"),
    [
        (
            "blcs",
            BLCSTrackingDataModule,
            BLCSTrackingLightningModule,
            _materialize_blcs,
        ),
        (
            "plcs",
            PLCSTrackingDataModule,
            PLCSTrackingLightningModule,
            _materialize_plcs,
        ),
    ],
)
def test_tracking_task_runs_one_training_and_validation_step(
    tmp_path: Path,
    task: str,
    datamodule_class: type[Any],
    module_class: type[Any],
    materialize: Any,
) -> None:
    config_dir = Path(f"src/tasks/{task}/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="train_tracking")
    dataset_dir = tmp_path / task
    materialize(dataset_dir)
    config.data.scene_dir = str(dataset_dir)
    config.data.batch_size = 2
    config.data.seq_len_range = [8, 8]
    config.data.num_views_range = [2, 2]
    config.data.camera_mode = "first"
    datamodule = datamodule_class(config)
    datamodule.setup("fit")
    first_val = datamodule.val_dataset[0]
    repeated_val = datamodule.val_dataset[0]
    assert all(
        torch.equal(value, repeated_val[key]) for key, value in first_val.items()
    )
    model_inputs = module_class._model_inputs(first_val)
    assert {
        "ball_score",
        "ball_candidate_mask",
        "human_vis",
        "detection_score",
        "bbox",
    }.isdisjoint(model_inputs)
    assert "court_vis" in model_inputs
    if task == "blcs":
        assert "ball_visible" in model_inputs
    trainer = pl.Trainer(
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module_class(config), datamodule=datamodule)
    assert trainer.global_step == 1

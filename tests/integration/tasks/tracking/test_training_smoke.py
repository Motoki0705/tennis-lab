"""One-step CPU training smoke tests for canonical track-query datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir

from src.tasks.base.data.court_peaks import (
    COURT_SEMANTIC_CLASS_NAMES,
    CourtPeakFrame,
)
from src.tasks.blcs.data.tracking_datamodule import BLCSTrackingDataModule
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData
from src.tasks.blcs.generate_dataset.scene_generator import CameraData as BLCSCameraData
from src.tasks.blcs.model_io import compose_blcs_track_query_model_io
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.tasks.court_detection.model_io import CourtKeypointPrediction
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
                        "C": [0, -20 if camera == 0 else 20, 5],
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
                        "C": [0, -20 if camera == 0 else 20, 5],
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
    assert config.model.court_observation_profile == "kp14_reference_baseline"
    assert config.model.kp7_camera_rope_enabled is False


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_tracking_ablation_presets_compose_owned_observation_fields(task: str) -> None:
    config_dir = Path(f"src/tasks/{task}/configs").resolve()
    expected_profiles = {
        "track_query_kp14": "kp14_reference_baseline",
        "track_query_kp7_no_reference": "kp7_no_reference",
        "track_query_kp7_reference": "kp7_reference",
    }

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = {
            name: compose(config_name="train_tracking", overrides=[f"model={name}"])
            for name in expected_profiles
        }

    for name, expected_profile in expected_profiles.items():
        assert composed[name].model.court_observation_profile == expected_profile
        assert composed[name].model.kp7_camera_rope_enabled is False
    if task == "blcs":
        assert all(
            config.model.observation_fusion == "linear"
            for config in composed.values()
        )


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
    config.paths.project_root = str(tmp_path)
    config.paths.data_root = "data"
    dataset_dir = tmp_path / "data" / task
    materialize(dataset_dir)
    config.data.scene_dir = task
    config.data.batch_size = 2
    config.data.seq_len_range = [8, 8]
    config.data.num_views_range = [2, 2]
    config.data.camera_mode = "first"
    config.training.warmup_steps = 0
    datamodule = datamodule_class(config)
    datamodule.setup("fit")
    first_val = datamodule.val_dataset[0]
    repeated_val = datamodule.val_dataset[0]
    assert all(
        torch.equal(value, repeated_val[key]) for key, value in first_val.items()
    )
    if task == "blcs":
        model_io = compose_blcs_track_query_model_io(config)
        lightning_module = module_class(config, model_io=model_io)
    else:
        lightning_module = module_class(config)
        model_io = lightning_module.model_io
    first_batch = next(iter(datamodule.val_dataloader()))
    model_inputs = model_io.build_call(first_batch).kwargs
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
    trainer.fit(lightning_module, datamodule=datamodule)
    assert trainer.global_step == 1


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
def test_court_predictor_kp7_connects_directly_to_tracking_backward(
    tmp_path: Path,
    task: str,
    datamodule_class: type[Any],
    module_class: type[Any],
    materialize: Any,
) -> None:
    config_dir = Path(f"src/tasks/{task}/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=["model=track_query_kp7_reference"],
        )
    config.paths.project_root = str(tmp_path)
    config.paths.data_root = "data"
    materialize(tmp_path / "data" / task)
    config.data.scene_dir = task
    config.data.batch_size = 1
    config.data.seq_len_range = [8, 8]
    config.data.num_views_range = [2, 2]
    config.data.camera_mode = "first"
    config.data.num_workers = 0
    datamodule = datamodule_class(config)
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))

    batch_size, views, frames = batch["court_peak_uv"].shape[:3]
    source_frames: list[CourtPeakFrame] = []
    generator = torch.Generator().manual_seed(719)
    for batch_index in range(batch_size):
        for view_index in range(views):
            for frame_index in range(frames):
                capacity = 5 if frame_index % 2 == 0 else 4
                keypoints = torch.rand(7, capacity, 2, generator=generator)
                keypoints *= torch.tensor([639.0, 359.0])
                valid = torch.ones(7, capacity, dtype=torch.bool)
                valid[0] = False  # explicit zero-peak class
                valid[2, 4:] = False
                score = torch.rand(7, capacity, generator=generator)
                covariance = (
                    torch.eye(2)
                    .reshape(1, 1, 2, 2)
                    .expand(7, capacity, 2, 2)
                )
                if frame_index % 2 == 0:
                    prediction = CourtKeypointPrediction(
                        keypoints=keypoints,
                        scores=score,
                        valid=valid,
                        covariance=covariance,
                        heatmaps=torch.zeros(7, 2, 2),
                        semantic_class_names=COURT_SEMANTIC_CLASS_NAMES,
                        image_size_hw=(360, 640),
                    )
                    source = CourtPeakFrame.from_prediction(
                        prediction,
                        batch_index=batch_index,
                        view_index=view_index,
                        frame_index=frame_index,
                    )
                else:
                    source = CourtPeakFrame.from_dataset_output(
                        {
                            "keypoints": keypoints,
                            "scores": score,
                            "valid": valid,
                            "covariance": covariance,
                            "image_size": torch.tensor([360, 640]),
                            "semantic_class_names": COURT_SEMANTIC_CLASS_NAMES,
                        },
                        batch_index=batch_index,
                        view_index=view_index,
                        frame_index=frame_index,
                    )
                source_frames.append(source)
    for key in (
        "court_peak_uv",
        "court_peak_score",
        "court_peak_covariance",
        "court_peak_valid",
    ):
        del batch[key]
    batch["court_peak_frames"] = source_frames

    if task == "blcs":
        model_io = compose_blcs_track_query_model_io(config)
        module = module_class(config, model_io=model_io)
        reference = int(batch["reference_view_index"][0])
        batch["ball_visible"][:, reference] = False
        batch["ball_score"][:, reference] = 0
    else:
        module = module_class(config)
        reference = int(batch["reference_view_index"][0])
        batch["detection_mask"][:, reference] = False
        batch["joint_visibility"][:, reference] = False
        batch["detection_score"][:, reference] = 0
    result = module.compute_tracking_step(batch, compute_metrics=True)
    result.losses["total"].backward()
    output = module.tracking_prediction_result(result)
    payload = module.test_prediction_payload(batch, output)

    assert torch.isfinite(result.losses["total"])
    assert "reference_consistency_y_m" in result.metrics
    assert result.counterfactual_prediction is not None
    assert payload["counterfactual_pred_position"].shape == payload[
        "pred_position"
    ].shape
    assert payload["counterfactual_reference_view_index"].shape == payload[
        "reference_view_index"
    ].shape
    assert payload["counterfactual_orientation_sign"].shape == payload[
        "orientation_sign"
    ].shape
    if task == "plcs":
        assert payload["counterfactual_pred_rotation"].shape == payload[
            "pred_rotation"
        ].shape

"""Canonical compact-store smoke tests for BLCS and PLCS training consumers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor, nn

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.composition import (
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinates,
)
from src.synthetic_data_generation.dataset.blcs.assembler import (
    assemble_blcs_dataset,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSCompositionAssets,
    BLCSTrack,
    BLCSTrajectory,
)
from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    BLCSRenderAttempt,
    BLCSRenderedTrajectory,
    build_blcs_sample_metadata,
)
from src.synthetic_data_generation.dataset.blcs.timeline import build_blcs_plans
from src.synthetic_data_generation.dataset.plcs.assembler import (
    PLCS_DATASET_SCHEMA,
)
from src.synthetic_data_generation.dataset.runtime import (
    BACKGROUND_STORE_SCHEMA,
    ChunkWriter,
    DatasetPerformanceBudget,
    ForegroundDelta,
    ForegroundDeltaBatch,
    PerformanceTimer,
    RenderSampleKey,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.tasks.base.generate_dataset.camera_profiles import CameraProfileConfig
from src.tasks.blcs.data.datamodule import BLCSDataModule
from src.tasks.blcs.data.dataset import (
    BallTrajectoryDataset,
    collate_multiview_trajectories,
)
from src.tasks.blcs.data.tracking_dataset import BLCSTrackingDataset
from src.tasks.blcs.model_io import MultiViewTrajectoryModelIOAdapter
from src.tasks.plcs.data import PLCSDataModule, SceneDataset
from src.tasks.plcs.data.tracking_dataset import PLCSTrackingDataset
from src.tasks.plcs.model_io import (
    PLCSInputProfile,
    PLCSModelIOAdapter,
)
from src.utils.paths import PROJECT_ROOT

_RELATIVE_DATA_ROOT = Path("canonical-consumer") / "datasets"


def _compose_task_config(
    task: str,
    overrides: Sequence[str],
    *,
    config_name: str = "train",
) -> DictConfig:
    config_dir = PROJECT_ROOT / "src" / "tasks" / task / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


def _camera_profile() -> CameraProfileConfig:
    slots = []
    for index, x_value in enumerate((-5.0, 5.0)):
        slots.append(
            {
                "slot_id": f"camera-{index}",
                "position_x_m": [x_value, x_value],
                "position_y_m": [-18.0, -18.0],
                "height_m": [8.0, 8.0],
                "look_at_x_m": [0.0, 0.0],
                "look_at_y_m": [0.0, 0.0],
                "look_at_height_m": [1.0, 1.0],
                "hfov_degrees": [110.0, 110.0],
            }
        )
    return CameraProfileConfig.from_mapping(
        {
            "profile": "broadcast",
            "image_size": [32, 24],
            "expected_camera_count": 2,
            "slots": slots,
        }
    )


def _blcs_assets() -> BLCSCompositionAssets:
    return BLCSCompositionAssets(
        background=GaussianAsset(
            asset_id="background",
            asset_class="court",
            role=GaussianAssetRole.BACKGROUND,
            coordinates=GaussianCoordinates.scene(),
            gaussian_count=8,
            feature_dim=8,
            floating_dtype="float32",
            appearance_model="nht-deferred",
            appearance_space="test-space",
        ),
        ball=GaussianAsset(
            asset_id="ball",
            asset_class="ball",
            role=GaussianAssetRole.MOVABLE,
            coordinates=GaussianCoordinates.asset_local_metres(),
            gaussian_count=8,
            feature_dim=8,
            floating_dtype="float32",
            appearance_model="nht-deferred",
            appearance_space="test-space",
        ),
        ball_radius_m=0.0335,
    )


def _court_layout() -> MultiCourtLayout:
    identity = RigidTransform.identity()
    return MultiCourtLayout(
        courts=(
            CourtInstance(
                court_instance_id="court-0",
                candidate_id="candidate-0",
                scene_from_court=identity,
                court_from_scene=identity,
                fit_status="accepted",
                fit_metrics={"error": 0.0},
                holdout_status="accepted",
                holdout_metrics={"error": 0.0},
            ),
        ),
        complex_bounds_scene=(-10.0, -20.0, -1.0, 10.0, 20.0, 10.0),
        primary_court_instance_id="court-0",
    )


def _blcs_trajectory() -> BLCSTrajectory:
    positions = np.asarray([[[-0.2, 0.0, 1.5]], [[0.2, 0.0, 1.5]]], dtype=np.float64)
    velocities = np.gradient(positions, axis=0)
    return BLCSTrajectory(
        trajectory_id="trajectory-test",
        split="test",
        fps=30.0,
        positions_court_m=positions,
        velocities_court_mps=velocities,
        present=np.ones((2, 1), dtype=np.bool_),
        tracks=(
            BLCSTrack(
                object_id="ball-test",
                source_trajectory_id="trajectory-test",
                source_frame_indices=(0, 1),
            ),
        ),
        source_metadata={"source": "canonical-consumer-smoke"},
    )


def _write_background_store(
    root: Path, *, scene_id: str, cameras: Sequence[SceneCamera]
) -> None:
    root.mkdir(parents=True)
    records = []
    for camera in cameras:
        camera_root = root / camera.camera_id
        camera_root.mkdir()
        np.save(
            camera_root / "rgb.npy",
            np.zeros((camera.height, camera.width, 3), dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            camera_root / "alpha.npy",
            np.ones((camera.height, camera.width, 1), dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            camera_root / "depth-metric.npy",
            np.full((camera.height, camera.width, 1), 100.0, dtype=np.float32),
            allow_pickle=False,
        )
        records.append(
            {
                "camera_id": camera.camera_id,
                "width": camera.width,
                "height": camera.height,
                "rgb": f"{camera.camera_id}/rgb.npy",
                "alpha": f"{camera.camera_id}/alpha.npy",
                "depth": f"{camera.camera_id}/depth-metric.npy",
            }
        )
    (root / "backgrounds.json").write_text(
        json.dumps(
            {
                "schema": BACKGROUND_STORE_SCHEMA,
                "scene_id": scene_id,
                "depth_coordinate_space": "metric_scene_metres",
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def _foreground_delta(frame_index: int, camera_id: str) -> ForegroundDelta:
    return ForegroundDelta(
        key=RenderSampleKey(frame_index, camera_id),
        pixel_indices=np.asarray([0], dtype=np.int32),
        rgb=np.asarray([[1.0, 0.8, 0.0]], dtype=np.float32),
        alpha=np.asarray([1.0], dtype=np.float32),
        depth=np.asarray([2.0], dtype=np.float32),
        instance_ids=np.asarray([1], dtype=np.int32),
    )


def _write_blcs_store(data_root: Path) -> Path:
    owner = data_root / _RELATIVE_DATA_ROOT / "blcs"
    snapshot = data_root / ".transactions" / "blcs_dataset" / "snapshot"
    snapshot.mkdir(parents=True)
    plans = build_blcs_plans(
        (_blcs_trajectory(),),
        dataset_scene_id="B00",
        layout=_court_layout(),
        camera_config=_camera_profile(),
        assets=_blcs_assets(),
        seed=19,
        chunk_size_frames=2,
    )
    rendered = []
    for plan in plans:
        cameras = tuple(value.scene_camera for value in plan.camera_rig.cameras)
        trajectory_root = snapshot / "samples" / plan.source.trajectory_id
        trajectory_root.mkdir(parents=True)
        background_root = trajectory_root / "backgrounds"
        _write_background_store(
            background_root, scene_id=plan.dataset_scene_id, cameras=cameras
        )
        writer = ChunkWriter(
            trajectory_root / "chunks",
            attempt_token="consumer-smoke",
            camera_ids=tuple(camera.camera_id for camera in cameras),
            width=cameras[0].width,
            height=cameras[0].height,
        )
        readers = []
        for chunk in plan.chunks:
            samples = tuple(
                (
                    frame_index,
                    camera_index,
                    _foreground_delta(frame_index, camera.camera_id),
                )
                for frame_index in chunk.frame_indices
                for camera_index, camera in enumerate(cameras)
            )
            deltas = tuple(delta for _, _, delta in samples)
            labels = tuple(
                build_blcs_sample_metadata(
                    plan=plan,
                    source_frame_index=frame_index,
                    camera_index=camera_index,
                    chunk_index=chunk.chunk_index,
                    delta=delta,
                )
                for frame_index, camera_index, delta in samples
            )
            readers.append(
                writer.write(
                    ForegroundDeltaBatch(
                        chunk_id=f"chunk-{chunk.chunk_index:06d}",
                        deltas=deltas,
                        metadata=labels,
                    )
                )
            )
        rendered.append(
            BLCSRenderedTrajectory(
                trajectory_id=plan.source.trajectory_id,
                directory=trajectory_root,
                background_directory=background_root,
                chunk_readers=tuple(readers),
                rendered_visible_object_views=plan.source.frame_count * len(cameras),
            )
        )
    assemble_blcs_dataset(
        snapshot,
        plans=plans,
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(np.eye(4)),
        render_attempt=BLCSRenderAttempt(
            attempt_token="consumer-smoke",
            trajectories=tuple(rendered),
            execution_device="test-cpu-oracle",
            cuda_peak_bytes=0,
            nht_invocations=len(plans),
            background_cache_misses=sum(len(plan.camera_rig.cameras) for plan in plans),
            generated_bytes=100_000_000,
        ),
        performance_timer=PerformanceTimer(),
        performance_budget=DatasetPerformanceBudget(
            maximum_wall_seconds=60.0,
            maximum_published_bytes=100_000_000,
            maximum_published_fraction_of_dense_reference=1.0,
            maximum_nht_invocations=1,
            maximum_background_cache_misses=2,
            maximum_complete_array_scans_per_sample=1,
            maximum_batch_frames=2,
            execution_device="test-cpu-oracle",
            require_cuda=False,
        ),
    )
    owner.parent.mkdir(parents=True)
    snapshot.replace(owner)
    return owner


def _write_plcs_store(data_root: Path, cameras: Sequence[SceneCamera]) -> Path:
    root = data_root / _RELATIVE_DATA_ROOT / "plcs"
    _write_background_store(root / "backgrounds", scene_id="B00", cameras=cameras)
    attempt_token = "consumer-smoke-plcs"
    writer = ChunkWriter(
        root / "scenes" / "scene-test" / "chunks",
        attempt_token=attempt_token,
        camera_ids=tuple(camera.camera_id for camera in cameras),
        width=cameras[0].width,
        height=cameras[0].height,
    )
    deltas = tuple(
        _foreground_delta(frame_index, camera.camera_id)
        for frame_index in range(2)
        for camera in cameras
    )
    chunk = writer.write(
        ForegroundDeltaBatch(
            chunk_id="chunk-000000",
            deltas=deltas,
            metadata=tuple({} for _ in deltas),
        )
    )
    present: NDArray[np.bool_] = np.ones((2, 1), dtype=np.bool_)
    rotation: NDArray[np.float32] = np.zeros((2, 1, 2), dtype=np.float32)
    rotation[..., 0] = 1.0
    supervision = root / "scenes" / "scene-test" / "supervision.npz"
    np.savez(
        supervision,
        human_kp=np.full((2, 2, 1, 17, 2), 0.5, dtype=np.float32),
        human_vis=np.ones((2, 2, 1, 17), dtype=np.bool_),
        court_kp=np.full((2, 2, 20, 2), 0.5, dtype=np.float32),
        court_vis=np.ones((2, 2, 20), dtype=np.bool_),
        human_mask=np.ones((2, 2, 1), dtype=np.bool_),
        position=np.zeros((2, 1, 3), dtype=np.float32),
        position_court_m=np.zeros((2, 1, 3), dtype=np.float32),
        rotation=rotation,
        present=present,
        human_kp_3d=np.zeros((2, 1, 17, 3), dtype=np.float32),
        canonical_pose_3d=np.zeros((2, 1, 52, 3), dtype=np.float32),
    )
    inventory = {
        "source": 2,
        "planned": 2,
        "rendered": 2,
        "labelled": 2,
        "first_frame": 0,
        "last_frame": 1,
    }
    scene = {
        "scene_id": "scene-test",
        "split": "test",
        "frame_inventory": inventory,
        "tracks": [
            {
                "object_id": "player-test",
                "instance_id": 1,
                "asset_id": "avatar-test",
                "start_frame": 0,
                "stop_frame": 2,
                "anchor_position_court_m": [0.0, 0.0, 0.0],
                "yaw_radians": 0.0,
            }
        ],
        "cameras": [
            {
                "slot_id": f"camera-{index}",
                "court_local_center_m": [0.0, 0.0, 1.0],
                "court_local_look_at_m": [0.0, 1.0, 1.0],
                "hfov_degrees": 60.0,
                "camera": camera.to_dict(),
            }
            for index, camera in enumerate(cameras)
        ],
    }
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": PLCS_DATASET_SCHEMA,
                "scene_id": "B00",
                "domain": "plcs",
                "frame_inventory": inventory,
                "target_courts": [],
                "metadata": {"logical_scenes": [scene]},
                "diagnostics": [],
                "storage": {
                    "layout": "shared-background-plus-per-scene-foreground-delta",
                    "background_store": "backgrounds",
                    "scenes": [
                        {
                            "scene_id": "scene-test",
                            "chunks": [str(chunk.directory.relative_to(root))],
                            "attempt_token": attempt_token,
                            "sample_order": "scene-frame-then-configured-camera",
                            "supervision": str(supervision.relative_to(root)),
                            "camera_ids": [camera.camera_id for camera in cameras],
                            "object_ids": ["player-test"],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "diagnostics").mkdir()
    return root


def test_canonical_datamodules_retain_all_views_through_model_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    blcs_root = _write_blcs_store(data_root)
    blcs_cameras = tuple(
        _camera.scene_camera
        for _camera in build_blcs_plans(
            (_blcs_trajectory(),),
            dataset_scene_id="B00",
            layout=_court_layout(),
            camera_config=_camera_profile(),
            assets=_blcs_assets(),
            seed=19,
            chunk_size_frames=2,
        )[0].camera_rig.cameras
    )
    plcs_root = _write_plcs_store(data_root, blcs_cameras)

    from src.tasks.blcs.data import datamodule as blcs_datamodule_module

    monkeypatch.setattr(blcs_datamodule_module, "PROJECT_ROOT", tmp_path)
    blcs_config = _compose_task_config(
        "blcs",
        (
            "~camera",
            "model=multiview",
            f"data.dataset_dir={_RELATIVE_DATA_ROOT.as_posix()}/blcs",
            "data.batch_size=1",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.seq_len_range=[2,2]",
            "data.augmentation.enabled=false",
        ),
    )
    blcs_module = BLCSDataModule(blcs_config, collate_fn=collate_multiview_trajectories)
    blcs_module.setup("test")
    assert isinstance(blcs_module.test_dataset, BallTrajectoryDataset)
    blcs_batch = cast(dict[str, Tensor], next(iter(blcs_module.test_dataloader())))
    blcs_index = blcs_module.test_dataset.index[0]
    blcs_all_views = blcs_module.test_dataset.reader.materialize_all_views(
        blcs_index.trajectory_id
    )
    assert blcs_index.split == "test"
    assert blcs_all_views.index.camera_ids == tuple(
        camera.camera_id for camera in blcs_cameras
    )
    assert blcs_batch["ball_uv"].shape[:3] == (1, 2, 2)
    assert blcs_batch["ball_mask"].bool().all()
    blcs_prepared = MultiViewTrajectoryModelIOAdapter(
        num_court_tokens=20,
        max_seq_len=2,
        predict_velocity=True,
        input_profile="multiview",
        max_num_cameras=6,
    ).build_training_batch(blcs_batch)
    assert cast(Tensor, blcs_prepared.call.kwargs["ball_uv"]).shape == (1, 2, 2, 2)
    assert blcs_prepared.camera_R.shape == (1, 2, 3, 3)
    assert blcs_prepared.loss_mask.shape == (1, 2)

    plcs_config = _compose_task_config(
        "plcs",
        (
            "model=multiview",
            "loss=no_canonical",
            f"paths.data_root={data_root.as_posix()}",
            f"data.dataset_dir={_RELATIVE_DATA_ROOT.as_posix()}/plcs",
            "data.batch_size=1",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.seq_len_range=[2,2]",
            "data.augmentation.enabled=false",
            "model.max_seq_len=2",
        ),
    )
    plcs_module = PLCSDataModule(plcs_config)
    plcs_module.setup("test")
    assert isinstance(plcs_module.test_dataset, SceneDataset)
    plcs_batch = cast(dict[str, Tensor], next(iter(plcs_module.test_dataloader())))
    plcs_index = plcs_module.test_dataset.index[0]
    plcs_all_views = plcs_module.test_dataset.reader.materialize_all_views(
        plcs_index.scene_id
    )
    assert plcs_index.split == "test"
    assert plcs_all_views.index.camera_ids == tuple(
        camera.camera_id for camera in blcs_cameras
    )
    assert plcs_batch["human_kp"].shape[:3] == (1, 2, 2)
    assert plcs_batch["human_mask"].bool().all()
    plcs_prepared = PLCSModelIOAdapter(
        model_type=nn.Identity,
        profile=PLCSInputProfile.MULTIVIEW,
        num_court_tokens=20,
        output_rank=3,
        predict_canonical_pose=False,
        predict_auxiliary_position=False,
        max_views=6,
        max_sequence_length=2,
        min_views=2,
    ).prepare_training_batch(plcs_batch)
    assert cast(Tensor, plcs_prepared.call.kwargs["human_kp"]).shape == (
        1,
        2,
        2,
        17,
        2,
    )
    assert plcs_prepared.target_human_mask is not None
    assert plcs_prepared.target_human_mask.shape == (1, 2, 2)

    blcs_tracking_config = _compose_task_config(
        "blcs",
        (
            "data.seq_len_range=[2,2]",
            "data.augmentation.enabled=false",
        ),
        config_name="train_tracking",
    )
    blcs_tracking = BLCSTrackingDataset(
        dataset_dir=blcs_root,
        split="test",
        config=blcs_tracking_config,
        augment=False,
    )[0]
    assert blcs_tracking["ball_uv"].shape == (2, 2, 1, 2)
    assert blcs_tracking["view_mask"].tolist() == [True, True]
    assert blcs_tracking["target_presence"].shape == (2, 4)
    assert blcs_tracking["target_instance_id"][:, 0].tolist() == [0, 0]

    plcs_tracking_config = _compose_task_config(
        "plcs",
        (
            "data.seq_len_range=[2,2]",
            "data.augmentation.enabled=false",
        ),
        config_name="train_tracking",
    )
    plcs_tracking = PLCSTrackingDataset(
        dataset_dir=plcs_root,
        split="test",
        config=plcs_tracking_config,
        augment=False,
    )[0]
    assert plcs_tracking["human_kp"].shape == (2, 2, 1, 17, 2)
    assert plcs_tracking["view_mask"].tolist() == [True, True]
    assert plcs_tracking["target_presence"].shape == (2, 4)
    assert plcs_tracking["target_instance_id"][:, 0].tolist() == [0, 0]

    assert set(path.name for path in blcs_root.iterdir()) == {
        "dataset.json",
        "samples",
        "diagnostics",
    }
    assert set(path.name for path in plcs_root.iterdir()) == {
        "dataset.json",
        "backgrounds",
        "scenes",
        "diagnostics",
    }

"""CourtKP20 artifact contract tests for PLCS chunk generation."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.tasks.base.generate_dataset import (
    apply_court_view_record,
    build_court_view_record,
    resolve_court_keypoint_contract,
    validate_dataset_court_keypoint_contract,
)
from src.tasks.plcs.court_keypoint_contract import PLCS_GENERATED_DATASET_SCHEMA_ID
from src.tasks.plcs.data import chunk_manager as chunk_manager_module
from src.tasks.plcs.data.chunk_manager import _PLCSChunkGenerator
from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene
from src.tasks.plcs.generate_dataset.scene_generator import CameraData, SceneData


def _camera_view_scene() -> SceneData:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    camera_center = [2.0, 12.0, 4.0]
    view = build_court_view_record(
        camera_id="camera_0",
        camera_center_court_m=camera_center,
        contract=contract,
    )
    physical_uv = np.arange(40, dtype=np.float32).reshape(20, 2) / 100.0
    physical_vis = np.arange(20) % 3 != 0
    disk_uv = apply_court_view_record(physical_uv, view, keypoint_axis=0)
    disk_vis = apply_court_view_record(physical_vis, view, keypoint_axis=0)
    assert isinstance(disk_uv, np.ndarray)
    assert isinstance(disk_vis, np.ndarray)
    camera = CameraData(
        camera_params={"C": camera_center, "R": np.eye(3).tolist()},
        human_kp_uv=np.zeros((1, 17, 2), dtype=np.float32),
        court_kp_uv=disk_uv[None],
        human_kp_vis=np.ones((1, 17), dtype=np.bool_),
        court_kp_vis=disk_vis[None],
        human_visibility_ratio=1.0,
        court_visibility_count=float(disk_vis.sum()),
        court_view=view,
    )
    return SceneData(
        meta={
            "scene_id": "scene_000000",
            "motion_source": "fixture",
            "motion_category": "test",
            "gender": "neutral",
            "fps": 30,
            "num_frames": 1,
            "initial_position": [0.0, 0.0],
            "initial_yaw": 0.0,
            "num_cameras_sampled": 1,
        },
        position=np.zeros((1, 3), dtype=np.float32),
        rotation=np.array([[1.0, 0.0]], dtype=np.float32),
        canonical_pose_3d=np.zeros((1, 17, 3), dtype=np.float32),
        cameras=[camera],
        num_persons=1,
        court_keypoint_contract=contract,
    )


def test_chunk_generator_publishes_root_contract_for_versioned_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _camera_view_scene()
    contract = resolve_court_keypoint_contract("camera_view_v2")
    root_contract_observed_before_scene = False

    def _one_scene(**kwargs: object) -> Iterator[SceneData]:
        nonlocal root_contract_observed_before_scene
        del kwargs
        assert not (tmp_path / "scenes" / "scene_000000").exists()
        initial_validation = validate_dataset_court_keypoint_contract(
            tmp_path,
            contract,
            expected_dataset_schema_id=PLCS_GENERATED_DATASET_SCHEMA_ID,
            scene_paths=(),
        )
        assert initial_validation.contract == contract
        assert not initial_validation.legacy_metadata_free
        root_contract_observed_before_scene = True
        yield scene

    monkeypatch.setattr(
        chunk_manager_module,
        "generate_parallel_scenes",
        _one_scene,
    )
    config = OmegaConf.create(
        {"court_keypoints": {"selector": "camera_view_v2"}}
    )
    generator = _PLCSChunkGenerator(
        config=config,
        generator_device="cpu",
        generation_workers=1,
    )

    generator(
        tmp_path,
        num_scenes=1,
        stop_event=threading.Event(),
    )

    assert root_contract_observed_before_scene
    scene_path = tmp_path / "scenes" / "scene_000000"
    assert scene_path.is_dir()
    validation = validate_dataset_court_keypoint_contract(
        tmp_path,
        contract,
        expected_dataset_schema_id=PLCS_GENERATED_DATASET_SCHEMA_ID,
        scene_paths=(scene_path,),
    )
    assert validation.contract == contract
    assert not validation.legacy_metadata_free
    loaded = load_scene(scene_path, court_keypoint_contract=contract)
    assert loaded["cameras"][0]["court_view"] == scene.cameras[0].court_view
    np.testing.assert_array_equal(
        loaded["cameras"][0]["court_kp_uv"],
        scene.cameras[0].court_kp_uv,
    )
    np.testing.assert_array_equal(
        loaded["cameras"][0]["court_kp_vis"],
        scene.cameras[0].court_kp_vis,
    )

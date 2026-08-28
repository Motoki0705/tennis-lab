"""CourtKP20 propagation tests for BLCS chunk generation."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from src.tasks.base.generate_dataset import (
    CourtViewRecord,
    apply_court_view_record,
    build_court_view_record,
    resolve_court_keypoint_contract,
    validate_dataset_court_keypoint_contract,
)
from src.tasks.blcs.data import chunk_manager as chunk_manager_module
from src.tasks.blcs.data.chunk_manager import _BLCSChunkGenerator
from src.tasks.blcs.generate_dataset.io.dataset_io import (
    BLCS_DATASET_SCHEMA_ID,
    load_scene,
)
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
    CameraData,
    GeneratorConfig,
)
from src.utils.schema.court_normalization import (
    normalize_court_position,
    normalize_court_velocity,
)


def _camera_view_scene() -> tuple[
    BLCSSceneData,
    CourtViewRecord,
    np.ndarray,
    np.ndarray,
]:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    center = (2.0, 12.0, 4.0)
    view = build_court_view_record(
        camera_id="cam_0",
        camera_center_court_m=center,
        contract=contract,
    )
    physical_uv = np.arange(40, dtype=np.float32).reshape(20, 2) / 100.0
    physical_vis = np.arange(20) % 3 != 0
    disk_uv = apply_court_view_record(physical_uv, view, keypoint_axis=0)
    disk_vis = apply_court_view_record(physical_vis, view, keypoint_axis=0)
    assert isinstance(disk_uv, np.ndarray)
    assert isinstance(disk_vis, np.ndarray)
    camera = CameraData(
        camera_params={
            "C": list(center),
            "R": np.eye(3, dtype=np.float32).tolist(),
            "f": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "w": 100,
            "h": 80,
        },
        ball_uv=np.full((2, 2), 0.5, dtype=np.float32),
        ball_vis=np.ones(2, dtype=np.bool_),
        ball_visibility_ratio=1.0,
        court_kp_uv=disk_uv,
        court_kp_vis=disk_vis,
        court_visibility_count=float(disk_vis.sum()),
        court_view=view,
    )
    position = torch.tensor(
        [[1.0, 2.0, 0.5], [1.5, 2.5, 0.75]],
        dtype=torch.float32,
    )
    velocity = torch.tensor(
        [[0.5, 1.0, 0.25], [0.75, 1.25, 0.0]],
        dtype=torch.float32,
    )
    scene = BLCSSceneData(
        scene_id="scene_000000",
        initial_from_cell=0,
        initial_from_side="near",
        rally_length=1,
        end_reason="fixture",
        winner_side=None,
        shots=[],
        ball_pos_world=position,
        ball_pos_norm=normalize_court_position(position),
        ball_vel_world=velocity,
        ball_vel_norm=normalize_court_velocity(velocity),
        cameras=[camera],
        num_cameras_sampled=1,
        fps_out=30,
        sim_fps=120,
        physics_config_dict={},
        court_config_dict={},
        num_balls=1,
    )
    return scene, view, disk_uv, disk_vis


def test_chunk_generator_publishes_and_propagates_camera_view_contract_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, expected_view, expected_uv, expected_vis = _camera_view_scene()
    contract = resolve_court_keypoint_contract("camera_view_v2")
    generator_config = cast(
        "GeneratorConfig",
        SimpleNamespace(court_keypoint_contract=contract),
    )
    root_contract_observed_before_scene = False

    def _one_scene(**kwargs: object) -> Iterator[BLCSSceneData]:
        nonlocal root_contract_observed_before_scene
        assert kwargs["generator_config"] is generator_config
        assert not (tmp_path / "scenes" / "scene_000000").exists()
        initial_validation = validate_dataset_court_keypoint_contract(
            tmp_path,
            contract,
            expected_dataset_schema_id=BLCS_DATASET_SCHEMA_ID,
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
    generator = _BLCSChunkGenerator(
        generator_config=generator_config,
        generator_device="cpu",
        generation_workers=1,
        generation_chunksize=1,
        generation_seed=799,
        multi_object=True,
        timeline_config=None,
        maximum_physics_attempts_per_object=1,
    )

    generator(
        tmp_path,
        num_scenes=1,
        stop_event=threading.Event(),
    )

    assert root_contract_observed_before_scene
    scene_path = tmp_path / "scenes" / "scene_000000"
    loaded = load_scene(scene_path, court_keypoint_contract=contract)
    assert loaded["cameras"][0]["court_view"] == expected_view
    np.testing.assert_array_equal(loaded["cameras"][0]["court_kp_uv"], expected_uv)
    np.testing.assert_array_equal(
        loaded["cameras"][0]["court_kp_vis"],
        expected_vis,
    )
    root_metadata = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert root_metadata["stats"]["total_scenes"] == 1

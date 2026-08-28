"""BLCS normalized scene serialization contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter, load_scene
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.utils.schema.court_normalization import (
    CourtCoordinateContractError,
    denormalize_court_position,
    denormalize_court_velocity,
    normalize_court_position,
    normalize_court_velocity,
)


def _scene() -> BLCSSceneData:
    position_m = torch.tensor(
        [[-5.485, -11.885, 0.5], [5.485, 11.885, 2.0]], dtype=torch.float32
    )
    velocity_mps = torch.tensor(
        [[8.0, 20.0, 3.0], [-4.0, 15.0, -2.0]], dtype=torch.float32
    )
    physical_v1 = resolve_court_keypoint_contract("physical_v1")
    camera_center = (0.0, -12.0, 5.0)
    camera = CameraData(
        camera_params={
            "R": np.eye(3).tolist(),
            "C": list(camera_center),
            "f": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "w": 100,
            "h": 80,
        },
        ball_uv=np.zeros((2, 2), dtype=np.float32),
        ball_vis=np.ones(2, dtype=np.bool_),
        ball_visibility_ratio=1.0,
        court_kp_uv=np.zeros((20, 2), dtype=np.float32),
        court_kp_vis=np.ones(20, dtype=np.bool_),
        court_visibility_count=20.0,
        court_view=build_court_view_record(
            camera_id="cam_0",
            camera_center_court_m=camera_center,
            contract=physical_v1,
        ),
    )
    return BLCSSceneData(
        scene_id="scene_000000",
        initial_from_cell=0,
        initial_from_side="near",
        rally_length=1,
        end_reason="test",
        winner_side=None,
        shots=[],
        ball_pos_world=position_m,
        ball_pos_norm=normalize_court_position(position_m),
        ball_vel_world=velocity_mps,
        ball_vel_norm=normalize_court_velocity(velocity_mps),
        cameras=[camera],
        num_cameras_sampled=1,
        fps_out=30,
        sim_fps=120,
        physics_config_dict={},
        court_config_dict={},
        num_balls=1,
    )


def test_writer_persists_contract_and_normalized_velocity(tmp_path: Path) -> None:
    scene = _scene()
    path = BLCSDatasetWriter(tmp_path).save_scene(scene)
    loaded = load_scene(path)

    assert loaded["court_keypoint_contract"] == resolve_court_keypoint_contract(
        "physical_v1"
    )
    assert loaded["meta"]["court_coordinate_normalization"]["scale_xyz_m"] == [
        11.885,
        11.885,
        11.885,
    ]
    assert loaded["cameras"][0]["params"] == scene.cameras[0].camera_params
    assert loaded["cameras"][0]["court_kp_uv"].shape == (20, 2)
    assert loaded["cameras"][0]["court_kp_vis"].shape == (20,)
    np.testing.assert_array_equal(
        loaded["cameras"][0]["court_kp_uv"],
        scene.cameras[0].court_kp_uv,
    )
    np.testing.assert_array_equal(
        loaded["cameras"][0]["court_kp_vis"],
        scene.cameras[0].court_kp_vis,
    )
    np.testing.assert_allclose(
        denormalize_court_position(loaded["ball_pos_norm"]),
        loaded["ball_pos_world"],
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        denormalize_court_velocity(loaded["ball_vel_norm"]),
        loaded["ball_vel_world"],
        atol=1e-5,
        rtol=0.0,
    )


def test_direct_loader_rejects_mismatched_scene_contract(tmp_path: Path) -> None:
    path = BLCSDatasetWriter(tmp_path).save_scene(_scene())
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["court_coordinate_normalization"]["scale_xyz_m"] = [5.485, 11.885, 1.07]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(CourtCoordinateContractError, match="mismatched"):
        load_scene(path)

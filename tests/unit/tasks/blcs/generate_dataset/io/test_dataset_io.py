"""BLCS normalized scene serialization contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter, load_scene
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData
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
        cameras=[],
        num_cameras_sampled=0,
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

    assert loaded["meta"]["court_coordinate_normalization"]["scale_xyz_m"] == [
        11.885,
        11.885,
        11.885,
    ]
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

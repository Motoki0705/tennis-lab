"""Tests for strict full-source BLCS trajectory contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.dataset.blcs.contracts import BLCSTrajectory


class _Scene:
    scene_id = "source-a"
    ball_pos_world = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.1], [0.2, 0.0, 1.2]],
        dtype=torch.float32,
    )
    ball_vel_world = torch.ones((3, 3), dtype=torch.float32)
    ball_present = None
    num_balls = 1
    fps_out = 30
    track_instances: list[dict[str, object]] = []


def test_scene_adapter_preserves_every_source_frame_in_order() -> None:
    trajectory = BLCSTrajectory.from_scene(_Scene(), split="train")

    assert trajectory.frame_count == 3
    assert trajectory.positions_court_m.shape == (3, 1, 3)
    assert trajectory.tracks[0].source_frame_indices == (0, 1, 2)
    assert trajectory.present[:, 0].tolist() == [True, True, True]


def test_trajectory_rejects_presence_mapping_and_nonfinite_data() -> None:
    trajectory = BLCSTrajectory.from_scene(_Scene(), split="train")

    with pytest.raises(ValueError, match="Presence disagrees"):
        replace(
            trajectory,
            present=np.asarray([[True], [False], [True]], dtype=np.bool_),
        )
    invalid = trajectory.positions_court_m.copy()
    invalid[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(trajectory, positions_court_m=invalid)


def test_multi_object_adapter_requires_explicit_lossless_placements() -> None:
    scene = _Scene()
    scene.ball_pos_world = torch.zeros((4, 2, 3), dtype=torch.float32)
    scene.ball_vel_world = torch.zeros((4, 2, 3), dtype=torch.float32)
    scene.ball_present = torch.tensor(
        [[True, False], [True, True], [False, True], [False, True]],
        dtype=torch.bool,
    )
    scene.num_balls = 2
    scene.track_instances = [
        {
            "track_id": 0,
            "source_scene_id": "rally-a",
            "source_start": 7,
            "source_end": 9,
            "birth_frame": 0,
            "death_frame": 2,
        },
        {
            "track_id": 1,
            "source_scene_id": "rally-b",
            "source_start": 3,
            "source_end": 6,
            "birth_frame": 1,
            "death_frame": 4,
        },
    ]

    trajectory = BLCSTrajectory.from_scene(scene, split="validation")

    assert trajectory.tracks[0].source_frame_indices == (7, 8, None, None)
    assert trajectory.tracks[1].source_frame_indices == (None, 3, 4, 5)
    assert trajectory.frame_count == 4

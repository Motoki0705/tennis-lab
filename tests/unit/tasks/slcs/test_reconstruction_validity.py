"""Visible 2D observations must never resurrect a rejected v2 3D teacher."""

from typing import Any

import numpy as np
import pytest

from src.tasks.slcs.data.quality import QualityConfig, build_label_masks


def test_v2_geometry_and_heading_are_hard_teacher_gates() -> None:
    values: dict[str, Any] = dict(
        human_kp_vis=np.ones((2, 3, 4, 17), np.float32), ball_vis=np.ones((3, 4), bool),
        player_position=np.zeros((2, 4, 3), np.float32), player_yaw=np.zeros((2, 4), np.float32),
        ball_3d=np.zeros((4, 3), np.float32), config=QualityConfig(.3, 2, 1., .1),
        teacher_quality={"schema_version": 1, "is_ground_truth": False, "player_weight": np.ones((2, 4)), "ball_weight": np.ones(4)},
        scene_schema_version=2,
    )
    with pytest.raises(ValueError, match="v2 requires"):
        build_label_masks(**values)
    player = np.array([[True, False, True, False]] * 2)
    heading = np.array([[True, True, False, True]] * 2)
    out = build_label_masks(**values, player_reconstruction_valid=player, player_heading_valid=heading, ball_reconstruction_valid=np.zeros(4, bool))
    assert out["player_label_valid"].tolist() == [[True, False, False, False]] * 2
    assert out["player_label_weight"].tolist() == [[1., 0, 0, 0]] * 2
    assert not out["ball_label_valid"].any()
    assert not out["ball_label_weight"].any()

"""Small scene fixtures shared by schema, archive and visualization tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.schema import SceneResult


@pytest.fixture
def v2_scene() -> SceneResult:
    """Two views, one player, valid origins at frame 0 and missing frame 1."""
    frames = 3
    player_valid = np.array([[True, False, True]])
    joints_vis: NDArray[np.bool_] = np.zeros((1, frames, 17), bool)
    joints_vis[0, [0, 2]] = True
    position: NDArray[np.float32] = np.zeros((1, frames, 3), np.float32)
    position[0, 2] = [1.1, 2.0, 0.9]
    joints: NDArray[np.float32] = np.zeros((1, frames, 17, 3), np.float32)
    joints[0, 2] = 1.0
    vertices: NDArray[np.float32] = np.zeros((1, frames, 5, 3), np.float32)
    vertices[0, 0] = 1.0
    return SceneResult(
        num_frames=frames,
        fps=30.0,
        width=640,
        height=360,
        court_kp=np.zeros((2, frames, 14, 2), np.float32),
        court_vis=np.ones((2, frames, 14), np.float32),
        player_position=position,
        player_yaw=np.zeros((1, frames), np.float32),
        smpl_body_pose=np.zeros((1, frames, 63), np.float32),
        smpl_global_orient=np.zeros((1, frames, 3), np.float32),
        smpl_betas=np.zeros((1, 10), np.float32),
        smpl_vertices_local=vertices,
        ball_uv=np.zeros((2, frames, 2), np.float32),
        ball_vis=np.array([[True, True, False], [True, False, False]]),
        ball_3d=np.zeros((frames, 3), np.float32),
        human_kp_2d=np.zeros((1, 2, frames, 17, 2), np.float32),
        human_kp_vis=np.ones((1, 2, frames, 17), np.float32),
        player_track_ids=np.array([0], np.int32),
        player_kp_3d=joints,
        player_observed=np.ones((1, frames), bool),
        player_valid=player_valid,
        player_heading_valid=player_valid.copy(),
        player_kp_3d_vis=joints_vis,
        player_smpl_valid=np.array([[True, False, False]]),
        ball_3d_valid=np.array([True, False, False]),
        player_rejection_code=np.where(player_valid, 0, 4).astype(np.uint8),
        player_kp_3d_rejection_code=np.where(joints_vis, 0, 1).astype(np.uint8),
        ball_rejection_code=np.array([0, 1, 1], np.uint8),
        metadata={"scene_schema_version": 2, "court_reference": {"camera_ids": ["near", "far"]}},
    )

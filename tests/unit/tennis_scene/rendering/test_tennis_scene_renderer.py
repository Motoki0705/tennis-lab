"""Unit tests for tennis scene SMPL rendering transforms."""

from __future__ import annotations

import numpy as np

from src.tennis_scene.io import SceneResult
from src.tennis_scene.rendering.tennis_scene_renderer import TennisSceneRenderer


def _make_renderer_with_fake_regressor() -> TennisSceneRenderer:
    renderer = TennisSceneRenderer.__new__(TennisSceneRenderer)
    renderer._smpl_joint_regressor = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    renderer._scene_vertices_cache = {}
    renderer._scene_joints_cache = {}
    return renderer


def test_build_players_smpl_vertices_court_maps_smpl_y_up_to_court_z_up() -> None:
    renderer = _make_renderer_with_fake_regressor()
    scene = SceneResult(
        num_frames=1,
        fps=30.0,
        width=1920,
        height=1080,
        court_kp=np.zeros((1, 1, 20, 2), dtype=np.float32),
        court_vis=np.ones((1, 1, 20), dtype=np.float32),
        player_position=np.array([[[10.0, 20.0, 0.5]]], dtype=np.float32),
        player_yaw=np.zeros((1, 1), dtype=np.float32),
        smpl_body_pose=np.zeros((1, 1, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((1, 1, 3), dtype=np.float32),
        smpl_betas=np.zeros((1, 10), dtype=np.float32),
        smpl_vertices_local=np.array(
            [[[[0.0, 0.0, 0.0], [0.0, 1.7, 0.0], [0.2, 1.0, 0.4]]]],
            dtype=np.float32,
        ),
        player_track_ids=np.array([0], dtype=np.int32),
    )

    vertices = renderer._build_players_smpl_vertices_court(scene)

    np.testing.assert_allclose(vertices[0, 0, 0], [10.0, 20.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(vertices[0, 0, 1], [10.0, 20.0, 2.2], atol=1e-6)
    np.testing.assert_allclose(vertices[0, 0, 2], [10.2, 19.6, 1.5], atol=1e-6)

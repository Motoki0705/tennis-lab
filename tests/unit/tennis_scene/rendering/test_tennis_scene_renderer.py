"""Unit tests for tennis scene SMPL rendering transforms and frame rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D

from src.tennis_scene.io import SceneResult
from src.tennis_scene.rendering.tennis_scene_renderer import (
    TennisSceneRenderer,
    TennisSceneStyle,
)
from src.utils.rendering.camera_view import CameraController


def _make_renderer_with_fake_regressor() -> TennisSceneRenderer:
    renderer = TennisSceneRenderer.__new__(TennisSceneRenderer)
    renderer._smpl_joint_regressor = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    renderer._scene_vertices_cache = {}
    renderer._scene_joints_cache = {}
    return renderer


def _make_renderer(
    style: TennisSceneStyle,
    assets: tuple[Path, Path],
) -> TennisSceneRenderer:
    faces_path, regressor_path = assets
    return TennisSceneRenderer(
        style,
        smpl_faces_path=faces_path,
        smpl_joint_regressor_path=regressor_path,
        camera=CameraController("broadcast"),
    )


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


def test_render_frame_dark_theme_smoke(
    tiny_scene: SceneResult,
    smpl_renderer_assets: tuple[Path, Path],
) -> None:
    """Full-feature frame render: bounce ring frame, HUD, minimap, dark theme."""
    renderer = _make_renderer(TennisSceneStyle(theme="dark"), smpl_renderer_assets)

    fig, ax = renderer.render_frame_3d(tiny_scene, 3)

    try:
        assert fig is not None
        # Dark theme: broadcast look with axes chrome removed.
        assert not ax._axis3don
        # Minimap inset was added next to the 3D axes.
        assert len(fig.axes) == 2
        # HUD text block is present with the ball speed line.
        hud_texts = [t.get_text() for t in ax.texts if "km/h" in t.get_text()]
        assert len(hud_texts) == 1
        # Speed and bounce caches were populated for the scene.
        assert renderer._get_ball_speeds(tiny_scene) is not None
        bounces = renderer._get_bounce_frames(tiny_scene)
        assert bounces is not None and bounces.tolist() == [2]
    finally:
        plt.close(fig)


def test_render_frame_light_theme_keeps_axes(
    tiny_scene: SceneResult,
    smpl_renderer_assets: tuple[Path, Path],
) -> None:
    renderer = _make_renderer(
        TennisSceneStyle(theme="light", show_minimap=False),
        smpl_renderer_assets,
    )

    fig, ax = renderer.render_frame_3d(tiny_scene, 0)

    try:
        assert fig is not None
        assert ax._axis3don
        assert len(fig.axes) == 1
        assert ax.get_title().startswith("Frame: 0/")
    finally:
        plt.close(fig)


def test_render_frame_does_not_draw_player_direction_arrows(
    tiny_scene: SceneResult,
    monkeypatch: pytest.MonkeyPatch,
    smpl_renderer_assets: tuple[Path, Path],
) -> None:
    renderer = _make_renderer(
        TennisSceneStyle(theme="light", show_minimap=False),
        smpl_renderer_assets,
    )
    quiver = Mock()
    monkeypatch.setattr(Axes3D, "quiver", quiver)
    fig, _ = renderer.render_frame_3d(tiny_scene, 0)

    try:
        quiver.assert_not_called()
    finally:
        plt.close(fig)


def test_render_into_external_axes_adds_no_minimap(
    tiny_scene: SceneResult,
    smpl_renderer_assets: tuple[Path, Path],
) -> None:
    renderer = _make_renderer(
        TennisSceneStyle(theme="dark"), smpl_renderer_assets
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    try:
        returned_fig, _ = renderer.render_frame_3d(tiny_scene, 0, ax=ax)
        assert returned_fig is None
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)

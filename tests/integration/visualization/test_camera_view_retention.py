"""Renderer integration tests for explicit 3D view retention across frames."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tasks.blcs.visualization.rendering.scene_renderer import BLCSSceneRenderer
from src.tasks.plcs.visualization.rendering.scene_renderer import PLCSSceneRenderer
from src.tennis_scene.io import SceneResult
from src.tennis_scene.rendering.tennis_scene_renderer import (
    TennisSceneRenderer,
    TennisSceneStyle,
)
from src.utils.rendering.ball_renderer import BallRenderer
from src.utils.rendering.camera_view import CameraView3DConfig
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.skeleton_renderer import SkeletonRenderer

pytestmark = pytest.mark.integration


def _view() -> CameraView3DConfig:
    return CameraView3DConfig(
        mode="look_at",
        center=(0.0, -20.0, 5.0),
        look_at=(0.0, 0.0, 0.0),
        roll_deg=7.0,
        hfov_deg=40.0,
    )


def _assert_view(ax: Any) -> None:
    assert ax.elev == pytest.approx(14.036243)
    assert ax.azim == pytest.approx(-90.0)
    assert ax.roll == pytest.approx(7.0)


def _finish_animation(animation: Any) -> None:
    animation._draw_was_started = True
    plt.close(animation._fig)


def test_blcs_animation_reapplies_view_each_frame() -> None:
    scene = {
        "ball_pos_world": np.array(
            [[0.0, 0.0, 1.0], [0.5, 1.0, 1.2]], dtype=np.float32
        ),
        "meta": {},
        "num_cameras": 0,
        "cameras": [],
    }
    animation: Any = BLCSSceneRenderer(view_3d=_view()).create_animation(
        scene, view="3d"
    )
    assert animation is not None
    ax = animation._fig.axes[0]
    try:
        ax.view_init(elev=1.0, azim=2.0, roll=3.0)
        animation._func(1)
        _assert_view(ax)
    finally:
        _finish_animation(animation)


def test_blcs_comparison_animation_reapplies_view_each_frame() -> None:
    positions = np.array(
        [[0.0, 0.0, 1.0], [0.5, 1.0, 1.2]], dtype=np.float32
    )
    animation: Any = BLCSSceneRenderer(view_3d=_view()).create_comparison_animation(
        positions,
        positions,
        view="3d",
    )
    assert animation is not None
    ax = animation._fig.axes[0]
    try:
        ax.view_init(elev=1.0, azim=2.0, roll=3.0)
        animation._func(1)
        _assert_view(ax)
    finally:
        _finish_animation(animation)


def test_plcs_animation_reapplies_view_after_axes_clear() -> None:
    scene = SimpleNamespace(
        meta={"num_frames": 2},
        position=np.zeros((2, 3), dtype=np.float32),
        rotation=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        canonical_pose_3d=np.zeros((2, 17, 3), dtype=np.float32),
        cameras=[],
    )
    animation: Any = PLCSSceneRenderer(view_3d=_view()).create_animation(
        scene, view="3d"
    )
    ax = animation._fig.axes[0]
    try:
        ax.view_init(elev=1.0, azim=2.0, roll=3.0)
        animation._func(1)
        _assert_view(ax)
    finally:
        _finish_animation(animation)


def test_plcs_comparison_animation_reapplies_view_after_axes_clear() -> None:
    scene = SimpleNamespace(
        meta={"num_frames": 2},
        position=np.zeros((2, 3), dtype=np.float32),
        rotation=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        canonical_pose_3d=np.zeros((2, 17, 3), dtype=np.float32),
        cameras=[],
    )
    animation: Any = PLCSSceneRenderer(
        view_3d=_view()
    ).create_comparison_animation(scene, scene, view="3d")
    ax = animation._fig.axes[0]
    try:
        ax.view_init(elev=1.0, azim=2.0, roll=3.0)
        animation._func(1)
        _assert_view(ax)
    finally:
        _finish_animation(animation)


def _minimal_tennis_scene() -> SceneResult:
    return SceneResult(
        num_frames=2,
        fps=30.0,
        width=1280,
        height=720,
        court_kp=np.zeros((1, 2, 20, 2), dtype=np.float32),
        court_vis=np.ones((1, 2, 20), dtype=np.float32),
        player_position=np.zeros((1, 2, 3), dtype=np.float32),
        player_yaw=np.zeros((1, 2), dtype=np.float32),
        smpl_body_pose=np.zeros((1, 2, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((1, 2, 3), dtype=np.float32),
        smpl_betas=np.zeros((1, 10), dtype=np.float32),
        smpl_vertices_local=np.zeros((1, 2, 1, 3), dtype=np.float32),
        player_track_ids=np.array([0], dtype=np.int32),
    )


def _lightweight_tennis_renderer() -> TennisSceneRenderer:
    renderer = TennisSceneRenderer.__new__(TennisSceneRenderer)
    renderer.style = TennisSceneStyle(show_direction=False, show_trail=False)
    renderer.view_3d = _view()
    renderer.court_renderer = CourtRenderer()
    renderer.ball_renderer = BallRenderer()
    renderer.skeleton_renderer = SkeletonRenderer(skeleton_type="smpl")
    renderer._mesh_renderer = None
    renderer_any = cast(Any, renderer)
    renderer_any._build_players_smpl_vertices_court = lambda scene: np.zeros(
        (1, 2, 1, 3), dtype=np.float32
    )
    renderer_any._get_players_kp_3d = lambda scene: np.zeros(
        (1, 2, 24, 3), dtype=np.float32
    )
    return renderer


def test_tennis_scene_animation_reapplies_view_after_axes_clear() -> None:
    animation: Any = _lightweight_tennis_renderer().create_animation(
        _minimal_tennis_scene()
    )
    ax = animation._fig.axes[0]
    try:
        ax.view_init(elev=1.0, azim=2.0, roll=3.0)
        animation._func(1)
        _assert_view(ax)
    finally:
        _finish_animation(animation)

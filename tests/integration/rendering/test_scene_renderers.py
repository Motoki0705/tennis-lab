"""Smoke tests wiring the shared rich-rendering primitives into the task
scene renderers (BLCS, PLCS, tennis_scene).

Each test drives a real ``FuncAnimation`` through a GIF save (so every frame
update runs: ``ax.clear()`` → theme/layers/camera reapplication) and then
verifies the post-animation axes state instead of pixel-exact output:
explicit layering, applied camera pose, HUD text, overlay axes count, and a
non-empty output file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import PillowWriter

from src.tasks.base.visualization.style import SceneStyleConfig
from src.tasks.blcs.visualization.rendering import BLCSSceneRenderer
from src.tasks.plcs.visualization.contracts import PoseRenderScene
from src.tasks.plcs.visualization.rendering import PLCSSceneRenderer
from src.utils.rendering.camera_view import CAMERA_PRESETS

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.integration]

_NUM_FRAMES = 6
_FPS = 5.0


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


def _save_and_get_axes(anim: Any, tmp_path: Path) -> tuple[Any, Any]:
    """Save the animation as a GIF (drives all frame updates) and return
    (figure, primary 3D axes)."""
    out = tmp_path / "anim.gif"
    anim.save(str(out), writer=PillowWriter(fps=int(_FPS)))
    assert out.stat().st_size > 0
    fig = anim._fig
    return fig, fig.axes[0]


def _assert_rich_3d_axes(ax: Any) -> None:
    """Common post-animation checks: layering, camera, court artists."""
    assert not ax.computed_zorder
    assert ax.elev == pytest.approx(CAMERA_PRESETS["broadcast"].elev)
    assert ax.azim == pytest.approx(CAMERA_PRESETS["broadcast"].azim)
    # Two-tone court + net surfaces land in collections, lines in ax.lines.
    assert len(ax.collections) > 0
    assert len(ax.lines) > 0


def _hud_texts(ax: Any) -> list[str]:
    return [t.get_text() for t in ax.texts if "Frame" in t.get_text()]


def _blcs_scene(num_frames: int = _NUM_FRAMES) -> dict[str, Any]:
    positions: np.ndarray = np.zeros((num_frames, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(-2.0, 2.0, num_frames)
    positions[:, 1] = np.linspace(-10.0, 10.0, num_frames)
    positions[:, 2] = np.abs(np.linspace(-1.5, 1.5, num_frames))
    meta = {
        "scene_id": "scene_test",
        "rally_length": 1,
        "end_reason": "test",
        "fps_out": _FPS,
        "num_frames": num_frames,
        "shots": [{"shot_index": 0, "t_start": 0, "t_bounce1": num_frames // 2}],
    }
    return {"ball_pos_world": positions, "meta": meta, "num_cameras": 0, "cameras": []}


def _multi_ball_blcs_scene(num_frames: int = _NUM_FRAMES) -> dict[str, Any]:
    scene = _blcs_scene(num_frames)
    first = scene["ball_pos_world"]
    second = first.copy()
    second[:, 0] += 1.5
    scene["ball_pos_world"] = np.stack([first, second], axis=1)
    scene["num_balls"] = 2
    scene["meta"]["shots"] = [
        {"ball_index": 0, "shots": [{"shot_index": 0, "t_bounce1": 2}]},
        {"ball_index": 1, "shots": [{"shot_index": 0, "t_bounce1": 3}]},
    ]
    return scene


def _plcs_scene(num_frames: int = _NUM_FRAMES) -> PoseRenderScene:
    position: np.ndarray = np.zeros((num_frames, 3), dtype=np.float32)
    position[:, 0] = np.linspace(-0.5, 0.5, num_frames)
    position[:, 1] = np.linspace(-0.8, -0.6, num_frames)
    rotation = np.tile(np.array([1.0, 0.0], dtype=np.float32), (num_frames, 1))
    rng = np.random.default_rng(0)
    canonical = rng.uniform(-0.3, 0.3, size=(num_frames, 17, 3)).astype(np.float32)
    canonical[:, :, 2] += 1.0
    return PoseRenderScene(
        position=position,
        rotation=rotation,
        canonical_pose_3d=canonical,
        meta={"num_frames": num_frames},
    )


class TestBLCSSceneRenderer:
    def test_dark_3d_animation_full_overlays(self, tmp_path: Path) -> None:
        renderer = BLCSSceneRenderer(style=SceneStyleConfig(theme="dark"))

        anim = renderer.create_animation(_blcs_scene(), view="3d", fps=_FPS)

        assert anim is not None
        fig, ax = _save_and_get_axes(anim, tmp_path)
        _assert_rich_3d_axes(ax)
        assert not ax._axis3don  # dark theme removes axes chrome
        assert len(fig.axes) == 2  # minimap inset
        (hud,) = _hud_texts(ax)
        assert "km/h" in hud and "Bounces" in hud

    def test_light_3d_animation_without_minimap(self, tmp_path: Path) -> None:
        renderer = BLCSSceneRenderer(
            style=SceneStyleConfig(theme="light", show_minimap=False, show_hud=False)
        )

        anim = renderer.create_animation(_blcs_scene(), view="3d", fps=_FPS)

        assert anim is not None
        fig, ax = _save_and_get_axes(anim, tmp_path)
        _assert_rich_3d_axes(ax)
        assert ax._axis3don
        assert len(fig.axes) == 1
        assert ax.get_title().startswith("Frame")
        assert _hud_texts(ax) == []

    def test_multi_ball_3d_animation(self, tmp_path: Path) -> None:
        renderer = BLCSSceneRenderer(style=SceneStyleConfig(theme="dark"))

        anim = renderer.create_animation(_multi_ball_blcs_scene(), view="3d", fps=_FPS)

        assert anim is not None
        fig, ax = _save_and_get_axes(anim, tmp_path)
        _assert_rich_3d_axes(ax)
        assert len(fig.axes) == 2
        assert ax.get_legend() is not None
        assert {text.get_text() for text in ax.get_legend().texts} == {
            "Ball 1",
            "Ball 2",
        }

    def test_dark_comparison_animation(self, tmp_path: Path) -> None:
        scene = _blcs_scene()
        gt = scene["ball_pos_world"]
        pred = gt + np.float32(0.3)
        renderer = BLCSSceneRenderer(style=SceneStyleConfig(theme="dark"))

        anim = renderer.create_comparison_animation(
            gt, pred, view="3d", fps=_FPS, events=None
        )

        assert anim is not None
        fig, ax = _save_and_get_axes(anim, tmp_path)
        _assert_rich_3d_axes(ax)
        assert ax.get_legend() is not None
        assert len(fig.axes) == 2

    def test_2d_animation_unchanged(self, tmp_path: Path) -> None:
        renderer = BLCSSceneRenderer(style=SceneStyleConfig(theme="dark"))

        anim = renderer.create_animation(_blcs_scene(), view="2d", fps=_FPS)

        assert anim is not None
        out = tmp_path / "anim2d.gif"
        anim.save(str(out), writer=PillowWriter(fps=int(_FPS)))
        assert out.stat().st_size > 0


class TestPLCSSceneRenderer:
    def test_dark_3d_animation_full_overlays(self, tmp_path: Path) -> None:
        renderer = PLCSSceneRenderer(style=SceneStyleConfig(theme="dark"))

        anim = renderer.create_animation(_plcs_scene(), view="3d", fps=_FPS)

        fig, ax = _save_and_get_axes(anim, tmp_path)
        _assert_rich_3d_axes(ax)
        assert not ax._axis3don
        assert len(fig.axes) == 2  # minimap inset
        (hud,) = _hud_texts(ax)
        # PLCS has no ball, so the HUD shows only the frame clock.
        assert "km/h" not in hud and "Bounces" not in hud

    def test_light_comparison_animation(self, tmp_path: Path) -> None:
        gt = _plcs_scene()
        pred = _plcs_scene()
        pred.position = pred.position + np.float32(0.05)
        renderer = PLCSSceneRenderer(
            style=SceneStyleConfig(theme="light", show_minimap=False)
        )

        anim = renderer.create_comparison_animation(gt, pred, view="3d", fps=_FPS)

        fig, ax = _save_and_get_axes(anim, tmp_path)
        _assert_rich_3d_axes(ax)
        assert ax._axis3don
        assert len(fig.axes) == 1
        assert ax.get_legend() is not None
        assert "GT vs Prediction" in ax.get_title()

    def test_2d_topdown_animation_unchanged(self, tmp_path: Path) -> None:
        renderer = PLCSSceneRenderer(style=SceneStyleConfig(theme="dark"))

        anim = renderer.create_animation(_plcs_scene(), view="2d_topdown", fps=_FPS)

        out = tmp_path / "topdown.gif"
        anim.save(str(out), writer=PillowWriter(fps=int(_FPS)))
        assert out.stat().st_size > 0


class TestTennisSceneRenderer:
    def test_dark_animation_smoke(self, tmp_path: Path) -> None:
        from src.tennis_scene.io import SceneResult
        from src.tennis_scene.rendering import TennisSceneRenderer, TennisSceneStyle

        num_frames = 4
        ball_3d: np.ndarray = np.zeros((num_frames, 3), dtype=np.float32)
        ball_3d[:, 0] = np.linspace(-1.0, 1.0, num_frames)
        ball_3d[:, 2] = np.array([0.5, 0.05, 0.3, 0.6], dtype=np.float32)
        scene = SceneResult(
            num_frames=num_frames,
            fps=_FPS,
            width=1280,
            height=720,
            court_kp=np.zeros((1, num_frames, 20, 2), dtype=np.float32),
            court_vis=np.ones((1, num_frames, 20), dtype=np.float32),
            player_position=np.tile(
                np.array([2.0, -8.0, 0.0], dtype=np.float32), (1, num_frames, 1)
            ),
            player_yaw=np.zeros((1, num_frames), dtype=np.float32),
            smpl_body_pose=np.zeros((1, num_frames, 63), dtype=np.float32),
            smpl_global_orient=np.zeros((1, num_frames, 3), dtype=np.float32),
            smpl_betas=np.zeros((1, 10), dtype=np.float32),
            smpl_vertices_local=np.zeros((1, num_frames, 6890, 3), dtype=np.float32),
            ball_3d=ball_3d,
            player_track_ids=np.array([0], dtype=np.int32),
        )
        renderer = TennisSceneRenderer(TennisSceneStyle(theme="dark"))

        anim = renderer.create_animation(scene, fps=_FPS)

        fig, ax = _save_and_get_axes(anim, tmp_path)
        _assert_rich_3d_axes(ax)
        assert not ax._axis3don
        assert len(fig.axes) == 2

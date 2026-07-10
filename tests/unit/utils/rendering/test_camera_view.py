"""Unit tests for the shared virtual camera controller and view application."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.utils.rendering.camera_view import (
    CAMERA_PRESETS,
    CameraController,
    CameraKeyframe,
    CameraView3D,
    apply_scene_camera,
    resolve_camera_view,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


class TestApplySceneCamera:
    def test_applies_view_and_fixed_court_bounds(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        try:
            apply_scene_camera(ax, CameraView3D(elev=25.0, azim=-70.0, zoom=1.5))

            assert ax.elev == pytest.approx(25.0)
            assert ax.azim == pytest.approx(-70.0)
            x_half_span = float(HALF_DOUBLES_WIDTH + 2.0)
            y_half_span = float(HALF_LENGTH + 2.0)
            assert ax.get_xlim() == pytest.approx((-x_half_span, x_half_span))
            assert ax.get_ylim() == pytest.approx((-y_half_span, y_half_span))
            assert ax.get_zlim() == pytest.approx((0.0, 4.0))
        finally:
            plt.close(fig)

    def test_restores_state_after_clear(self) -> None:
        """The per-frame reapplication contract: ax.clear() loses the view."""
        view = CameraView3D(elev=18.0, azim=-90.0, zoom=1.6)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        try:
            apply_scene_camera(ax, view)
            first = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim(), ax.elev, ax.azim)

            ax.clear()
            ax.set_xlim(100.0, 200.0)
            apply_scene_camera(ax, view)

            assert (ax.get_xlim(), ax.get_ylim(), ax.get_zlim(), ax.elev, ax.azim) == first
        finally:
            plt.close(fig)


class TestPresets:
    def test_expected_presets_exist(self) -> None:
        assert {"broadcast", "side", "top", "corner", "behind_far"} <= set(CAMERA_PRESETS)

    def test_resolve_passes_through_views(self) -> None:
        view = CameraView3D(elev=10.0, azim=20.0)
        assert resolve_camera_view(view) is view

    def test_resolve_unknown_preset_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown camera preset"):
            resolve_camera_view("dolly")

    def test_non_positive_zoom_raises(self) -> None:
        with pytest.raises(ValueError, match="zoom must be positive"):
            CameraView3D(elev=0.0, azim=0.0, zoom=0.0)


class TestStaticMode:
    def test_returns_base_for_any_frame(self) -> None:
        controller = CameraController("side")
        assert controller.view_at(0, 30.0) == CAMERA_PRESETS["side"]
        assert controller.view_at(1000, 30.0) == CAMERA_PRESETS["side"]

    def test_invalid_fps_raises(self) -> None:
        controller = CameraController("side")
        with pytest.raises(ValueError, match="fps must be positive"):
            controller.view_at(0, 0.0)

    def test_keyframes_with_static_mode_raise(self) -> None:
        keyframes = [
            CameraKeyframe(0, CameraView3D(0.0, 0.0)),
            CameraKeyframe(10, CameraView3D(1.0, 1.0)),
        ]
        with pytest.raises(ValueError, match="mode is 'static'"):
            CameraController("side", keyframes=keyframes)


class TestOrbitMode:
    def test_azimuth_revolves_once_per_period(self) -> None:
        base = CameraView3D(elev=20.0, azim=-90.0, zoom=1.2)
        controller = CameraController(base, mode="orbit", orbit_period_s=10.0)

        half_turn = controller.view_at(150, 30.0)  # 5 s = half a period

        assert half_turn.azim == pytest.approx(-90.0 + 180.0)
        assert half_turn.elev == pytest.approx(20.0)
        assert half_turn.zoom == pytest.approx(1.2)

    def test_non_positive_period_raises(self) -> None:
        with pytest.raises(ValueError, match="orbit_period_s"):
            CameraController("broadcast", mode="orbit", orbit_period_s=0.0)


class TestKeyframesMode:
    def _controller(self) -> CameraController:
        return CameraController(
            "broadcast",
            mode="keyframes",
            keyframes=[
                CameraKeyframe(10, CameraView3D(elev=0.0, azim=0.0, zoom=1.0)),
                CameraKeyframe(20, CameraView3D(elev=40.0, azim=-80.0, zoom=2.0)),
            ],
        )

    def test_exact_views_at_keyframes(self) -> None:
        controller = self._controller()
        assert controller.view_at(10, 30.0) == CameraView3D(0.0, 0.0, 1.0)
        assert controller.view_at(20, 30.0) == CameraView3D(40.0, -80.0, 2.0)

    def test_midpoint_uses_smoothstep(self) -> None:
        controller = self._controller()
        mid = controller.view_at(15, 30.0)
        # smoothstep(0.5) == 0.5, so the midpoint is the exact average.
        assert mid.elev == pytest.approx(20.0)
        assert mid.azim == pytest.approx(-40.0)
        assert mid.zoom == pytest.approx(1.5)

    def test_clamped_outside_range(self) -> None:
        controller = self._controller()
        assert controller.view_at(0, 30.0) == CameraView3D(0.0, 0.0, 1.0)
        assert controller.view_at(99, 30.0) == CameraView3D(40.0, -80.0, 2.0)

    def test_fewer_than_two_keyframes_raise(self) -> None:
        with pytest.raises(ValueError, match="at least 2 keyframes"):
            CameraController(
                "broadcast",
                mode="keyframes",
                keyframes=[CameraKeyframe(0, CameraView3D(0.0, 0.0))],
            )

    def test_non_increasing_frames_raise(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            CameraController(
                "broadcast",
                mode="keyframes",
                keyframes=[
                    CameraKeyframe(10, CameraView3D(0.0, 0.0)),
                    CameraKeyframe(10, CameraView3D(1.0, 1.0)),
                ],
            )


class TestFromConfig:
    def test_preset_with_zoom_override(self) -> None:
        controller = CameraController.from_config({"preset": "broadcast", "zoom": 2.5})
        assert controller.base.elev == CAMERA_PRESETS["broadcast"].elev
        assert controller.base.zoom == pytest.approx(2.5)

    def test_explicit_angles(self) -> None:
        controller = CameraController.from_config({"elev": 45.0, "azim": 30.0})
        assert controller.base == CameraView3D(45.0, 30.0)

    def test_null_zoom_keeps_preset_zoom(self) -> None:
        controller = CameraController.from_config({"preset": "broadcast", "zoom": None})
        assert controller.base.zoom == CAMERA_PRESETS["broadcast"].zoom

    def test_preset_and_angles_together_raise(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            CameraController.from_config({"preset": "broadcast", "elev": 10.0, "azim": 0.0})

    def test_missing_view_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 'preset' or both"):
            CameraController.from_config({"elev": 10.0})

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown camera mode"):
            CameraController.from_config({"preset": "broadcast", "mode": "dolly"})

    def test_keyframes_parsed(self) -> None:
        controller = CameraController.from_config(
            {
                "preset": "broadcast",
                "mode": "keyframes",
                "keyframes": [
                    {"frame": 0, "preset": "broadcast"},
                    {"frame": 60, "elev": 40.0, "azim": -45.0, "zoom": 1.4},
                ],
            }
        )
        assert controller.keyframes[0].view == CAMERA_PRESETS["broadcast"]
        assert controller.keyframes[1].view == CameraView3D(40.0, -45.0, 1.4)

    def test_keyframe_missing_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required key 'frame'"):
            CameraController.from_config(
                {
                    "preset": "broadcast",
                    "mode": "keyframes",
                    "keyframes": [{"preset": "broadcast"}],
                }
            )

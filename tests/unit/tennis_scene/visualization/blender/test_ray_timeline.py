"""The 2D/pullback/3D source clock and calibrated camera contract."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.visualization.blender.export.camera import (
    PinholeCamera,
    to_blender_camera,
)
from src.tennis_scene.visualization.blender.shots.ray_timeline import (
    RayShot,
    calibrated_pose,
    plan_timeline,
    select_camera,
)


def manifest(
    *, fps: float = 60.0, frames: tuple[int, int] = (782, 872)
) -> dict[str, Any]:
    return {
        "frames": frames,
        "fps": fps,
        "cameras": [{"id": "cam0"}, {"id": "cam1"}, {"id": "cam2"}],
    }


def test_continuous_advances_without_rewind_or_t0_hold() -> None:
    shot = RayShot(pullback_seconds=3.0)
    timeline = plan_timeline(manifest(), shot)
    stages = timeline.stages
    assert [(s.source_start, s.source_stop) for s in stages] == [
        (782, 827),
        (827, 850),
        (850, 872),
    ]
    assert [s.name for s in stages] == ["two_d", "pullback", "playback"]
    assert [(s.start, s.stop) for s in stages] == [(0, 23), (23, 113), (113, 124)]
    source = timeline.source_frames
    assert source[0] == 782 and source[-1] == 870
    assert np.all(np.diff(source) >= 0)
    assert len(np.unique(source[23:113])) == 23
    assert source[23] == 827 and source[113] == 850
    assert stages[0].source_rate_fps == stages[2].source_rate_fps == 60.0
    assert stages[1].to_dict(30, 60)["source_speed"] == pytest.approx(23 / 180)


def test_replay_rewinds_only_after_pullback_and_holds_last_frame_explicitly() -> None:
    timeline = plan_timeline(manifest(fps=59.94006), RayShot(time_mode="replay"))
    first, pull, replay = timeline.stages
    a = timeline.source_frames[first.start : first.stop]
    b = timeline.source_frames[replay.start : replay.stop]
    np.testing.assert_array_equal(a, b)
    assert first.stop == 46 and replay.start == 136 and replay.stop == 182
    assert np.all(timeline.source_frames[pull.start : pull.stop] == 871)
    assert pull.source_rate_fps == 0
    assert np.flatnonzero(np.diff(timeline.source_frames) < 0).tolist() == [135]


def test_native_and_retimed_seconds_are_explicit_in_every_stage() -> None:
    shot = RayShot(two_d_seconds=2.0, pullback_seconds=3.0, playback_seconds=4.0)
    timeline = plan_timeline(manifest(), shot)
    assert len(timeline.source_frames) == 270
    assert [s.to_dict(30, 60)["seconds"] for s in timeline.stages] == [2.0, 3.0, 4.0]
    for stage in timeline.stages:
        source = timeline.source_frames[stage.start : stage.stop]
        assert source.min() >= stage.source_start and source.max() < stage.source_stop
        assert source[0] == stage.source_start


def test_fractional_fps_clock_matches_nearest_native_timestamp() -> None:
    data = manifest(fps=59.94006)
    timeline = plan_timeline(data, RayShot(time_mode="replay"))
    expected = np.minimum(
        782 + np.floor(np.arange(46) * 59.94006 / 30 + 0.5).astype(int), 871
    )
    np.testing.assert_array_equal(timeline.source_frames[:46], expected)
    assert all(s.stop > s.start for s in timeline.stages)


def test_clock_and_camera_selection_ignore_t0() -> None:
    data = manifest()
    first = plan_timeline({**data, "t0": 826}, RayShot())
    second = plan_timeline({**data, "t0": 870}, RayShot())
    np.testing.assert_array_equal(first.source_frames, second.source_frames)


@pytest.mark.parametrize(
    "settings",
    [
        {"time_mode": "unknown"},
        {"pullback_seconds": 0},
        {"two_d_seconds": float("nan")},
        {"playback_seconds": -1},
        {"samples": 0},
        {"continuous_two_d_fraction": 0.8, "continuous_pullback_fraction": 0.3},
        {"continuous_pullback_fraction": float("inf")},
        {"camera_id": ""},
        {"frustum_depth_m": 0.001},
    ],
)
def test_bad_settings_fail(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RayShot(**settings)


def test_clock_rejects_short_intervals_invalid_fps_and_one_frame_move() -> None:
    with pytest.raises(ValueError, match="too short"):
        plan_timeline(manifest(frames=(0, 2)), RayShot())
    with pytest.raises(ValueError, match="fps"):
        plan_timeline(manifest(fps=0), RayShot())
    with pytest.raises(ValueError, match="two output frames"):
        plan_timeline(manifest(), RayShot(pullback_seconds=0.01))


def test_camera_uses_visible_count_and_stable_id_tie_break_and_records_reason() -> None:
    data = manifest(frames=(0, 5))
    data["cameras"] = [{"id": "cam2"}, {"id": "cam0"}, {"id": "cam1"}]
    uv = np.zeros((3, 5, 2))
    uv[1, 0] = np.nan
    view, reason = select_camera(data, uv, None)
    assert view == 2 and reason["camera_id"] == "cam1"
    assert reason["visible_frames"] == 5 and reason["interval_frames"] == 5
    assert reason["visibility_counts"] == {"cam2": 5, "cam0": 4, "cam1": 5}
    assert reason["policy"] == "most_visible_frames"
    view, reason = select_camera(data, uv, "cam0")
    assert view == 1 and reason["policy"] == "explicit_camera_id"
    assert reason["visible_frames"] == 4


def test_camera_never_switches_silently() -> None:
    data = manifest(frames=(0, 5))
    uv = np.full((3, 5, 2), np.nan)
    with pytest.raises(ValueError, match="no visible"):
        select_camera(data, uv, None)
    with pytest.raises(ValueError, match="Unknown"):
        select_camera(data, uv, "missing")
    with pytest.raises(ValueError, match="shape"):
        select_camera(data, uv[:, :2], "cam0")
    uv[1] = 0
    with pytest.raises(ValueError, match="no visible"):
        select_camera(data, uv, "cam0")


def calibrated_fixture() -> dict[str, Any]:
    # Nontrivial orientation, with camera-local +Z facing the court.
    angle = 0.3
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [0, 0, -1],
            [np.sin(angle), np.cos(angle), 0],
        ]
    )
    center = np.array([2.0, -18.0, 5.0])
    return {
        "id": "cam2",
        "K": [[900.0, 0, 960], [0, 900.0, 540], [0, 0, 1]],
        "R": R.tolist(),
        "t": (-R @ center).tolist(),
        "center": center.tolist(),
        "width": 1920,
        "height": 1080,
    }


def test_calibrated_pose_matches_shared_export_camera_and_original_pixel_rays() -> None:
    cam = calibrated_fixture()
    center, rotation, lens = calibrated_pose(cam, RayShot())
    pinhole = PinholeCamera(
        np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"]), 1920, 1080
    )
    expected = to_blender_camera(pinhole, sensor_width_mm=36.0)
    np.testing.assert_allclose(center, expected.location, atol=1e-12)
    np.testing.assert_allclose(rotation, expected.rotation_matrix, atol=1e-12)
    assert lens == expected.lens_mm
    for uv in [(0, 0), (1920, 1080), (432, 731)]:
        local = np.linalg.solve(np.array(cam["K"]), [*uv, 1]) * 1.495
        world = center + np.array(cam["R"]).T @ local
        blender = rotation.T @ (world - center)
        focal = lens / 36.0 * 1920
        actual_uv = [
            960 - focal * blender[0] / blender[2],
            540 + focal * blender[1] / blender[2],
        ]
        np.testing.assert_allclose(actual_uv, uv, atol=1e-10)
    calibrated_pose(cam, replace(RayShot(), width=960, height=540))


@pytest.mark.parametrize(
    "change", ["skew", "principal_point", "aspect", "center", "rotation"]
)
def test_unsupported_calibration_is_rejected_instead_of_cropped(change: str) -> None:
    cam, shot = calibrated_fixture(), RayShot()
    if change == "skew":
        cam["K"][0][1] = 10
    elif change == "principal_point":
        cam["K"][0][2] -= 10
    elif change == "aspect":
        shot = replace(shot, width=1080, height=1080)
    elif change == "center":
        cam["center"][0] += 1
    else:
        cam["R"][0][0] += 0.2
    with pytest.raises(ValueError):
        calibrated_pose(cam, shot)

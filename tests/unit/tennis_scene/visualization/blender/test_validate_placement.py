"""CPU tests for the placement validator's pure comparison functions.

The Blender-importing collection functions are exercised by the real Blender
runs; this module corrupts expected/actual values in CPython and proves that the
comparisons reject the four failure modes from the review: intermediate stage
keys replaced by final values, a replay cache frozen at ``t0``, an off-by-one
60 fps cache mapping, and a uniform +90 degree camera rotation.
"""

from __future__ import annotations

import copy
import math
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import src.tennis_scene.visualization.blender.shots.validate_placement as validator
from src.tennis_scene.visualization.blender.shots.player_placement import (
    OUTPUT_FPS,
    PlacementShot,
    PlacementTimeline,
    _rotation_z,
    body_forward,
    body_transform_at,
    replay_source_indices,
)

START = 12
STOP = 212
T0 = 52
SCALE = 1.07
POINTS = 32


def _timeline(fps: float = OUTPUT_FPS) -> PlacementTimeline:
    replay = replay_source_indices(START, STOP, fps)
    stages = (
        ("court", 0, 60),
        ("appear", 60, 120),
        ("scale", 120, 180),
        ("yaw", 180, 240),
        ("position", 240, 300),
        ("final", 300, 330),
        ("replay", 330, 330 + len(replay)),
    )
    return PlacementTimeline(
        shot=PlacementShot(),
        fps=float(fps),
        start=START,
        stop=STOP,
        t0=T0,
        intro_source_frames=(),
        stages=stages,
        replay_source_frames=replay,
        frame_start=0,
        frame_end=330 + len(replay) - 1,
    )


def _arrays(
    timeline: PlacementTimeline,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261022)
    frames = timeline.stop - timeline.start
    cached = rng.normal(0.0, 0.05, (frames, POINTS, 3)).astype(np.float32)
    hip = np.stack(
        [
            2.0 + 0.6 * np.sin(0.04 * np.arange(frames)),
            -1.2 + 0.5 * np.cos(0.03 * np.arange(frames)),
            0.95 + 0.03 * np.sin(0.07 * np.arange(frames)),
        ],
        axis=1,
    )
    yaw = 0.35 + 0.55 * np.sin(0.02 * np.arange(frames))
    valid = np.ones(frames, bool)
    valid[[3, 4, 20]] = False
    return cached, hip, yaw, valid


def _expected_transforms(
    timeline: PlacementTimeline,
    hip: np.ndarray,
    yaw: np.ndarray,
) -> dict[int, dict[str, Any]]:
    t0_index = timeline.t0 - timeline.start
    t0_hip = np.asarray(hip[t0_index], np.float64)
    t0_yaw = float(yaw[t0_index])
    anchor = t0_hip + np.asarray(timeline.shot.display_offset_m, np.float64)
    frames = [
        frame
        for stage_frames in validator.stage_sample_frames(timeline).values()
        for frame in stage_frames
    ]
    result: dict[int, dict[str, Any]] = {}
    for frame in frames:
        frame_scale, frame_yaw, frame_location = body_transform_at(
            timeline,
            frame,
            scale=SCALE,
            t0_yaw=t0_yaw,
            t0_hip=t0_hip,
            anchor=anchor,
            hip_position=hip,
            yaw_correction=yaw,
        )
        result[frame] = {
            "scale": np.full(3, float(frame_scale), np.float64),
            "yaw": float(frame_yaw),
            "location": np.asarray(frame_location, np.float64),
        }
    return result


def _copy_transforms(
    values: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        frame: {
            "scale": np.array(item["scale"], copy=True),
            "yaw": float(item["yaw"]),
            "location": np.array(item["location"], copy=True),
        }
        for frame, item in values.items()
    }


def _world_vertices(
    timeline: PlacementTimeline,
    cached: np.ndarray,
    hip: np.ndarray,
    yaw: np.ndarray,
    valid: np.ndarray,
    *,
    cache_offset: int = 0,
    frozen_t0: bool = False,
) -> dict[int, np.ndarray]:
    begin, end = timeline.stage_bounds()["replay"]
    t0_index = timeline.t0 - timeline.start
    indices = validator.replay_vertex_indices(cached.shape[1])
    result: dict[int, np.ndarray] = {}
    for frame in range(begin, end):
        index = int(timeline.pc2_frame_at(frame))
        if not bool(valid[index]):
            continue
        if frozen_t0:
            cache_index = t0_index
        else:
            cache_index = min(index + cache_offset, len(cached) - 1)
        result[frame] = np.asarray(
            np.asarray(hip[index], np.float64)
            + SCALE
            * (
                _rotation_z(float(yaw[index]))
                @ np.asarray(cached[cache_index][indices], np.float64).T
            ).T,
            np.float64,
        )
    return result


def test_stage_sample_frames_cover_every_stage() -> None:
    frames = validator.stage_sample_frames(_timeline())
    assert frames["scale"] == (120, 149, 179)
    assert frames["appear"] == (60, 89, 119)
    assert set(frames) == {"appear", "scale", "yaw", "position", "final"}


def test_uncorrupted_stage_transforms_pass() -> None:
    timeline = _timeline()
    cached, hip, yaw, _ = _arrays(timeline)
    expected = _expected_transforms(timeline, hip, yaw)
    info, failures = validator.compare_stage_transforms(
        timeline, expected, _copy_transforms(expected)
    )
    assert info["passed"] is True
    assert failures == []
    assert info["frames_checked"] == 15


def test_negative_intermediate_stage_values_replaced_by_final() -> None:
    timeline = _timeline()
    cached, hip, yaw, _ = _arrays(timeline)
    expected = _expected_transforms(timeline, hip, yaw)
    actual = _copy_transforms(expected)
    final_frame = timeline.stage_bounds()["final"][0]
    final_values = expected[final_frame]
    replaced: set[str] = set()
    for stage in ("scale", "yaw", "position"):
        middle = validator.stage_sample_frames(timeline)[stage][1]
        actual[middle] = _copy_transforms({middle: final_values})[middle]
        replaced.add(stage)
    info, failures = validator.compare_stage_transforms(timeline, expected, actual)
    assert info["passed"] is False
    flagged = {item["stage"] for item in failures}
    assert replaced <= flagged
    for item in failures:
        assert item["check"] == "stage_transform"
        assert item["expected"] and item["actual"]
        assert "frame" in item


def test_uncorrupted_replay_cache_passes() -> None:
    for fps in (30.0, 60.0):
        timeline = _timeline(fps)
        cached, hip, yaw, valid = _arrays(timeline)
        actual = _world_vertices(timeline, cached, hip, yaw, valid)
        info, failures = validator.compare_replay_vertices(
            timeline, cached, hip, yaw, SCALE, valid, actual
        )
        assert info["passed"] is True, failures
        begin, end = timeline.stage_bounds()["replay"]
        expected_frames = sum(
            bool(valid[int(timeline.pc2_frame_at(frame))])
            for frame in range(begin, end)
        )
        assert info["frames_checked"] == expected_frames
        assert info["vertices_checked"] == POINTS


def test_negative_replay_cache_frozen_at_t0() -> None:
    timeline = _timeline()
    cached, hip, yaw, valid = _arrays(timeline)
    actual = _world_vertices(timeline, cached, hip, yaw, valid, frozen_t0=True)
    info, failures = validator.compare_replay_vertices(
        timeline, cached, hip, yaw, SCALE, valid, actual
    )
    assert info["passed"] is False
    assert failures
    assert all(item["check"] == "replay_body" for item in failures)
    assert all("max_abs_error_m" in item for item in failures)


def test_negative_60fps_cache_frame_shifted_by_one() -> None:
    timeline = _timeline(60.0)
    cached, hip, yaw, valid = _arrays(timeline)
    indices = [int(timeline.pc2_frame_at(frame)) for frame in range(330, 430)]
    # 60 fps input is sampled every two source frames; an off-by-one cache
    # selection is visible on every replay frame.
    assert all(
        int(timeline.pc2_frame_at(frame + 1)) - int(timeline.pc2_frame_at(frame)) == 2
        for frame in range(330, 429)
    )
    assert max(indices) < len(cached) - 1
    actual = _world_vertices(timeline, cached, hip, yaw, valid, cache_offset=1)
    info, failures = validator.compare_replay_vertices(
        timeline, cached, hip, yaw, SCALE, valid, actual
    )
    assert info["passed"] is False
    assert failures
    assert all(item["check"] == "replay_body" for item in failures)


def test_uncorrupted_absolute_camera_azimuth_passes() -> None:
    timeline = _timeline()
    _, hip, yaw, _ = _arrays(timeline)
    t0_index = timeline.t0 - timeline.start
    forward = _fixture_forward(float(yaw[t0_index]))
    expected = validator.expected_camera_directions(timeline, timeline.shot, forward)
    actual = {
        frame: direction * timeline.shot.camera_distance_m
        for frame, direction in expected.items()
    }
    info, failures = validator.compare_camera_azimuths(
        timeline, expected, actual, timeline.shot.camera_distance_m
    )
    assert info["passed"] is True, failures
    assert info["frames"] == len(timeline.replay_source_frames)


def test_negative_camera_azimuth_rotated_by_90_degrees() -> None:
    timeline = _timeline()
    _, hip, yaw, _ = _arrays(timeline)
    t0_index = timeline.t0 - timeline.start
    expected = validator.expected_camera_directions(
        timeline, timeline.shot, _fixture_forward(float(yaw[t0_index]))
    )
    rotation = _rotation_z(math.pi / 2.0)
    actual = {
        frame: (rotation @ direction) * timeline.shot.camera_distance_m
        for frame, direction in expected.items()
    }
    info, failures = validator.compare_camera_azimuths(
        timeline, expected, actual, timeline.shot.camera_distance_m
    )
    assert info["passed"] is False
    assert failures
    assert all(item["check"] == "camera_orbit" for item in failures)
    assert info["max_azimuth_error_deg"] == pytest.approx(90.0, abs=1e-6)
    # The whole replay is shifted uniformly; there is no first-frame anchor
    # that could cancel it.
    assert {round(item["max_azimuth_error_deg"], 6) for item in failures} == {90.0}


def test_camera_angle_wraps_across_pi() -> None:
    timeline = _timeline()
    _, hip, yaw, _ = _arrays(timeline)
    t0_index = timeline.t0 - timeline.start
    expected = validator.expected_camera_directions(
        timeline, timeline.shot, _fixture_forward(float(yaw[t0_index]))
    )
    distance = timeline.shot.camera_distance_m
    rotation = _rotation_z(2.0 * math.pi)
    actual = {
        frame: (rotation @ direction) * distance
        for frame, direction in expected.items()
    }
    info, failures = validator.compare_camera_azimuths(
        timeline, expected, actual, distance
    )
    assert info["passed"] is True, failures


_FIXTURE_JOINTS = np.asarray(
    [
        [0.00, 0.05, 0.72],
        [-0.04, 0.08, 0.76],
        [0.04, 0.08, 0.76],
        [-0.08, 0.04, 0.74],
        [0.08, 0.04, 0.74],
        [-0.18, 0.00, 0.50],
        [0.18, 0.00, 0.50],
        [-0.28, 0.00, 0.22],
        [0.28, 0.00, 0.22],
        [-0.34, 0.00, -0.05],
        [0.34, 0.00, -0.05],
        [-0.10, 0.00, -0.05],
        [0.10, 0.00, -0.05],
        [-0.11, 0.00, -0.50],
        [0.11, 0.00, -0.50],
        [-0.12, 0.00, -0.90],
        [0.12, 0.00, -0.90],
    ],
    np.float64,
)


def _fixture_forward(yaw: float) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        body_forward(yaw, copy.deepcopy(_FIXTURE_JOINTS)),
    )

"""Validate a saved player-placement scene inside Blender.

blender --background <shot>/tennis_placement.blend --python-exit-code 1 \
    --python src/tennis_scene/visualization/blender/shots/validate_placement.py -- \
    --bundle <bundle> --report <shot>/validation.json

No rendering is performed.  The checks use Blender's evaluated mesh and object
matrices:

* on the staged frames of ``appear``/``scale``/``yaw``/``position``/``final``,
  the evaluated ``Body`` world transform equals ``body_transform_at`` (scale,
  Z rotation with a 2*pi wrap, and location; the first, middle and last frame
  of every stage are sampled);
* on the same staged frames, the evaluated *local* ``Body`` vertices equal the
  ``body_local.pc2`` sample at ``t0`` (tolerance 1e-4 m);
* on every valid replay frame, the evaluated ``Body`` world vertices equal
  ``hip_position[i] + scale * Rz(yaw_correction[i]) @ u[i]`` with
  ``i = timeline.pc2_frame_at(frame)`` (a fixed vertex subset is compared so
  the input-fps to 30 fps mapping is checked on every frame);
* at the final staged frames, the world-space ``Body`` vertices equal
  ``placed_vertices_t0`` (tolerance 1e-4 m);
* on every replay frame, each ``SourceJoints`` world position equals
  ``hip_position + scale * Rz(yaw) @ joints_local`` (tolerance 1e-4 m);
* on every replay frame, the absolute camera-hip azimuth equals the direction
  from ``_camera_direction(shot, t0_forward, progress)``; the comparison never
  cancels the first frame's error (tolerance 1e-3 degrees);
* ``Body`` and every ``SourceJoint`` hide in render *and* viewport exactly where
  ``valid`` is false (and only there);
* when ``intro_plate`` is enabled, the plate backdrop is visible only during the
  intro and every copied plate frame is the source frame selected from
  ``manifest["frames"]["start"]``.

The comparison functions at the top of this module do not import Blender, so
they run under CPython's pytest; only the collection functions at the bottom
import ``bpy``.  Every comparison rejects non-finite values instead of letting
NaN slip through.  The report is written on success and on failure; mismatches
exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from player_placement import (  # noqa: E402
    INTRO_PLATE_OBJECT,
    JOINT_COUNT,
    OUTPUT_FPS,
    TOLERANCE_M,
    PlacementShot,
    PlacementTimeline,
    _camera_direction,
    _rotation_z,
    body_forward,
    body_transform_at,
    body_visible_at,
    build_timeline,
    load_manifest,
)

# Existing tolerances from the shot: metres for positions and 1e-3 degrees for
# angles.  Scale is dimensionless and uses the same numeric bound as TOLERANCE_M.
ANGLE_TOLERANCE_DEG = 1e-3
ANGLE_TOLERANCE_RAD = math.radians(ANGLE_TOLERANCE_DEG)
STAGED_STAGES = ("appear", "scale", "yaw", "position", "final")
REPLAY_VERTEX_COUNT = 64


def _bpy() -> Any:
    """Import Blender only while running inside it.

    The comparison functions below are exercised by CPython tests, where ``bpy``
    does not exist.
    """
    import bpy

    return bpy


def _wrap_angle(radians: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (float(radians) + math.pi) % (2.0 * math.pi) - math.pi


def _angle_error(actual: float, expected: float) -> float:
    return abs(_wrap_angle(float(actual) - float(expected)))


def stage_sample_frames(timeline: PlacementTimeline) -> dict[str, tuple[int, ...]]:
    """Frames sampled for the staged transform/cache checks.

    Every staged interval is half-open, so its last displayed frame is
    ``end - 1``.  The first, middle and last frame are compared; the middle
    frame is what rejects intermediate keys replaced by the final value.
    """
    bounds = timeline.stage_bounds()
    result: dict[str, tuple[int, ...]] = {}
    for name in STAGED_STAGES:
        begin, end = bounds[name]
        result[name] = tuple(sorted({begin, (begin + end - 1) // 2, end - 1}))
    return result


def _transform_json(
    scale: np.ndarray, yaw: float, location: np.ndarray
) -> dict[str, Any]:
    return {
        "scale": [float(value) for value in np.asarray(scale, np.float64)],
        "yaw_rad": float(yaw),
        "yaw_deg": float(math.degrees(yaw)),
        "location_m": [
            float(value) for value in np.asarray(location, np.float64)
        ],
    }


def compare_stage_transforms(
    timeline: PlacementTimeline,
    expected: dict[int, dict[str, Any]],
    actual: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare the staged ``Body`` transform with ``body_transform_at``.

    ``expected`` and ``actual`` map a frame to ``{"scale": (sx, sy, sz),
    "yaw": radians, "location": (x, y, z)}``.  Scale and location use
    TOLERANCE_M as their numeric bound; the Z rotation is wrapped to
    ``[-pi, pi)`` and compared in degrees (ANGLE_TOLERANCE_DEG).
    """
    failures: list[dict[str, Any]] = []
    stages: dict[str, Any] = {}
    for name, frames in stage_sample_frames(timeline).items():
        entries: list[dict[str, Any]] = []
        max_scale = 0.0
        max_yaw = 0.0
        max_location = 0.0
        for frame in frames:
            expected_item = expected.get(frame)
            actual_item = actual.get(frame)
            if expected_item is None or actual_item is None:
                failures.append(
                    {
                        "check": "stage_transform",
                        "stage": name,
                        "frame": int(frame),
                        "error": "missing expected or actual transform",
                    }
                )
                entries.append({"frame": int(frame), "passed": False})
                continue
            expected_scale = np.asarray(expected_item["scale"], np.float64)
            expected_yaw = float(expected_item["yaw"])
            expected_location = np.asarray(expected_item["location"], np.float64)
            actual_scale = np.asarray(actual_item["scale"], np.float64)
            actual_yaw = float(actual_item["yaw"])
            actual_location = np.asarray(actual_item["location"], np.float64)
            scale_error = float(np.max(np.abs(actual_scale - expected_scale)))
            yaw_error = _angle_error(actual_yaw, expected_yaw)
            location_error = float(
                np.max(np.abs(actual_location - expected_location))
            )
            max_scale = max(max_scale, scale_error)
            max_yaw = max(max_yaw, yaw_error)
            max_location = max(max_location, location_error)
            passed = (
                math.isfinite(scale_error)
                and math.isfinite(yaw_error)
                and math.isfinite(location_error)
                and scale_error <= TOLERANCE_M
                and yaw_error <= ANGLE_TOLERANCE_RAD
                and location_error <= TOLERANCE_M
            )
            entries.append(
                {
                    "frame": int(frame),
                    "max_scale_error": scale_error,
                    "yaw_error_deg": float(math.degrees(yaw_error)),
                    "max_location_error_m": location_error,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(
                    {
                        "check": "stage_transform",
                        "stage": name,
                        "frame": int(frame),
                        "expected": _transform_json(
                            expected_scale, expected_yaw, expected_location
                        ),
                        "actual": _transform_json(
                            actual_scale, actual_yaw, actual_location
                        ),
                        "max_scale_error": scale_error,
                        "yaw_error_deg": float(math.degrees(yaw_error)),
                        "max_location_error_m": location_error,
                        "scale_tolerance": TOLERANCE_M,
                        "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
                        "tolerance_m": TOLERANCE_M,
                    }
                )
        stages[name] = {
            "frames": entries,
            "max_scale_error": max_scale,
            "max_yaw_error_deg": float(math.degrees(max_yaw)),
            "max_location_error_m": max_location,
            "passed": all(item.get("passed", False) for item in entries),
        }
    passed = all(stage["passed"] for stage in stages.values())
    return (
        {
            "frames_checked": sum(len(stage["frames"]) for stage in stages.values()),
            "stages": stages,
            "scale_tolerance": TOLERANCE_M,
            "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
            "tolerance_m": TOLERANCE_M,
            "passed": passed,
        },
        failures,
    )


def compare_cache_hold(
    timeline: PlacementTimeline,
    expected_t0: np.ndarray,
    actual_local: dict[int, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check that staged ``Body`` local vertices stay on the ``t0`` PC2 sample."""
    expected = np.asarray(expected_t0, np.float64)
    failures: list[dict[str, Any]] = []
    stages: dict[str, Any] = {}
    for name, frames in stage_sample_frames(timeline).items():
        entries: list[dict[str, Any]] = []
        maximum = 0.0
        for frame in frames:
            actual = actual_local.get(frame)
            if actual is None:
                failures.append(
                    {
                        "check": "cache_hold",
                        "stage": name,
                        "frame": int(frame),
                        "error": "missing evaluated Body vertices",
                    }
                )
                entries.append({"frame": int(frame), "passed": False})
                continue
            actual_array = np.asarray(actual, np.float64)
            if actual_array.shape != expected.shape:
                failures.append(
                    {
                        "check": "cache_hold",
                        "stage": name,
                        "frame": int(frame),
                        "error": f"vertex count {actual_array.shape} != "
                        f"{expected.shape}",
                    }
                )
                entries.append({"frame": int(frame), "passed": False})
                continue
            error = float(np.max(np.abs(actual_array - expected)))
            maximum = max(maximum, error)
            passed = math.isfinite(error) and error <= TOLERANCE_M
            entries.append(
                {
                    "frame": int(frame),
                    "max_abs_error_m": error,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(
                    {
                        "check": "cache_hold",
                        "stage": name,
                        "frame": int(frame),
                        "max_abs_error_m": error,
                        "tolerance_m": TOLERANCE_M,
                    }
                )
        stages[name] = {
            "frames": entries,
            "max_abs_error_m": maximum,
            "passed": all(item.get("passed", False) for item in entries),
        }
    passed = all(stage["passed"] for stage in stages.values())
    return (
        {
            "frames_checked": sum(len(stage["frames"]) for stage in stages.values()),
            "stages": stages,
            "tolerance_m": TOLERANCE_M,
            "passed": passed,
        },
        failures,
    )


def replay_vertex_indices(point_count: int) -> np.ndarray:
    """Fixed, evenly spaced vertex subset used on every replay frame."""
    if point_count <= 0:
        raise ValueError("body_local.pc2 has no points.")
    count = min(REPLAY_VERTEX_COUNT, int(point_count))
    return np.unique(np.round(np.linspace(0, point_count - 1, count)).astype(np.int64))


def compare_replay_vertices(
    timeline: PlacementTimeline,
    cached: np.ndarray,
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
    scale: float,
    valid: np.ndarray,
    actual_world: dict[int, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare evaluated replay ``Body`` vertices with the placement trace.

    ``actual_world`` holds the selected vertices (see ``replay_vertex_indices``)
    for the frames that were visible.  Every valid replay frame is checked; the
    PC2 sample index comes from ``timeline.pc2_frame_at``, so a frozen ``t0``
    cache or an off-by-one fps mapping is rejected.
    """
    cached_array = np.asarray(cached)
    indices = replay_vertex_indices(cached_array.shape[1])
    begin, end = timeline.stage_bounds()["replay"]
    failures: list[dict[str, Any]] = []
    maximum = 0.0
    checked = 0
    for frame in range(begin, end):
        index = int(timeline.pc2_frame_at(frame))
        if not bool(valid[index]):
            continue
        observed = actual_world.get(frame)
        if observed is None:
            failures.append(
                {
                    "check": "replay_body",
                    "frame": int(frame),
                    "source_frame": int(timeline.source_frame_at(frame)),
                    "error": "missing evaluated Body vertices",
                }
            )
            continue
        expected = np.asarray(hip_position[index], np.float64) + float(scale) * (
            _rotation_z(float(yaw_correction[index]))
            @ np.asarray(cached_array[index][indices], np.float64).T
        ).T
        observed_array = np.asarray(observed, np.float64)
        if observed_array.shape != expected.shape:
            failures.append(
                {
                    "check": "replay_body",
                    "frame": int(frame),
                    "source_frame": int(timeline.source_frame_at(frame)),
                    "error": f"vertex subset shape {observed_array.shape} != "
                    f"{expected.shape}",
                }
            )
            continue
        if not np.isfinite(expected).all() or not np.isfinite(observed_array).all():
            failures.append(
                {
                    "check": "replay_body",
                    "frame": int(frame),
                    "source_frame": int(timeline.source_frame_at(frame)),
                    "error": "non-finite vertex coordinate",
                }
            )
            continue
        error = float(np.max(np.abs(observed_array - expected)))
        maximum = max(maximum, error)
        checked += 1
        if error > TOLERANCE_M:
            failures.append(
                {
                    "check": "replay_body",
                    "frame": int(frame),
                    "source_frame": int(timeline.source_frame_at(frame)),
                    "pc2_frame": index,
                    "max_abs_error_m": error,
                    "tolerance_m": TOLERANCE_M,
                }
            )
    return (
        {
            "frames_checked": checked,
            "vertices_checked": int(len(indices)),
            "vertex_indices": [int(value) for value in indices],
            "max_abs_error_m": maximum,
            "tolerance_m": TOLERANCE_M,
            "passed": not any(item["check"] == "replay_body" for item in failures),
        },
        failures,
    )


def expected_camera_directions(
    timeline: PlacementTimeline,
    shot: PlacementShot,
    t0_forward: np.ndarray,
) -> dict[int, np.ndarray]:
    """Absolute camera direction from the hip for every replay frame.

    The reference is the fixed ``t0`` body forward (never the per-frame body
    yaw); the camera starts at ``camera_azimuth_offset_deg`` and orbits by
    ``camera_orbit_deg`` with the same progress used to key the camera.
    """
    begin, end = timeline.stage_bounds()["replay"]
    count = end - begin
    return {
        frame: _camera_direction(
            shot, t0_forward, (frame - begin) / max(count - 1, 1)
        )
        for frame in range(begin, end)
    }


def compare_camera_azimuths(
    timeline: PlacementTimeline,
    expected_directions: dict[int, np.ndarray],
    actual_offsets: dict[int, np.ndarray],
    camera_distance_m: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare absolute camera-hip azimuths, wrapped to ``[-pi, pi)``.

    The actual and expected azimuths are each measured from the world +X axis;
    the expected value already contains ``camera_azimuth_offset_deg``.  The
    first frame is not used as a reference, so a uniform rotation of the whole
    replay cannot cancel itself.
    """
    begin, end = timeline.stage_bounds()["replay"]
    failures: list[dict[str, Any]] = []
    maximum_angle = 0.0
    maximum_distance = 0.0
    checked = 0
    endpoints: dict[str, float] = {}
    for frame in range(begin, end):
        direction = expected_directions.get(frame)
        offset = actual_offsets.get(frame)
        if direction is None or offset is None:
            failures.append(
                {
                    "check": "camera_orbit",
                    "frame": int(frame),
                    "error": "missing expected or actual camera offset",
                }
            )
            continue
        expected = np.asarray(direction, np.float64)
        horizontal = np.asarray(np.asarray(offset, np.float64)[:2], np.float64)
        norm = float(np.linalg.norm(horizontal))
        if (
            not np.isfinite(expected[:2]).all()
            or not np.isfinite(horizontal).all()
            or not math.isfinite(norm)
            or norm <= 1e-9
        ):
            failures.append(
                {
                    "check": "camera_orbit",
                    "frame": int(frame),
                    "error": "camera-hip horizontal offset is zero/non-finite.",
                }
            )
            continue
        expected_azimuth = math.atan2(float(expected[1]), float(expected[0]))
        actual_azimuth = math.atan2(float(horizontal[1]), float(horizontal[0]))
        azimuth_error = _angle_error(actual_azimuth, expected_azimuth)
        distance_error = abs(norm - float(camera_distance_m))
        maximum_angle = max(maximum_angle, azimuth_error)
        maximum_distance = max(maximum_distance, distance_error)
        checked += 1
        if frame == begin:
            endpoints = {
                "expected_start_deg": float(math.degrees(expected_azimuth)),
                "actual_start_deg": float(math.degrees(actual_azimuth)),
            }
        if frame == end - 1:
            endpoints.update(
                expected_end_deg=float(math.degrees(expected_azimuth)),
                actual_end_deg=float(math.degrees(actual_azimuth)),
            )
        if azimuth_error > ANGLE_TOLERANCE_RAD:
            failures.append(
                {
                    "check": "camera_orbit",
                    "frame": int(frame),
                    "source_frame": int(timeline.source_frame_at(frame)),
                    "expected_azimuth_deg": float(math.degrees(expected_azimuth)),
                    "actual_azimuth_deg": float(math.degrees(actual_azimuth)),
                    "max_azimuth_error_deg": float(math.degrees(azimuth_error)),
                    "tolerance_deg": ANGLE_TOLERANCE_DEG,
                }
            )
        if distance_error > TOLERANCE_M:
            failures.append(
                {
                    "check": "camera_orbit",
                    "frame": int(frame),
                    "source_frame": int(timeline.source_frame_at(frame)),
                    "camera_distance_m": norm,
                    "expected_camera_distance_m": float(camera_distance_m),
                    "max_distance_error_m": distance_error,
                    "tolerance_m": TOLERANCE_M,
                }
            )
    return (
        {
            "frames": checked,
            **endpoints,
            "max_azimuth_error_deg": float(math.degrees(maximum_angle)),
            "max_distance_error_m": maximum_distance,
            "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
            "tolerance_m": TOLERANCE_M,
            "passed": not any(item["check"] == "camera_orbit" for item in failures),
        },
        failures,
    )



def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _object(name: str) -> Any:
    bpy = _bpy()
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(f"Scene is missing required object {name!r}.")
    return obj


def _world_matrix(obj: Any) -> np.ndarray:
    """Compose the transform from ``matrix_basis`` and parents.

    Blender leaves ``matrix_world`` stale for hidden objects, which happens on
    every invalid placement frame.  ``matrix_basis`` and the parent chain stay
    current, so composing them evaluates the keyed transform without changing
    visibility.
    """
    matrix = np.eye(4)
    node = obj
    while node is not None:
        matrix = np.asarray(node.matrix_basis, np.float64) @ matrix
        if node.parent is not None:
            matrix = np.asarray(node.matrix_parent_inverse, np.float64) @ matrix
        node = node.parent
    return matrix


def _body_vertices(obj: Any) -> np.ndarray:
    bpy = _bpy()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    vertices = np.empty((len(evaluated.data.vertices), 3), dtype=np.float64)
    evaluated.data.vertices.foreach_get("co", vertices.ravel())
    matrix = _world_matrix(obj)
    return vertices @ matrix[:3, :3].T + matrix[:3, 3]


def _body_local_vertices(obj: Any) -> np.ndarray:
    """Evaluated mesh vertices in ``Body`` local space (cache deformation only)."""
    bpy = _bpy()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    vertices = np.empty((len(evaluated.data.vertices), 3), dtype=np.float64)
    evaluated.data.vertices.foreach_get("co", vertices.ravel())
    return vertices


def _check_hidden(
    failures: list[dict[str, Any]],
    name: str,
    obj: Any,
    frame: int,
    expected_visible: bool,
) -> None:
    for attribute in ("hide_render", "hide_viewport"):
        actual_hidden = bool(getattr(obj, attribute))
        if actual_hidden != (not expected_visible):
            failures.append(
                {
                    "check": "visibility",
                    "object": name,
                    "frame": int(frame),
                    "attribute": attribute,
                    "expected_visible": bool(expected_visible),
                    "actual_hidden": actual_hidden,
                }
            )


def _validate_visibility(
    timeline: PlacementTimeline,
    valid: np.ndarray,
    joints_local: np.ndarray,
    failures: list[dict[str, Any]],
) -> int:
    body = _object("Body")
    sources = [_object(f"SourceJoint{joint:02d}") for joint in range(JOINT_COUNT)]
    checked_frames = 0
    bpy = _bpy()
    for frame in range(timeline.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        visible = body_visible_at(timeline, frame, valid)
        pc2_index = timeline.pc2_frame_at(frame)
        _check_hidden(failures, "Body", body, frame, visible)
        for joint, obj in enumerate(sources):
            finite = bool(np.isfinite(joints_local[pc2_index, joint]).all())
            _check_hidden(
                failures, f"SourceJoint{joint:02d}", obj, frame, visible and finite
            )
        checked_frames += 1
    return checked_frames


def _validate_t0_vertices(
    timeline: PlacementTimeline,
    placed_vertices_t0: np.ndarray,
    valid: np.ndarray,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    body = _object("Body")
    begin, end = timeline.stage_bounds()["final"]
    frames = sorted({begin, end - 1})
    maximum = 0.0
    bpy = _bpy()
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        visible = body_visible_at(timeline, frame, valid)
        if not visible:
            failures.append(
                {"check": "t0_body", "frame": int(frame), "error": "Body is hidden."}
            )
            continue
        world = _body_vertices(body)
        if world.shape != placed_vertices_t0.shape:
            failures.append(
                {
                    "check": "t0_body",
                    "frame": int(frame),
                    "error": f"vertex count {world.shape[0]} != "
                    f"{placed_vertices_t0.shape[0]}",
                }
            )
            continue
        if not np.isfinite(world).all() or not np.isfinite(placed_vertices_t0).all():
            failures.append(
                {
                    "check": "t0_body",
                    "frame": int(frame),
                    "error": "non-finite vertex coordinate",
                }
            )
            continue
        error = float(np.max(np.abs(world - placed_vertices_t0)))
        maximum = max(maximum, error)
        if error > TOLERANCE_M:
            failures.append(
                {
                    "check": "t0_body",
                    "frame": int(frame),
                    "max_abs_error_m": error,
                    "tolerance_m": TOLERANCE_M,
                }
            )
    return {
        "frames": [int(frame) for frame in frames],
        "max_abs_error_m": maximum,
        "tolerance_m": TOLERANCE_M,
        "passed": not any(item["check"] == "t0_body" for item in failures),
    }


def _expected_body_transform(
    timeline: PlacementTimeline,
    frame: int,
    *,
    scale: float,
    t0_yaw: float,
    t0_hip: np.ndarray,
    anchor: np.ndarray,
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
) -> dict[str, Any]:
    frame_scale, frame_yaw, frame_location = body_transform_at(
        timeline,
        frame,
        scale=scale,
        t0_yaw=t0_yaw,
        t0_hip=t0_hip,
        anchor=anchor,
        hip_position=hip_position,
        yaw_correction=yaw_correction,
    )
    return {
        "scale": np.full(3, float(frame_scale), np.float64),
        "yaw": float(frame_yaw),
        "location": np.asarray(frame_location, np.float64),
    }


def _validate_stage_transforms(
    timeline: PlacementTimeline,
    manifest: dict[str, Any],
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read the evaluated staged ``Body`` transform from the real scene."""
    bpy = _bpy()
    scene = bpy.context.scene
    body = _object("Body")
    t0_index = timeline.t0 - timeline.start
    t0_hip = np.asarray(hip_position[t0_index], np.float64)
    t0_yaw = float(yaw_correction[t0_index])
    anchor = t0_hip + np.asarray(timeline.shot.display_offset_m, np.float64)
    scale = float(manifest["scale"])
    expected: dict[int, dict[str, Any]] = {}
    actual: dict[int, dict[str, Any]] = {}
    for frames in stage_sample_frames(timeline).values():
        for frame in frames:
            expected[frame] = _expected_body_transform(
                timeline,
                frame,
                scale=scale,
                t0_yaw=t0_yaw,
                t0_hip=t0_hip,
                anchor=anchor,
                hip_position=hip_position,
                yaw_correction=yaw_correction,
            )
            scene.frame_set(frame)
            matrix = _world_matrix(body)
            actual[frame] = {
                "scale": np.asarray(
                    [
                        np.linalg.norm(matrix[:3, 0]),
                        np.linalg.norm(matrix[:3, 1]),
                        np.linalg.norm(matrix[:3, 2]),
                    ],
                    np.float64,
                ),
                "yaw": float(math.atan2(matrix[1, 0], matrix[0, 0])),
                "location": np.asarray(matrix[:3, 3], np.float64),
            }
    info, stage_failures = compare_stage_transforms(timeline, expected, actual)
    failures.extend(stage_failures)
    return info


def _validate_cache_hold(
    timeline: PlacementTimeline,
    bundle: Path,
    manifest: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read the evaluated staged ``Body`` local vertices from the real scene."""
    bpy = _bpy()
    scene = bpy.context.scene
    body = _object("Body")
    pc2_path = bundle / str(manifest["files"]["body_local"])
    points, samples = _pc2_shape(pc2_path)
    cached = np.memmap(
        pc2_path, dtype="<f4", mode="r", offset=32, shape=(samples, points, 3)
    )
    t0_index = timeline.t0 - timeline.start
    expected = np.asarray(cached[t0_index], np.float64)
    actual: dict[int, np.ndarray] = {}
    for frames in stage_sample_frames(timeline).values():
        for frame in frames:
            scene.frame_set(frame)
            actual[frame] = _body_local_vertices(body)
    del cached
    info, cache_failures = compare_cache_hold(timeline, expected, actual)
    failures.extend(cache_failures)
    return info


def _validate_replay_body(
    timeline: PlacementTimeline,
    bundle: Path,
    manifest: dict[str, Any],
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
    valid: np.ndarray,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read evaluated replay ``Body`` world vertices from the real scene."""
    bpy = _bpy()
    scene = bpy.context.scene
    body = _object("Body")
    pc2_path = bundle / str(manifest["files"]["body_local"])
    points, samples = _pc2_shape(pc2_path)
    cached = np.memmap(
        pc2_path, dtype="<f4", mode="r", offset=32, shape=(samples, points, 3)
    )
    indices = replay_vertex_indices(points)
    begin, end = timeline.stage_bounds()["replay"]
    actual: dict[int, np.ndarray] = {}
    for frame in range(begin, end):
        index = int(timeline.pc2_frame_at(frame))
        if not bool(valid[index]):
            continue
        scene.frame_set(frame)
        world = _body_vertices(body)
        if world.shape != (points, 3):
            failures.append(
                {
                    "check": "replay_body",
                    "frame": int(frame),
                    "source_frame": int(timeline.source_frame_at(frame)),
                    "error": f"vertex count {world.shape[0]} != {points}",
                }
            )
            continue
        actual[frame] = world[indices]
    scale = float(manifest["scale"])
    info, replay_failures = compare_replay_vertices(
        timeline,
        cached,
        hip_position,
        yaw_correction,
        scale,
        valid,
        actual,
    )
    del cached
    failures.extend(replay_failures)
    return info


def _validate_replay_joints(
    timeline: PlacementTimeline,
    manifest: dict[str, Any],
    joints_local: np.ndarray,
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    scale = float(manifest["scale"])
    begin, end = timeline.stage_bounds()["replay"]
    maximum = 0.0
    checked = 0
    bpy = _bpy()
    body = _object("Body")
    sources = [_object(f"SourceJoint{joint:02d}") for joint in range(JOINT_COUNT)]
    for joint, obj in enumerate(sources):
        if obj.parent is not body:
            failures.append(
                {
                    "check": "replay_joints",
                    "joint": joint,
                    "error": "SourceJoint is not parented to Body.",
                }
            )
        if not np.allclose(np.asarray(obj.matrix_parent_inverse), np.eye(4)):
            failures.append(
                {
                    "check": "replay_joints",
                    "joint": joint,
                    "error": "SourceJoint parent inverse is not identity.",
                }
            )
    for frame in range(begin, end):
        bpy.context.scene.frame_set(frame)
        source_frame = timeline.replay_source_frames[frame - begin]
        index = source_frame - timeline.start
        expected = hip_position[index] + scale * (
            _rotation_z(float(yaw_correction[index])) @ joints_local[index].T
        ).T
        if not np.isfinite(expected).all():
            failures.append(
                {
                    "check": "replay_joints",
                    "frame": int(frame),
                    "source_frame": int(source_frame),
                    "error": "expected world position is non-finite",
                }
            )
            continue
        for joint, obj in enumerate(sources):
            # Blender does not refresh matrix_world for hidden objects, so the
            # world transform is composed from the keyed parent chain (the
            # identity parent inverse is checked above).
            actual = _world_matrix(obj)[:3, 3]
            if not bool(obj.hide_viewport):
                evaluated = np.asarray(obj.matrix_world.translation, np.float64)
                if not np.allclose(evaluated, actual, atol=1e-6, rtol=0.0):
                    failures.append(
                        {
                            "check": "replay_joints",
                            "frame": int(frame),
                            "joint": joint,
                            "error": "evaluated world transform disagrees with "
                            "the composed parent/basis transform",
                        }
                    )
            if not np.isfinite(actual).all():
                failures.append(
                    {
                        "check": "replay_joints",
                        "frame": int(frame),
                        "source_frame": int(source_frame),
                        "joint": joint,
                        "error": "actual world position is non-finite",
                    }
                )
                continue
            error = float(np.max(np.abs(actual - expected[joint])))
            maximum = max(maximum, error)
            if error > TOLERANCE_M:
                failures.append(
                    {
                        "check": "replay_joints",
                        "frame": int(frame),
                        "source_frame": int(source_frame),
                        "joint": joint,
                        "max_abs_error_m": error,
                        "tolerance_m": TOLERANCE_M,
                    }
                )
            checked += 1
    return {
        "frames_checked": end - begin,
        "joint_positions_checked": checked,
        "max_abs_error_m": maximum,
        "tolerance_m": TOLERANCE_M,
        "passed": not any(item["check"] == "replay_joints" for item in failures),
    }


def _find_plate_texture(objects: list[Any]) -> Any:
    for obj in objects:
        if obj.type != "MESH" or not obj.data.materials:
            continue
        for material in obj.data.materials:
            if material is None or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    return node
    return None


def _t0_forward(
    timeline: PlacementTimeline,
    joints_local: np.ndarray,
    yaw_correction: np.ndarray,
) -> np.ndarray:
    index = timeline.t0 - timeline.start
    return body_forward(float(yaw_correction[index]), joints_local[index])


def _validate_camera_orbit(
    timeline: PlacementTimeline,
    joints_local: np.ndarray,
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare the actual replay camera azimuth with the absolute t0 orbit.

    The expected direction comes from ``_camera_direction(shot, t0_forward,
    progress)`` and is measured from the world +X axis.  The actual direction is
    ``camera - hip_position[source_frame]``.  No first-frame offset is
    subtracted, so a uniform rotation of the whole replay cannot cancel itself.
    """
    bpy = _bpy()
    shot = timeline.shot
    forward = _t0_forward(timeline, joints_local, yaw_correction)
    expected = expected_camera_directions(timeline, shot, forward)
    begin, end = timeline.stage_bounds()["replay"]
    scene = bpy.context.scene
    actual: dict[int, np.ndarray] = {}
    for frame in range(begin, end):
        scene.frame_set(frame)
        camera = scene.camera
        if camera is None:
            failures.append(
                {"check": "camera_orbit", "error": "Scene has no camera."}
            )
            return {"passed": False}
        index = timeline.source_frame_at(frame) - timeline.start
        actual[frame] = np.asarray(
            _world_matrix(camera)[:3, 3] - hip_position[index], np.float64
        )
    info, orbit_failures = compare_camera_azimuths(
        timeline, expected, actual, shot.camera_distance_m
    )
    failures.extend(orbit_failures)
    return info


def _validate_camera_front(
    timeline: PlacementTimeline,
    joints_local: np.ndarray,
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """The pre-replay camera must sit on the body-forward side, not the back."""
    bpy = _bpy()
    shot = timeline.shot
    t0_index = timeline.t0 - timeline.start
    forward = _t0_forward(timeline, joints_local, yaw_correction)
    expected = _camera_direction(shot, forward, 0.0)
    scene = bpy.context.scene
    scene.frame_set(0)
    camera = scene.camera
    if camera is None:
        failures.append({"check": "camera_front", "error": "Scene has no camera."})
        return {"passed": False}
    offset = np.asarray(camera.location, np.float64) - hip_position[t0_index]
    horizontal = np.asarray(offset[:2], np.float64)
    norm = float(np.linalg.norm(horizontal))
    if not np.isfinite(horizontal).all() or not math.isfinite(norm) or norm <= 1e-9:
        failures.append(
            {
                "check": "camera_front",
                "error": "Camera position has no finite horizontal offset from t0.",
            }
        )
        return {"passed": False}
    actual = np.asarray([horizontal[0] / norm, horizontal[1] / norm, 0.0])
    cosine = float(np.dot(forward, actual))
    expected_cosine = float(np.dot(forward, expected))
    error = abs(cosine - expected_cosine)
    if error > 1e-6:
        failures.append(
            {
                "check": "camera_front",
                "cosine": cosine,
                "expected_cosine": expected_cosine,
                "max_abs_error": error,
            }
        )
    return {
        "cosine": cosine,
        "expected_cosine": expected_cosine,
        "offset_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, cosine))))),
        "max_abs_error": error,
        "passed": not any(item["check"] == "camera_front" for item in failures),
    }


def _validate_intro_plate(
    bundle: Path,
    manifest: dict[str, Any],
    timeline: PlacementTimeline,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    info: dict[str, Any] = {"enabled": bool(timeline.shot.intro_plate)}
    bpy = _bpy()
    plate = bpy.data.objects.get(INTRO_PLATE_OBJECT)
    if not timeline.shot.intro_plate:
        if plate is not None:
            failures.append(
                {
                    "check": "intro_plate",
                    "error": "IntroPlate exists although intro_plate is disabled.",
                }
            )
        info["passed"] = not any(
            failure["check"] == "intro_plate" for failure in failures
        )
        return info
    if plate is None:
        failures.append(
            {"check": "intro_plate", "error": "IntroPlate object is missing."}
        )
        return info
    scene = bpy.context.scene
    for frame in range(timeline.frame_end + 1):
        scene.frame_set(frame)
        _check_hidden(
            failures,
            INTRO_PLATE_OBJECT,
            plate,
            frame,
            frame < timeline.intro_frames,
        )
    shot_dir = Path(bpy.data.filepath).resolve().parent
    pattern = str(manifest["plate"]["pattern"])
    compared = 0
    for index, source_frame in enumerate(timeline.intro_source_frames):
        copied = shot_dir / "intro_plate" / f"{index:05d}.jpg"
        original = bundle / pattern.format(frame=source_frame)
        if not copied.is_file() or not original.is_file():
            failures.append(
                {
                    "check": "intro_plate",
                    "index": index,
                    "source_frame": int(source_frame),
                    "error": "plate frame file is missing",
                }
            )
            continue
        if copied.read_bytes() != original.read_bytes():
            failures.append(
                {
                    "check": "intro_plate",
                    "index": index,
                    "source_frame": int(source_frame),
                    "error": "copied plate frame is not the selected source frame",
                }
            )
        compared += 1
    texture = _find_plate_texture([plate])
    if texture is None:
        failures.append(
            {"check": "intro_plate", "error": "IntroPlate has no image texture."}
        )
    else:
        user = texture.image_user
        expected = {
            "frame_duration": timeline.intro_frames,
            "frame_start": 0,
            "frame_offset": -1,
        }
        for name, value in expected.items():
            if int(getattr(user, name)) != int(value):
                failures.append(
                    {
                        "check": "intro_plate",
                        "attribute": name,
                        "expected": value,
                        "actual": int(getattr(user, name)),
                    }
                )
        expected_path = (shot_dir / "intro_plate" / "00000.jpg").resolve()
        actual_path = Path(bpy.path.abspath(texture.image.filepath)).resolve()
        if actual_path != expected_path:
            failures.append(
                {
                    "check": "intro_plate",
                    "attribute": "image.filepath",
                    "expected": str(expected_path),
                    "actual": str(actual_path),
                }
            )
    info.update(
        frames=timeline.intro_frames,
        source_frames=[int(frame) for frame in timeline.intro_source_frames],
        files_compared=compared,
    )
    info["passed"] = not any(
        failure["check"] == "intro_plate" for failure in failures
    )
    return info


def validate(bundle: Path) -> dict[str, Any]:
    bpy = _bpy()
    scene = bpy.context.scene
    manifest = load_manifest(bundle)
    if "placement_bundle" not in scene:
        raise ValueError("Scene has no placement_bundle custom property.")
    if "placement_shot" not in scene:
        raise ValueError("Scene has no placement_shot custom property.")
    stored_bundle = Path(str(scene["placement_bundle"])).resolve()
    if stored_bundle != bundle.resolve():
        raise ValueError(
            f"Scene was built from {stored_bundle}, not {bundle.resolve()}."
        )
    shot = PlacementShot(**json.loads(str(scene["placement_shot"])))
    timeline = build_timeline(manifest, shot)

    actual_fps = scene.render.fps / scene.render.fps_base
    if not np.isclose(actual_fps, OUTPUT_FPS, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Placement scene must run at {OUTPUT_FPS} fps, got {actual_fps}."
        )
    if (scene.frame_start, scene.frame_end) != (0, timeline.frame_end):
        raise ValueError(
            "Placement scene frame range "
            f"{scene.frame_start}:{scene.frame_end} does not match the full "
            f"timeline 0:{timeline.frame_end}; validate the un-subset scene."
        )
    if (scene.render.resolution_x, scene.render.resolution_y) != (
        int(manifest["width"]),
        int(manifest["height"]),
    ):
        raise ValueError("Scene resolution disagrees with the bundle manifest.")

    files = manifest["files"]
    joints_local = np.asarray(np.load(bundle / str(files["joints_local"])), np.float64)
    target_weights = np.asarray(
        np.load(bundle / str(files["target_weights"])), np.float64
    )
    hip_position = np.asarray(np.load(bundle / str(files["hip_position"])), np.float64)
    yaw_correction = np.asarray(
        np.load(bundle / str(files["yaw_correction"])), np.float64
    )
    valid = np.asarray(np.load(bundle / str(files["valid"])), bool)
    placed = np.asarray(
        np.load(bundle / str(files["placed_vertices_t0"])), np.float64
    )
    if joints_local.shape != (timeline.stop - timeline.start, JOINT_COUNT, 3):
        raise ValueError("joints_local shape disagrees with the bundle window.")
    if valid.shape != (timeline.stop - timeline.start,):
        raise ValueError("valid shape disagrees with the bundle window.")

    failures: list[dict[str, Any]] = []
    body_t0 = _validate_t0_vertices(timeline, placed, valid, failures)
    stage_transforms = _validate_stage_transforms(
        timeline, manifest, hip_position, yaw_correction, failures
    )
    body_cache_hold = _validate_cache_hold(timeline, bundle, manifest, failures)
    replay_body = _validate_replay_body(
        timeline,
        bundle,
        manifest,
        hip_position,
        yaw_correction,
        valid,
        failures,
    )
    replay_joints = _validate_replay_joints(
        timeline, manifest, joints_local, hip_position, yaw_correction, failures
    )
    visibility_frames = _validate_visibility(
        timeline, valid, joints_local, failures
    )
    camera_front = _validate_camera_front(
        timeline, joints_local, hip_position, yaw_correction, failures
    )
    camera_orbit = _validate_camera_orbit(
        timeline, joints_local, hip_position, yaw_correction, failures
    )
    intro_plate = _validate_intro_plate(bundle, manifest, timeline, failures)

    pc2_path = bundle / str(files["body_local"])
    points, samples = _pc2_shape(pc2_path)
    cached = np.memmap(
        pc2_path,
        dtype="<f4",
        mode="r",
        offset=32,
        shape=(samples, points, 3),
    )
    invalid_zero = bool(
        not (~valid).any() or np.all(np.asarray(cached[~valid]) == 0.0)
    )
    if not invalid_zero:
        failures.append(
            {
                "check": "pc2_invalid_zero",
                "error": "body_local.pc2 has non-zero vertices on invalid frames.",
            }
        )
    del cached
    if target_weights.shape != (timeline.stop - timeline.start, JOINT_COUNT):
        raise ValueError("target_weights shape disagrees with the bundle window.")

    return {
        "ok": not failures,
        "bundle": str(bundle.resolve()),
        "scene": str(bpy.data.filepath),
        "blender_version": bpy.app.version_string,
        "schema": manifest["schema"],
        "input_fps": float(manifest["fps"]),
        "scene_fps": float(actual_fps),
        "scene_frames": timeline.frame_end + 1,
        "intro_plate": intro_plate,
        "stages": stage_transforms["stages"],
        "replay": {
            "output_frames": len(timeline.replay_source_frames),
            "source_start": timeline.start,
            "source_stop": timeline.stop,
            "first_source_frames": list(timeline.replay_source_frames[:5]),
            "last_source_frames": list(timeline.replay_source_frames[-5:]),
        },
        "checks": {
            "body_t0": body_t0,
            "stage_transforms": stage_transforms,
            "body_cache_hold": body_cache_hold,
            "replay_body_vertices": replay_body,
            "replay_source_joints": replay_joints,
            "visibility": {
                "frames_checked": visibility_frames,
                "objects_checked": 1 + JOINT_COUNT,
                "passed": not any(
                    item["check"] == "visibility" for item in failures
                ),
            },
            "camera_front": camera_front,
            "camera_orbit": camera_orbit,
            "pc2_invalid_zero": invalid_zero,
        },
        "failures": failures,
    }


def _pc2_shape(path: Path) -> tuple[int, int]:
    header = np.fromfile(path, dtype=np.uint8, count=32)
    if header.size != 32:
        raise ValueError(f"{path} is too short to be a PC2 file.")
    points = int(np.frombuffer(header[16:20].tobytes(), dtype="<i4")[0])
    samples = int(np.frombuffer(header[28:32].tobytes(), dtype="<i4")[0])
    return points, samples


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"ok": False, "bundle": str(args.bundle.resolve())}
    try:
        report = validate(args.bundle)
    except BaseException as error:
        report["ok"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        _write_report(args.report, report)
        print(json.dumps(report))
        raise
    _write_report(args.report, report)
    print(json.dumps(report))
    if not report["ok"]:
        first = report["failures"][:5]
        raise RuntimeError(
            f"Placement validation failed with {len(report['failures'])} issue(s): "
            f"{json.dumps(first)}"
        )


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    main(arguments)

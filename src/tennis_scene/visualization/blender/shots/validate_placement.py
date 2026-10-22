"""Validate a saved player-placement scene inside Blender.

blender --background <shot>/tennis_placement.blend --python-exit-code 1 \
    --python src/tennis_scene/visualization/blender/shots/validate_placement.py -- \
    --bundle <bundle> --report <shot>/validation.json

No rendering is performed.  The checks use Blender's evaluated mesh and object
matrices:

* at the final staged frames, the world-space ``Body`` vertices equal
  ``placed_vertices_t0`` (tolerance 1e-4 m);
* on every replay frame, each ``SourceJoints`` world position equals
  ``hip_position + scale * Rz(yaw) @ joints_local`` (tolerance 1e-4 m);
* ``Body`` and every ``SourceJoint`` hide in render *and* viewport exactly where
  ``valid`` is false (and only there);
* when ``intro_plate`` is enabled, the plate backdrop is visible only during the
  intro and every copied plate frame is the source frame selected from
  ``manifest["frames"]["start"]``.

Every comparison rejects non-finite values instead of letting NaN slip through.
The report is written on success and on failure; mismatches exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import bpy
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
    body_visible_at,
    build_timeline,
    load_manifest,
)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _object(name: str) -> Any:
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
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    vertices = np.empty((len(evaluated.data.vertices), 3), dtype=np.float64)
    evaluated.data.vertices.foreach_get("co", vertices.ravel())
    matrix = _world_matrix(obj)
    return vertices @ matrix[:3, :3].T + matrix[:3, 3]


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
    """The actual scene camera must orbit 45 deg around the moving hip.

    The azimuth is measured as ``camera - hip_position[source_frame]`` so the
    moving origin cannot be confused with the orbit.  The reference is the
    fixed t0 body forward; per-frame body yaw/pose is deliberately not used.
    """
    shot = timeline.shot
    forward = _t0_forward(timeline, joints_local, yaw_correction)
    base_direction = _camera_direction(shot, forward, 0.0)
    base_azimuth = math.atan2(float(base_direction[1]), float(base_direction[0]))
    begin, end = timeline.stage_bounds()["replay"]
    scene = bpy.context.scene
    azimuths: list[float] = []
    distances: list[float] = []
    for frame in range(begin, end):
        scene.frame_set(frame)
        camera = scene.camera
        if camera is None:
            failures.append(
                {"check": "camera_orbit", "error": "Scene has no camera."}
            )
            return {"passed": False}
        source_frame = timeline.replay_source_frames[frame - begin]
        index = source_frame - timeline.start
        offset = _world_matrix(camera)[:3, 3] - hip_position[index]
        horizontal = np.asarray(offset[:2], np.float64)
        norm = float(np.linalg.norm(horizontal))
        if (
            not np.isfinite(horizontal).all()
            or not math.isfinite(norm)
            or norm <= 1e-9
        ):
            failures.append(
                {
                    "check": "camera_orbit",
                    "frame": int(frame),
                    "source_frame": int(source_frame),
                    "error": "camera-hip horizontal offset is zero/non-finite.",
                }
            )
            continue
        azimuths.append(float(math.atan2(horizontal[1], horizontal[0])))
        distances.append(norm)
    if len(azimuths) != len(range(begin, end)):
        return {"passed": False}
    actual = np.unwrap(np.asarray(azimuths, np.float64))
    expected_total = math.radians(float(shot.camera_orbit_deg))
    count = len(actual)
    tolerance_deg = 1e-3
    tolerance_m = 1e-3
    total = float(actual[-1] - actual[0]) if count > 1 else 0.0
    step = 0.0
    step_error = 0.0
    azimuth_error = 0.0
    monotonic = True
    if count > 1:
        steps = np.diff(actual)
        step = float(steps[0]) if len(steps) else 0.0
        step_error = float(np.max(np.abs(steps - expected_total / (count - 1))))
        monotonic = bool((np.sign(steps) == np.sign(expected_total)).all())
        shifted = actual - actual[0] + base_azimuth
        expected = base_azimuth + (expected_total / (count - 1)) * np.arange(count)
        azimuth_error = float(np.max(np.abs(shifted - expected)))
        if abs(total - expected_total) > math.radians(tolerance_deg):
            failures.append(
                {
                    "check": "camera_orbit",
                    "error": "total camera-hip azimuth is not the configured orbit.",
                    "total_deg": float(math.degrees(total)),
                    "expected_total_deg": float(shot.camera_orbit_deg),
                }
            )
        if step_error > math.radians(tolerance_deg):
            failures.append(
                {
                    "check": "camera_orbit",
                    "error": "camera-hip orbit step is not uniform.",
                    "step_deg": float(math.degrees(step)),
                    "expected_step_deg": float(
                        math.degrees(expected_total / (count - 1))
                    ),
                    "max_step_error_deg": float(math.degrees(step_error)),
                }
            )
        if not monotonic:
            failures.append(
                {
                    "check": "camera_orbit",
                    "error": "camera-hip orbit is not monotonic.",
                }
            )
        if azimuth_error > math.radians(tolerance_deg):
            failures.append(
                {
                    "check": "camera_orbit",
                    "error": "camera-hip azimuth does not follow the fixed t0 orbit.",
                    "max_azimuth_error_deg": float(math.degrees(azimuth_error)),
                }
            )
    else:
        difference = math.atan2(
            math.sin(float(actual[0]) - base_azimuth),
            math.cos(float(actual[0]) - base_azimuth),
        )
        if abs(difference) > math.radians(tolerance_deg):
            failures.append(
                {
                    "check": "camera_orbit",
                    "error": "single replay camera is not on the t0 orbit start.",
                }
            )
    distance_error = float(
        np.max(np.abs(np.asarray(distances, np.float64) - shot.camera_distance_m))
    )
    if distance_error > tolerance_m:
        failures.append(
            {
                "check": "camera_orbit",
                "error": "camera-hip horizontal distance changed during replay.",
                "max_distance_error_m": distance_error,
            }
        )
    return {
        "frames": count,
        "total_deg": float(math.degrees(total)),
        "expected_total_deg": float(shot.camera_orbit_deg),
        "step_deg": float(math.degrees(step)),
        "expected_step_deg": float(
            math.degrees(expected_total / (count - 1)) if count > 1 else 0.0
        ),
        "max_step_error_deg": float(math.degrees(step_error)),
        "max_azimuth_error_deg": float(math.degrees(azimuth_error)),
        "max_distance_error_m": distance_error,
        "monotonic": monotonic,
        "passed": not any(item["check"] == "camera_orbit" for item in failures),
    }


def _validate_camera_front(
    timeline: PlacementTimeline,
    joints_local: np.ndarray,
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """The pre-replay camera must sit on the body-forward side, not the back."""
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
        "stages": {
            name: {"start": int(begin), "stop": int(end)}
            for name, begin, end in timeline.stages
        },
        "replay": {
            "output_frames": len(timeline.replay_source_frames),
            "source_start": timeline.start,
            "source_stop": timeline.stop,
            "first_source_frames": list(timeline.replay_source_frames[:5]),
            "last_source_frames": list(timeline.replay_source_frames[-5:]),
        },
        "checks": {
            "body_t0": body_t0,
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

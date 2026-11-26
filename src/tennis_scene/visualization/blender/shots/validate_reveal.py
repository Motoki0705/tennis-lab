"""Validate a saved reveal or playback scene against its exported bundle.

blender --background <shot>/tennis_reveal.blend --python-exit-code 1 \
    --python src/tennis_scene/visualization/blender/shots/validate_reveal.py -- \
    --bundle <bundle> --report <shot>/validation.json

No rendering is performed. Checks use Blender's evaluated meshes and camera,
including visibility on every source frame and PC2 samples across the clip.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from playback import (  # noqa: E402
    EVENT_HOLD_SECONDS,
    PLAYBACK_BUNDLE_SCHEMA,
    event_badge_name,
    event_ring_name,
)
from reveal import (  # noqa: E402
    PLAYBACK_LENS_MM,
    PLAYBACK_SENSOR_WIDTH_MM,
    Shot,
    load_manifest,
)


def _playback_visibility(
    manifest: dict[str, Any], frames: int
) -> dict[str, np.ndarray]:
    hold = math.ceil(EVENT_HOLD_SECONDS * manifest["fps"])
    badges: dict[str, np.ndarray] = {}
    visibility: dict[str, np.ndarray] = {}
    for event in manifest["ball_events"]:
        frame, kind = event["frame"], event["kind"]
        active = np.zeros(frames, dtype=bool)
        active[frame : frame + hold] = True
        if kind not in badges:
            badges[kind] = active.copy()
        else:
            badges[kind] |= active
        if event["position"] is not None:
            name = event_ring_name(kind, frame)
            np.testing.assert_allclose(
                bpy.data.objects[name].location, event["position"], atol=2e-5, rtol=0
            )
            visibility[name] = active
    for kind, active in badges.items():
        name = event_badge_name(kind)
        for suffix in ("", ".Accent", ".Label"):
            visibility[name + suffix] = active
    actual = {
        obj.name for obj in bpy.data.objects
        if obj.name.startswith(("BallEvent.", "EventBadge."))
    }
    assert actual == visibility.keys(), actual.symmetric_difference(visibility.keys())
    return visibility


def validate(bundle: Path) -> dict[str, object]:
    manifest = load_manifest(bundle)
    playback = manifest["schema"] == PLAYBACK_BUNDLE_SCHEMA
    scene = bpy.context.scene
    frames = int(manifest["num_frames"])
    samples = sorted({0, frames - 1, *(int(x) for x in np.linspace(0, frames - 1, 9))})
    player_valid = np.load(bundle / manifest["player_valid"])
    ball_valid = np.load(bundle / manifest["ball"]["valid"])
    ball_positions = np.load(bundle / manifest["ball"]["positions"])
    camera = scene.camera
    assert camera is not None
    actual_fps = scene.render.fps / scene.render.fps_base
    assert np.isclose(actual_fps, manifest["fps"], rtol=1e-6), actual_fps
    assert (scene.render.resolution_x, scene.render.resolution_y) == (
        manifest["width"],
        manifest["height"],
    )
    mesh_error = 0.0
    ball_error = 0.0
    projection_error = 0.0
    pc2 = {}
    for player in manifest["players"]:
        path = bundle / player["pc2"]
        header = np.fromfile(path, dtype=np.uint8, count=32)
        count = int(np.frombuffer(header[16:20].tobytes(), dtype="<i4")[0])
        pc2[player["index"]] = np.memmap(
            path, dtype="<f4", mode="r", offset=32, shape=(frames, count, 3)
        )
    points = np.array([[0, 0, 0], [3, 8, 1], [-3, -8, 1]], dtype=float)
    expected_uv = None
    visibility = {}
    if playback:
        assert camera.data.type == "PERSP" and camera.animation_data is None
        assert camera.data.sensor_fit == "HORIZONTAL"
        assert math.isclose(camera.data.lens, PLAYBACK_LENS_MM, rel_tol=1e-6)
        assert math.isclose(camera.data.sensor_width, PLAYBACK_SENSOR_WIDTH_MM)
        shot = Shot()
        np.testing.assert_allclose(camera.location, shot.move_end_location, atol=1e-6)
        target = world_to_camera_view(scene, camera, Vector(shot.move_end_target))
        np.testing.assert_allclose((target.x, target.y), (0.5, 0.5), atol=1e-6)
        assert all(image.source != "SEQUENCE" for image in bpy.data.images)
        visibility = _playback_visibility(manifest, frames)
    else:
        K = np.asarray(manifest["camera"]["opencv"]["K"])
        R = np.asarray(manifest["camera"]["opencv"]["R"])
        t = np.asarray(manifest["camera"]["opencv"]["t"])
        calibrated = points @ R.T + t
        expected_uv = calibrated[:, :2] / calibrated[:, 2:] @ K[:2, :2].T + K[:2, 2]
    initial_camera = None
    final_camera = None
    for frame in range(frames):
        scene.frame_set(frame)
        for name, active in visibility.items():
            obj = bpy.data.objects[name]
            hidden = not active[frame]
            assert obj.hide_render == obj.hide_viewport == hidden, (name, frame)
        for player in manifest["players"]:
            index = player["index"]
            obj = bpy.data.objects[f"Player{index}"]
            assert obj.hide_render == (not player_valid[index, frame]), (
                obj.name,
                frame,
            )
            # Hidden objects may retain the prior evaluated mesh in Blender.
            if player_valid[index, frame] and frame in samples:
                evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                vertices = np.empty((len(evaluated.data.vertices), 3), dtype=np.float32)
                evaluated.data.vertices.foreach_get("co", vertices.ravel())
                error = float(np.max(np.abs(vertices - pc2[index][frame])))
                mesh_error = max(mesh_error, error)
                assert error < 2e-5, (obj.name, frame, error)
        ball = bpy.data.objects["Ball"]
        assert ball.hide_render == (not ball_valid[frame]), ("Ball", frame)
        assert bpy.data.objects["BallTrail"].hide_render == (not ball_valid[frame]), (
            "BallTrail",
            frame,
        )
        if ball_valid[frame]:
            error = float(
                np.max(np.abs(np.asarray(ball.location) - ball_positions[frame]))
            )
            ball_error = max(ball_error, error)
            assert error < 2e-5, ("Ball", frame, error)
        # These frames precede the default reveal and camera movement.
        if expected_uv is not None and frame in (0, min(100, frames - 1)):
            actual_uv = []
            for point in points:
                projected = world_to_camera_view(scene, camera, Vector(point))
                actual_uv.append(
                    [
                        projected.x * manifest["width"],
                        (1 - projected.y) * manifest["height"],
                    ]
                )
            error = float(np.max(np.abs(np.asarray(actual_uv) - expected_uv)))
            projection_error = max(projection_error, error)
            assert error < 0.002, ("Camera", frame, error)
        if frame == 0:
            initial_camera = np.asarray(camera.matrix_world).copy()
        if frame == frames - 1:
            final_camera = np.asarray(camera.matrix_world).copy()
    moved = bool(not np.allclose(initial_camera, final_camera))
    if not playback and frames > 421:
        assert moved, "Camera did not leave the calibrated viewpoint"
    scene.frame_set(0)
    report = {
        "blender_version": bpy.app.version_string,
        "frames_checked": frames,
        "mesh_frames_checked": samples,
        "max_mesh_error_m": mesh_error,
        "max_ball_error_m": ball_error,
        "max_camera_projection_error_px": None if playback else projection_error,
        "visibility_matches_masks": True,
        "fps": actual_fps,
        "render_frame_step": scene.frame_step,
        "render_resolution_percentage": scene.render.resolution_percentage,
        "render_samples": scene.eevee.taa_render_samples,
        "camera_moved": moved,
    }
    if playback:
        report["in_play_events_checked"] = len(manifest["ball_events"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    report = validate(args.bundle)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))

"""Check evaluated concept geometry and camera continuity against exported data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ball_diffusion import verify_assets  # noqa: E402


def validate(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    verify_assets(bundle, manifest)
    scene = bpy.context.scene
    if (
        scene["concept_bundle_schema"] != manifest["schema"]
        or json.loads(scene["concept_origins"]) != manifest["origins"]
    ):
        raise AssertionError("Concept schema/provenance differs from bundle")
    if (
        scene.frame_start != 0
        or scene.frame_end != manifest["num_frames"] - 1
        or scene.render.fps / scene.render.fps_base != manifest["fps"]
    ):
        raise AssertionError("Output clock differs from bundle")
    maximum, checks = 0.0, 0

    def near(actual, expected, label: str) -> None:
        nonlocal maximum, checks
        error = float(np.max(np.abs(np.asarray(actual) - expected)))
        if not np.isfinite(error) or error > 1e-5:
            raise AssertionError(f"{label}: {error} > 1e-5")
        maximum = max(maximum, error)
        checks += 1

    process = np.load(bundle / "process.npy", mmap_mode="r")
    ball_positions = np.load(bundle / "ball_position.npy")
    ball_visible = np.load(bundle / "ball_visible.npy")
    curve_visible = np.load(bundle / "curve_visible.npy")
    centers = np.load(bundle / "tube_center_indices.npy")
    states = np.load(bundle / "condition_state.npy")
    positions = np.load(bundle / "condition_position.npy")
    trajectory = np.load(bundle / "trajectory.npy")
    continuity = manifest["camera_continuity"]
    snapshot = json.loads(scene["source_continuity"])
    near(
        snapshot["camera_matrix_world"],
        np.array(continuity["matrix_world"]),
        "Copied source camera",
    )
    if snapshot["source_blend_sha256"] != continuity["source_blend_sha256"]:
        raise AssertionError("Source blend identity differs")
    for index, state in enumerate(states):
        obj = bpy.data.objects[f"Condition_{index}"]
        near(
            obj.location,
            trajectory[index] if state == 0 else positions[index],
            f"Condition {index}",
        )
        if (
            obj.hide_render
            or obj["condition_state"] != int(state)
            or obj["origin"] != ("schematic" if state == 0 else "measured")
        ):
            raise AssertionError(f"Condition state/provenance {index}")
        if obj.data.materials[0].name != f"ConditionMaterial_{int(state)}":
            raise AssertionError(f"Condition color assignment {index}")
        if state == 0 and not any(mod.type == "WIREFRAME" for mod in obj.modifiers):
            raise AssertionError(f"Missing marker must be hollow: {index}")
    badge = bpy.data.objects["ConceptBadge"]
    if badge["label"] != "構想" or badge.hide_render or badge.parent != scene.camera:
        raise AssertionError("Missing camera-attached concept badge")
    for frame in range(manifest["num_frames"]):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        camera = scene.camera.evaluated_get(graph)
        near(
            np.array(camera.matrix_world),
            np.array(continuity["matrix_world"]),
            f"Camera {frame}",
        )
        if abs(camera.data.lens - continuity["lens_mm"]) > 1e-5:
            raise AssertionError(f"Camera lens {frame}")
        for cam in manifest["cameras"]:
            root = bpy.data.objects[f"CameraFrustum_{cam['id']}"]
            near(
                np.array(root.matrix_world),
                np.array(snapshot["frusta"][cam["id"]]),
                f"Frustum {cam['id']}/{frame}",
            )
        curve = bpy.data.objects["CompletionCurve"].evaluated_get(graph)
        if curve.hide_render != (not bool(curve_visible[frame])):
            raise AssertionError(f"Curve visibility at {frame}")
        vertices = np.empty((len(curve.data.vertices), 3), np.float32)
        curve.data.vertices.foreach_get("co", vertices.ravel())
        near(vertices[centers], process[frame], f"Process {frame}")
        ball = bpy.data.objects["CompletedBall"].evaluated_get(graph)
        if ball.hide_render != (not bool(ball_visible[frame])):
            raise AssertionError(f"Ball visibility at {frame}")
        near(ball.matrix_world.translation, ball_positions[frame], f"Ball {frame}")
    return {
        "ok": True,
        "frames_checked": manifest["num_frames"],
        "coordinate_checks": checks,
        "max_error_m": maximum,
        "origins": manifest["origins"],
        "stages": manifest["stages"],
    }


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate(args.bundle.resolve())
    except Exception as error:
        args.report.write_text(json.dumps({"ok": False, "error": str(error)}, indent=2))
        raise
    args.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__" and "--" in sys.argv:
    main(sys.argv[sys.argv.index("--") + 1 :])

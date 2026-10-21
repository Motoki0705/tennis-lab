"""Validate evaluated rays, source clocks and the calibrated 2D camera."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from ray_timeline import (  # noqa: E402
    RayShot,
    calibrated_pose,
    plan_timeline,
    select_camera,
    source_window,
)


def validate(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema"] != "tennis_scene_blender_rays_v1":
        raise ValueError("Unsupported bundle schema")
    scene = bpy.context.scene
    if scene["rays_bundle_schema"] != manifest["schema"]:
        raise ValueError("Saved scene is not a rays shot")
    shot = RayShot(**json.loads(scene["ray_shot"]))
    timeline = plan_timeline(manifest, shot)
    sources = np.array(scene["source_frames"])
    if not np.array_equal(sources, timeline.source_frames):
        raise AssertionError("Source clock disagrees with configured timeline")
    arrays = {
        name: np.load(bundle / f"{name}.npy", allow_pickle=False)
        for name in (
            "ray_origin",
            "ray_direction",
            "ray_length",
            "ray_state",
            "ball_position",
            "detection_uv",
        )
    }
    focus_view, selection = select_camera(
        manifest, arrays["detection_uv"], shot.camera_id
    )
    if json.loads(scene["camera_selection"]) != selection:
        raise AssertionError("Camera selection/visibility counts disagree with bundle")
    center, rotation, lens = calibrated_pose(manifest["cameras"][focus_view], shot)
    checks = material_checks = camera_checks = projection_checks = 0
    max_error = max_projection_error = 0.0
    states_seen: set[int] = set()
    material_samples = []
    checked_plates: set[Path] = set()
    pixel_tolerance = 0.01
    previous_camera: np.ndarray | None = None
    previous_lens: float | None = None

    def near(actual: Any, expected: Any, label: str) -> None:
        nonlocal max_error, checks
        error = float(np.linalg.norm(np.array(actual) - np.array(expected)))
        if not np.isfinite(error) or error > 1e-5:
            raise AssertionError(
                f"{label}: error={error}, expected={expected}, actual={actual}"
            )
        max_error = max(error, max_error)
        checks += 1

    def visible(obj: Any, expected: bool, label: str) -> None:
        if obj.hide_render != (not expected) or obj.hide_viewport != (not expected):
            raise AssertionError(f"{label} visibility")

    def projected(world: Any, expected: Any, camera: Any, label: str) -> None:
        nonlocal max_projection_error, projection_checks
        ndc = world_to_camera_view(scene, camera, Vector(world))
        focus_cam = manifest["cameras"][focus_view]
        actual = np.array(
            [ndc.x * focus_cam["width"], (1 - ndc.y) * focus_cam["height"]]
        )
        error = float(np.linalg.norm(actual - expected))
        if not np.isfinite(error) or error > pixel_tolerance or ndc.z <= 0:
            raise AssertionError(f"{label}: projection error={error}px")
        max_projection_error = max(error, max_projection_error)
        projection_checks += 1

    for frame, source in enumerate(sources):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        t = int(source - manifest["frames"][0])
        in_2d = frame < timeline.pullback_start
        in_3d = frame >= timeline.playback_start
        camera = scene.camera.evaluated_get(depsgraph)
        if frame in (timeline.pullback_start, timeline.playback_start) or in_3d:
            near(
                camera.matrix_world, previous_camera, f"Camera stage continuity {frame}"
            )
            if abs(camera.data.lens - previous_lens) > 1e-5:
                raise AssertionError(f"Camera lens stage continuity {frame}")
        previous_camera = np.array(camera.matrix_world)
        previous_lens = camera.data.lens
        if in_2d:
            near(camera.matrix_world.translation, center, f"2D camera center {frame}")
            near(
                np.array(camera.matrix_world)[:3, :3],
                rotation,
                f"2D camera orientation {frame}",
            )
            if (
                camera.data.sensor_fit != "HORIZONTAL"
                or abs(camera.data.sensor_width - 36.0) > 1e-6
                or abs(camera.data.lens - lens) > 1e-5
                or camera.data.shift_x != 0
                or camera.data.shift_y != 0
                or scene.render.pixel_aspect_x != scene.render.pixel_aspect_y
            ):
                raise AssertionError(f"2D camera intrinsics at {frame}")
            camera_checks += 1
        point = arrays["ball_position"][t]
        valid = bool(np.isfinite(point).all())
        ball = bpy.data.objects["Ball"].evaluated_get(depsgraph)
        visible(ball, valid and in_3d, f"Ball at {frame}")
        if valid and in_3d:
            near(ball.matrix_world.translation, point, f"Ball at {frame}")
        visible(
            bpy.data.objects["BallTrail"].evaluated_get(depsgraph),
            valid and in_3d,
            f"Trail at {frame}",
        )
        for view, cam in enumerate(manifest["cameras"]):
            root = bpy.data.objects[f"CameraFrustum_{cam['id']}"].evaluated_get(
                depsgraph
            )
            near(
                root.matrix_world.translation,
                arrays["ray_origin"][view],
                f"Frustum center {view}/{frame}",
            )
            near(
                np.array(root.matrix_world)[:3, :3],
                np.array(cam["R"]).T,
                f"Frustum orientation {view}/{frame}",
            )
            plane = bpy.data.objects[f"ImagePlane_{cam['id']}"].evaluated_get(depsgraph)
            visible(plane, view == focus_view or not in_2d, f"Plate {view}/{frame}")
            visible(
                bpy.data.objects[f"FrustumLines_{cam['id']}"].evaluated_get(depsgraph),
                not in_2d,
                f"Frustum {view}/{frame}",
            )
            pixels = [
                (0, 0),
                (cam["width"], 0),
                (cam["width"], cam["height"]),
                (0, cam["height"]),
            ]
            for vertex, pixel in zip(plane.data.vertices, pixels, strict=True):
                expected_local = (
                    np.linalg.solve(np.array(cam["K"]), [*pixel, 1.0])
                    * shot.frustum_depth_m
                )
                near(vertex.co, expected_local, f"Image plane {view}/{frame}")
                if in_2d and view == focus_view:
                    projected(
                        plane.matrix_world @ vertex.co,
                        pixel,
                        camera,
                        f"Full-screen plate {frame}",
                    )
            state = int(arrays["ray_state"][view, t])
            states_seen.add(state)
            ray = bpy.data.objects[f"Ray_{view}"].evaluated_get(depsgraph)
            if ray["geometry_representation"] != "midpoint_relative_pc2":
                raise ValueError("Unsupported ray geometry representation")
            visible(ray, state != 0 and in_3d, f"Ray {view}/{frame}")
            # Blender omits viewport-hidden objects from cache/animation
            # evaluation. Inspect their visibility, and inspect evaluated
            # geometry/custom properties whenever they actually appear.
            if state != 0 and in_3d and ray["ray_state"] != state:
                raise AssertionError(f"Ray {view} state at {frame}")
            world = np.array(ray.matrix_world, dtype=np.float64)
            base = np.array(ray.data.vertices[ray["base_vertex"]].co, dtype=np.float64)
            tip = np.array(ray.data.vertices[ray["tip_vertex"]].co, dtype=np.float64)
            base_world = world[:3, :3] @ base + world[:3, 3]
            tip_world = world[:3, :3] @ tip + world[:3, 3]
            if state != 0 and in_3d:
                # Full length on every visible frame, with no growth stage.
                expected = (
                    arrays["ray_origin"][view]
                    + arrays["ray_direction"][view, t] * arrays["ray_length"][view, t]
                )
                near(
                    base_world, arrays["ray_origin"][view], f"Ray origin {view}/{frame}"
                )
                near(tip_world, expected, f"Ray endpoint {view}/{frame}")
            material = ray.active_material
            if len(ray.data.materials) != 1 or material.name != f"RayMaterial_{view}":
                raise AssertionError(f"Ray {view} material assignment at {frame}")
            if (
                material.surface_render_method != "BLENDED"
                or material.use_transparency_overlap
            ):
                raise AssertionError(f"Ray {view} transparency method at {frame}")
            nodes = material.node_tree.nodes
            dashed = float(nodes["Dashed"].outputs[0].default_value)
            opacity = float(nodes["Opacity"].outputs[0].default_value)
            color = tuple(nodes["Color"].outputs[0].default_value)
            expected_hex = (
                "#E63946"
                if state == 2
                else ("#6C757D" if state == 4 else shot.camera_colors[view])
            )
            expected_color = tuple(
                int(expected_hex[i : i + 2], 16) / 255 for i in (1, 3, 5)
            ) + (1.0,)
            if (
                dashed != float(state == 3)
                or abs(opacity - (0.6 if state == 2 else 1.0)) > 1e-6
                or not np.allclose(color, expected_color, atol=1e-6, rtol=0)
            ):
                raise AssertionError(
                    f"Ray {view} material state at {frame}: state={state}, dashed={dashed}, opacity={opacity}, color={color}"
                )
            material_checks += 1
            if in_3d and frame in (
                timeline.playback_start,
                (timeline.playback_start + len(sources) - 1) // 2,
                len(sources) - 1,
            ):
                camera_point = camera.matrix_world.inverted() @ Vector(
                    (base_world + tip_world) * 0.5
                )
                focal_px = (
                    camera.data.lens
                    / camera.data.sensor_width
                    * scene.render.resolution_x
                    * scene.render.resolution_percentage
                    / 100
                )
                material_samples.append(
                    {
                        "frame": frame,
                        "source_frame": int(source),
                        "view": view,
                        "state": state,
                        "dashed": dashed,
                        "opacity": opacity,
                        "surface_render_method": material.surface_render_method,
                        "length_m": float(np.linalg.norm(tip_world - base_world)),
                        "projected_diameter_px_approx": 2
                        * shot.ray_radius_m
                        * focal_px
                        / (-camera_point.z)
                        if camera_point.z < 0
                        else None,
                    }
                )
            ring = bpy.data.objects[f"Detection_{view}"].evaluated_get(depsgraph)
            uv = arrays["detection_uv"][view, t]
            ring_visible = bool(np.isfinite(uv).all()) and (
                view == focus_view or not in_2d
            )
            visible(ring, ring_visible, f"Detection {view}/{frame}")
            if ring_visible:
                local = np.linalg.solve(np.array(cam["K"]), [*uv, 1.0]) * (
                    shot.frustum_depth_m - 0.005
                )
                expected = np.array(cam["center"]) + np.array(cam["R"]).T @ local
                near(
                    ring.matrix_world.translation, expected, f"Detection {view}/{frame}"
                )
                if in_2d:
                    projected(
                        ring.matrix_world.translation,
                        uv,
                        camera,
                        f"2D ring {view}/{frame}",
                    )
            texture = bpy.data.materials[f"Plate_{view}"].node_tree.nodes[
                "PlateSequence"
            ]
            if texture.image_user.frame_offset != int(source) - frame - 1:
                raise AssertionError(f"Plate frame offset {view}/{frame}")
            # Resolve ImageUser explicitly: headless validation does not refresh
            # textures on the GPU, but uses the same absolute filename clock.
            user = texture.image_user
            user.frame_current = frame - user.frame_start + user.frame_offset + 1
            actual_path = Path(texture.image.filepath_from_user(image_user=user))
            expected_path = bundle / cam["plate_dir"] / f"{source:05d}.jpg"
            if actual_path != expected_path or (
                actual_path not in checked_plates and not actual_path.is_file()
            ):
                raise AssertionError(
                    f"Plate image {view}/{frame}: {actual_path}, expected {expected_path}"
                )
            checked_plates.add(actual_path)
    if scene.render.fps / scene.render.fps_base != shot.fps:
        raise AssertionError("Render fps disagrees with shot clock")
    return {
        "ok": True,
        "frames_checked": len(sources),
        "coordinate_checks": checks,
        "max_error_m": max_error,
        "tolerance_m": 1e-5,
        "two_d_camera_checks": camera_checks,
        "projection_checks": projection_checks,
        "max_projection_error_px": max_projection_error,
        "projection_tolerance_px": pixel_tolerance,
        "camera_selection": selection,
        "time_mode": shot.time_mode,
        "source_window": source_window(manifest["fps"], shot),
        "stages": [s.to_dict(shot.fps, manifest["fps"]) for s in timeline.stages],
        "ray_states_seen": sorted(states_seen),
        "material_checks": material_checks,
        "ray_geometry": "midpoint_relative_pc2",
        "ray_material_surface": "BLENDED",
        "render_samples": int(scene.eevee.taa_render_samples),
        "material_samples": material_samples,
        "ball_source": manifest["ball_source"],
    }


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate(args.bundle)
    except BaseException as error:
        args.report.write_text(json.dumps({"ok": False, "error": str(error)}, indent=2))
        raise
    args.report.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "ball_source"},
            indent=2,
        )
    )


if __name__ == "__main__" and "--" in sys.argv:
    main(sys.argv[sys.argv.index("--") + 1 :])

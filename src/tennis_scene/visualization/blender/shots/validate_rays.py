"""Validate evaluated Blender ray endpoints, balls and masks against a bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Vector


def validate(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema"] != "tennis_scene_blender_rays_v1":
        raise ValueError("Unsupported bundle schema")
    scene = bpy.context.scene
    if scene["rays_bundle_schema"] != manifest["schema"]:
        raise ValueError("Saved scene is not a rays shot")
    shot = json.loads(scene["ray_shot"])
    sources = np.array(scene["source_frames"])
    arrays = {
        name: np.load(bundle / f"{name}.npy", allow_pickle=False)
        for name in (
            "ray_origin",
            "ray_direction",
            "ray_length",
            "ray_state",
            "ball_position",
            "detection_uv",
            "reprojection_uv",
            "closest_point",
        )
    }
    max_error = 0.0
    checks = 0
    states_seen = set()
    material_checks = 0
    growth_material_samples = []
    checked_plates: set[Path] = set()

    def near(actual: Any, expected: Any, label: str) -> None:
        nonlocal max_error, checks
        error = float(np.linalg.norm(np.array(actual) - np.array(expected)))
        if not np.isfinite(error) or error > 1e-5:
            raise AssertionError(
                f"{label}: error={error}m, expected={expected}, actual={actual}"
            )
        max_error = max(error, max_error)
        checks += 1

    for view, cam in enumerate(manifest["cameras"]):
        root = bpy.data.objects[f"CameraFrustum_{cam['id']}"]
        near(
            root.matrix_world.translation, arrays["ray_origin"][view], f"camera {view}"
        )
    for frame, source in enumerate(sources):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        t = int(source - manifest["frames"][0])
        point = arrays["ball_position"][t]
        valid = bool(np.isfinite(point).all())
        ball = bpy.data.objects["Ball"].evaluated_get(depsgraph)
        ball_visible = valid and frame >= shot["ball_start"]
        if ball.hide_render != (not ball_visible) or ball.hide_viewport != (
            not ball_visible
        ):
            raise AssertionError(f"Ball visibility at frame {frame}")
        if ball_visible:
            near(ball.matrix_world.translation, point, f"ball at frame {frame}")
        trail = bpy.data.objects["BallTrail"].evaluated_get(depsgraph)
        if trail.hide_render != (not (valid and frame >= shot["playback_start"])):
            raise AssertionError(f"Trail visibility at frame {frame}")
        for view, cam in enumerate(manifest["cameras"]):
            state = int(arrays["ray_state"][view, t])
            states_seen.add(state)
            ray = bpy.data.objects[f"Ray_{view}"].evaluated_get(depsgraph)
            visible = state != 0 and frame >= shot["rays_start"]
            if ray.hide_render != (not visible) or ray.hide_viewport != (not visible):
                raise AssertionError(f"Ray {view} visibility at frame {frame}")
            if ray["ray_state"] != state:
                raise AssertionError(f"Ray {view} state at frame {frame}")
            material = ray.active_material
            if len(ray.data.materials) != 1 or material.name != f"RayMaterial_{view}":
                raise AssertionError(f"Ray {view} material assignment at frame {frame}")
            nodes = material.node_tree.nodes
            dashed = float(nodes["Dashed"].outputs[0].default_value)
            opacity = float(nodes["Opacity"].outputs[0].default_value)
            color = tuple(nodes["Color"].outputs[0].default_value)
            expected_hex = (
                "#E63946"
                if state == 2
                else ("#6C757D" if state == 4 else shot["camera_colors"][view])
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
                    f"Ray {view} material state at frame {frame}: state={state}, dashed={dashed}, opacity={opacity}, color={color}"
                )
            material_checks += 1
            if frame in (
                shot["rays_start"],
                (shot["rays_start"] + shot["ball_start"]) // 2,
                shot["ball_start"] - 1,
                shot["ball_start"],
            ):
                midpoint = ray.matrix_world @ Vector((0.0, 0.0, 0.5))
                camera = scene.camera.evaluated_get(depsgraph)
                camera_point = camera.matrix_world.inverted() @ midpoint
                focal_px = (
                    camera.data.lens
                    / camera.data.sensor_width
                    * scene.render.resolution_x
                    * scene.render.resolution_percentage
                    / 100
                )
                growth_material_samples.append(
                    {
                        "frame": frame,
                        "source_frame": int(source),
                        "view": view,
                        "material": material.name,
                        "state": state,
                        "dashed": dashed,
                        "opacity": opacity,
                        "color": color,
                        "length_m": float(ray.scale.z),
                        "projected_diameter_px_approx": 2
                        * shot["ray_radius_m"]
                        * focal_px
                        / (-camera_point.z)
                        if camera_point.z < 0
                        else None,
                    }
                )
            if visible:
                growth = float(
                    np.clip(
                        (frame - shot["rays_start"])
                        / (shot["ball_start"] - shot["rays_start"]),
                        0.0,
                        1.0,
                    )
                )
                expected = (
                    arrays["ray_origin"][view]
                    + arrays["ray_direction"][view, t]
                    * arrays["ray_length"][view, t]
                    * growth
                )
                near(
                    ray.matrix_world @ Vector((0.0, 0.0, 0.0)),
                    arrays["ray_origin"][view],
                    f"ray origin {view}/{frame}",
                )
                near(
                    ray.matrix_world @ Vector((0.0, 0.0, 1.0)),
                    expected,
                    f"ray endpoint {view}/{frame}",
                )
            for label, array_name, visible_from in (
                ("Detection", "detection_uv", shot["plate_start"]),
                ("Reprojection", "reprojection_uv", shot["zoom_start"]),
            ):
                obj = bpy.data.objects[f"{label}_{view}"].evaluated_get(depsgraph)
                uv = arrays[array_name][view, t]
                expected_visible = bool(np.isfinite(uv).all()) and frame >= visible_from
                if obj.hide_render != (not expected_visible):
                    raise AssertionError(f"{label} visibility at {view}/{frame}")
                if expected_visible:
                    # Independent pixel-to-image-plane check, using original
                    # image pixels rather than the downscaled plate dimensions.
                    local = (
                        np.linalg.solve(np.array(cam["K"]), [*uv, 1.0])
                        * shot["frustum_depth_m"]
                    )
                    local[2] += 0.005 if label == "Detection" else 0.008
                    expected = np.array(cam["center"]) + np.array(cam["R"]).T @ local
                    near(
                        obj.matrix_world.translation,
                        expected,
                        f"{label} position {view}/{frame}",
                    )
            texture = bpy.data.materials[f"Plate_{view}"].node_tree.nodes[
                "PlateSequence"
            ]
            if texture.image_user.frame_offset != int(source) - frame - 1:
                raise AssertionError(f"Plate frame offset {view}/{frame}")
            # In a headless validation pass Blender does not refresh unused GPU
            # textures. Resolve the animated ImageUser explicitly (same absolute
            # filename convention as reveal), before checking the physical JPEG.
            user = texture.image_user
            user.frame_current = frame - user.frame_start + user.frame_offset + 1
            actual_path = Path(
                texture.image.filepath_from_user(image_user=texture.image_user)
            )
            expected_path = bundle / cam["plate_dir"] / f"{source:05d}.jpg"
            if actual_path != expected_path or (
                actual_path not in checked_plates and not actual_path.is_file()
            ):
                raise AssertionError(
                    f"Plate image at {view}/{frame}: {actual_path}, expected {expected_path}"
                )
            checked_plates.add(actual_path)
    if scene.render.fps / scene.render.fps_base != 30.0:
        raise AssertionError("Shot must render at 30fps")
    return {
        "ok": True,
        "frames_checked": len(sources),
        "coordinate_checks": checks,
        "max_error_m": max_error,
        "tolerance_m": 1e-5,
        "ray_states_seen": sorted(states_seen),
        "material_checks": material_checks,
        "growth_material_samples": growth_material_samples,
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

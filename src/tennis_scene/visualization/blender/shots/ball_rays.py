"""Display an exported rays bundle with Blender 5.2 (no repository imports).

blender -b --factory-startup --python-exit-code 1 --python ball_rays.py -- \
    --bundle /path/to/bundle --output /new/shot --render
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).parent))
from camera_planes import camera_frustum, image_plane_point  # noqa: E402
from common import (  # noqa: E402
    _build_compositor,
    _key_visibility,
    _look_rotation,
    _net_geometry,
    _new_material,
    _write_pc2,
    ball_shader,
    court_image,
    net_shader,
)
from ray_timeline import (  # noqa: E402
    RayShot,
    Timeline,
    calibrated_pose,
    plan_timeline,
    select_camera,
)


def _rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def _object(name: str, vertices: Any, faces: Any) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _material(
    name: str, color: tuple[float, float, float], *, strength: float = 1.0
) -> bpy.types.Material:
    material, n, surface = _new_material(name, transparent=True)
    color_node = n.new("ShaderNodeRGB")
    color_node.name = "Color"
    color_node.outputs[0].default_value = (*color, 1.0)
    opacity = n.new("ShaderNodeValue")
    opacity.name = "Opacity"
    opacity.outputs[0].default_value = 1.0
    dash = n.new("ShaderNodeValue")
    dash.name = "Dashed"
    dash.outputs[0].default_value = 0.0
    # The longitudinal coordinate survives the ray's Mesh Cache deformation.
    axis = n.new("ShaderNodeAttribute")
    axis.attribute_name = "ray_axis"
    stripe = n.math(
        "LESS_THAN", n.math("FRACT", n.math("MULTIPLY", axis.outputs["Fac"], 64.0)), 0.6
    )
    mask = n.math(
        "SUBTRACT",
        1.0,
        n.math("MULTIPLY", dash.outputs[0], n.math("SUBTRACT", 1.0, stripe)),
    )
    look = n.emission(color_node.outputs[0], strength)
    n.set(
        surface,
        n.mix_shader(
            n.math("MULTIPLY", opacity.outputs[0], mask),
            n.new("ShaderNodeBsdfTransparent").outputs[0],
            look,
        ),
    )
    return material


def _key_value(material: bpy.types.Material, name: str, value: Any, frame: int) -> None:
    socket = material.node_tree.nodes[name].outputs[0]
    socket.default_value = value
    socket.keyframe_insert("default_value", frame=frame)


def _cylinder(
    name: str, radius: float, material: bpy.types.Material
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=1.0)
    obj = bpy.context.object
    obj.name = name
    # A unit cylinder whose local endpoints are exactly (0,0,0) and (0,0,1).
    for vertex in obj.data.vertices:
        vertex.co.z += 0.5
    obj.data.materials.append(material)
    obj.rotation_mode = "QUATERNION"
    return obj


def _ray_mesh(
    name: str, radius: float, material: bpy.types.Material
) -> tuple[bpy.types.Object, np.ndarray]:
    """Cylinder with explicit cap centers for evaluated endpoint checks."""
    angles = np.arange(12) * (2 * math.pi / 12)
    ring = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.zeros(12))
    )
    unit = np.concatenate(
        (ring, ring + [0.0, 0.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    faces = []
    for i in range(12):
        j = (i + 1) % 12
        faces.extend(((i, j, j + 12, i + 12), (24, j, i), (25, i + 12, j + 12)))
    obj = _object(name, unit.tolist(), faces)
    obj.data.materials.append(material)
    attr = obj.data.attributes.new("ray_axis", "FLOAT", "POINT")
    attr.data.foreach_set("value", unit[:, 2].astype(np.float32))
    obj["base_vertex"], obj["tip_vertex"] = 24, 25
    obj["geometry_representation"] = "midpoint_relative_pc2"
    return obj, unit


def _ray_cache_frame(
    unit: np.ndarray, origin: np.ndarray, direction: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Display exported ray geometry around its midpoint to retain precision.

    Long rejected rays can meet the ground hundreds of meters away. An object
    quaternion loses their tiny vertical component. Midpoint-relative cached
    vertices avoid that cancellation and keep the supplied origin/end intact.
    """
    side = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
    side /= np.linalg.norm(side)
    up = np.cross(direction, side)
    points = (
        origin
        + unit[:, :1] * side
        + unit[:, 1:2] * up
        + unit[:, 2:] * direction * length
    )
    # Subtract the actual float32 location that Blender will store.
    anchor = np.asarray(origin + direction * length * 0.5, np.float32)
    return anchor, np.asarray(points - anchor.astype(np.float64), np.float32)


def _court(court: dict[str, Any]) -> None:
    xmin, xmax, ymin, ymax = court["runoff_xy"]
    ground = _object(
        "Court",
        [(xmin, ymin, 0), (xmax, ymin, 0), (xmax, ymax, 0), (xmin, ymax, 0)],
        [(0, 1, 2, 3)],
    )
    uv = ground.data.uv_layers.new()
    for loop, value in zip(uv.data, [(0, 0), (1, 0), (1, 1), (0, 1)], strict=True):
        loop.uv = value
    mat, n, surface = _new_material("Court", transparent=False)
    texture = n.new("ShaderNodeTexImage")
    texture.image = court_image(
        court, court_color=(0.025, 0.11, 0.13), runoff_color=(0.025, 0.035, 0.04)
    )
    n.set(surface, n.emission(texture.outputs["Color"], 0.8))
    ground.data.materials.append(mat)
    verts, faces, tape = _net_geometry(court)
    net = _object("Net", verts.tolist(), faces.tolist())
    attr = net.data.attributes.new("tape", "FLOAT", "POINT")
    attr.data.foreach_set("value", tape.astype(np.float32))
    mat, n, surface = _new_material("Net", transparent=True)
    n.set(surface, net_shader(n))
    net.data.materials.append(mat)
    posts = _material("NetPosts", (0.8, 0.8, 0.8))
    for side in (-1, 1):
        post = _cylinder(f"NetPost{side}", 0.04, posts)
        post.location = (side * court["net"]["post_x"], 0, 0)
        post.scale.z = court["net"]["height_post"]


def _marker(name: str, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_segments=24, minor_segments=6, major_radius=0.022, minor_radius=0.003
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def _build_view(
    bundle: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    view: int,
    timeline: Timeline,
    focus_view: int,
    shot: RayShot,
    output: Path,
) -> None:
    sources = timeline.source_frames
    cam = manifest["cameras"][view]
    color = _rgb(shot.camera_colors[view])
    root, plane = camera_frustum(cam, color=color, depth=shot.frustum_depth_m)
    reveal = np.arange(len(sources)) >= timeline.pullback_start
    _key_visibility(bpy.data.objects[f"FrustumLines_{cam['id']}"], reveal)
    _key_visibility(plane, reveal | (view == focus_view))
    plate_mat, n, surface = _new_material(f"Plate_{view}", transparent=False)
    texture = n.new("ShaderNodeTexImage")
    texture.name = "PlateSequence"
    texture.image = bpy.data.images.load(
        str(bundle / cam["plate_dir"] / f"{manifest['frames'][0]:05d}.jpg")
    )
    texture.image.source = "SEQUENCE"
    texture.image.colorspace_settings.name = "sRGB"
    texture.image_user.frame_duration = max(len(sources), manifest["frames"][1])
    texture.image_user.frame_start = 0
    texture.image_user.use_auto_refresh = True
    n.set(surface, n.emission(texture.outputs["Color"]))
    plane.data.materials.clear()
    plane.data.materials.append(plate_mat)
    ring_mat = _material(f"DetectionMaterial_{view}", color, strength=2.0)
    ring = _marker(f"Detection_{view}", ring_mat)
    ring.parent = root
    ray_mat = _material(f"RayMaterial_{view}", color, strength=2.0)
    ray, unit = _ray_mesh(f"Ray_{view}", shot.ray_radius_m, ray_mat)
    cache = np.zeros((len(sources), len(unit), 3), np.float32)
    ray_valid, ring_valid = [], []
    for frame, src in enumerate(sources):
        t = int(src - manifest["frames"][0])
        state = int(arrays["ray_state"][view, t])
        exists = state != 0
        ray_valid.append(exists and frame >= timeline.playback_start)
        ring_valid.append(exists and (view == focus_view or reveal[frame]))
        # Absolute numeric filename, with no fade or frozen t0 intro.
        texture.image_user.frame_offset = int(src) - frame - 1
        texture.image_user.keyframe_insert("frame_offset", frame=frame)
        if exists:
            anchor, cache[frame] = _ray_cache_frame(
                unit,
                arrays["ray_origin"][view],
                arrays["ray_direction"][view, t],
                float(arrays["ray_length"][view, t]),
            )
            ray.location = anchor.tolist()
            ray.keyframe_insert("location", frame=frame)
            # Offset towards the real camera along the same pixel ray. An
            # additive z offset shifts the projected uv and hides it behind
            # the plate when looking from the calibrated camera center.
            ring.location = image_plane_point(
                cam, arrays["detection_uv"][view, t], depth=shot.frustum_depth_m - 0.005
            )
            ring.keyframe_insert("location", frame=frame)
        ray["ray_state"] = state
        ray.keyframe_insert('["ray_state"]', frame=frame)
        state_color = (
            _rgb("#E63946")
            if state == 2
            else (_rgb("#6C757D") if state == 4 else color)
        )
        _key_value(ray_mat, "Color", (*state_color, 1.0), frame)
        _key_value(ray_mat, "Opacity", 0.6 if state == 2 else 1.0, frame)
        _key_value(ray_mat, "Dashed", float(state == 3), frame)
    cache_path = output / f"ray_{view}.pc2"
    _write_pc2(cache_path, cache)
    modifier = ray.modifiers.new("RayMotion", "MESH_CACHE")
    modifier.cache_format, modifier.filepath, modifier.frame_start = (
        "PC2",
        str(cache_path),
        0.0,
    )
    _key_visibility(ray, np.array(ray_valid))
    _key_visibility(ring, np.array(ring_valid))


def _ball(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    timeline: Timeline,
    shot: RayShot,
    output: Path,
) -> None:
    source_frames = timeline.source_frames
    positions = arrays["ball_position"]
    valid = np.isfinite(positions).all(axis=1)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=shot.ball_radius_m, segments=24, ring_count=12
    )
    ball = bpy.context.object
    ball.name = "Ball"
    material = _material("BallMaterial", (0.85, 1.0, 0.08), strength=3.0)
    ball.data.materials.append(material)
    timeline_valid = np.array(
        [
            frame >= timeline.playback_start and valid[src - manifest["frames"][0]]
            for frame, src in enumerate(source_frames)
        ]
    )
    for frame, src in enumerate(source_frames):
        if timeline_valid[frame]:
            ball.location = positions[src - manifest["frames"][0]].tolist()
            ball.keyframe_insert("location", frame=frame)
    _key_visibility(ball, timeline_valid)
    # Same shrinking-sphere PC2 trail as reveal: 16 points, stride two source
    # frames, and every intervening source mask must be valid.
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, subdivisions=2)
    template = bpy.context.object
    unit = np.array([v.co[:] for v in template.data.vertices])
    tri = np.array([p.vertices[:] for p in template.data.polygons])
    bpy.data.objects.remove(template, do_unlink=True)
    count, per = shot.trail_count, len(unit)
    scales = shot.ball_radius_m * 0.8 * (1 - np.arange(count) / count)
    frames = np.zeros((len(source_frames), count * per, 3), np.float32)
    for frame, src in enumerate(source_frames):
        t = int(src - manifest["frames"][0])
        for k in range(count):
            past = t - (k + 1) * shot.trail_stride
            ok = (
                frame >= timeline.playback_start
                and past >= 0
                and valid[past : t + 1].all()
            )
            center = (
                positions[past] if ok else (positions[t] if valid[t] else np.zeros(3))
            )
            frames[frame, k * per : (k + 1) * per] = center + unit * (
                scales[k] if ok else 0.0
            )
    cache = output / "ball_trail.pc2"
    _write_pc2(cache, frames)
    trail = _object(
        "BallTrail",
        frames[0].tolist(),
        np.concatenate([tri + k * per for k in range(count)]).tolist(),
    )
    mat, n, surface = _new_material("TrailMaterial", transparent=True)
    n.set(surface, ball_shader(n, (0.85, 1.0, 0.08), fade_attribute="age"))
    trail.data.materials.append(mat)
    attr = trail.data.attributes.new("age", "FLOAT", "POINT")
    attr.data.foreach_set(
        "value", (np.repeat(1 - np.arange(count) / count, per) * 0.8).astype(np.float32)
    )
    mod = trail.modifiers.new("Trail", "MESH_CACHE")
    mod.cache_format, mod.filepath, mod.frame_start = "PC2", str(cache), 0.0
    _key_visibility(trail, timeline_valid)


def _camera(
    manifest: dict[str, Any],
    timeline: Timeline,
    arrays: dict[str, np.ndarray],
    focus_view: int,
    shot: RayShot,
) -> None:
    data = bpy.data.cameras.new("ShotCamera")
    camera = bpy.data.objects.new("ShotCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    data.sensor_fit, data.sensor_width = "HORIZONTAL", 36.0
    data.clip_start, data.clip_end = 0.05, 1000.0
    center, rotation, lens = calibrated_pose(manifest["cameras"][focus_view], shot)
    start_location = Vector(center)
    start_rotation = Matrix(rotation).to_quaternion()
    radius = max(
        45.0, float(np.linalg.norm(arrays["ray_origin"][:, :2], axis=1).max()) + 30.0
    )
    # Pull backwards from the selected camera's side of the court. The plate
    # remains the base of its frustum; no cut, orbit or reprojection zoom.
    angle = math.atan2(float(center[1]), float(center[0]))
    overview = Vector(
        (radius * math.cos(angle), radius * math.sin(angle), radius * 0.7)
    )
    focus = Vector((0.0, 0.0, 1.0))
    end_rotation = _look_rotation(focus - overview, 0.0)
    for frame in range(len(timeline.source_frames)):
        t = float(
            np.clip(
                (frame - timeline.pullback_start)
                / (timeline.playback_start - timeline.pullback_start - 1),
                0.0,
                1.0,
            )
        )
        t = t * t * (3.0 - 2.0 * t)
        camera.location = start_location.lerp(overview, t)
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = start_rotation.slerp(end_rotation, t)
        data.lens = lens + (32.0 - lens) * t
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)
        data.keyframe_insert("lens", frame=frame)


def _constant_keys() -> None:
    # Shader, sequence, visibility and source geometry must switch together;
    # missing observations are never interpolated into the next source frame.
    camera = bpy.context.scene.camera
    camera_actions = (camera.animation_data.action, camera.data.animation_data.action)
    for action in bpy.data.actions:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for point in curve.keyframe_points:
                            point.interpolation = (
                                "LINEAR" if action in camera_actions else "CONSTANT"
                            )


def build(bundle: Path, output: Path, shot: RayShot) -> bpy.types.Scene:
    if output.exists():
        raise FileExistsError(f"Shot output already exists: {output}")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema"] != "tennis_scene_blender_rays_v1":
        raise ValueError("Unsupported rays bundle schema")
    if len(manifest["cameras"]) > len(shot.camera_colors):
        raise ValueError("RayShot.camera_colors does not cover every camera")
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
    timeline = plan_timeline(manifest, shot)
    sources = timeline.source_frames
    focus_view, camera_selection = select_camera(
        manifest, arrays["detection_uv"], shot.camera_id
    )
    calibrated_pose(manifest["cameras"][focus_view], shot)
    output.mkdir(parents=True)
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = shot.samples
        scene.render.fps, scene.render.fps_base = shot.fps, 1.0
        scene.render.resolution_x, scene.render.resolution_y = shot.width, shot.height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.frame_start, scene.frame_end = 0, len(sources) - 1
        world = bpy.data.worlds.new("RaysWorld")
        scene.world = world
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (
            0.006,
            0.006,
            0.006,
            1.0,
        )
        sun_data = bpy.data.lights.new("Sun", "SUN")
        sun_data.energy = 2.0
        sun = bpy.data.objects.new("Sun", sun_data)
        scene.collection.objects.link(sun)
        sun.rotation_euler = (math.radians(35.0), 0.0, math.radians(-140.0))
        _court(manifest["court"])
        for view in range(len(manifest["cameras"])):
            _build_view(
                bundle, manifest, arrays, view, timeline, focus_view, shot, output
            )
        _ball(arrays, manifest, timeline, shot, output)
        _camera(manifest, timeline, arrays, focus_view, shot)
        _constant_keys()
        _build_compositor(scene, name="RaysComposite")
        scene["rays_bundle_schema"] = manifest["schema"]
        scene["source_frames"] = sources.tolist()
        scene["ray_shot"] = json.dumps(asdict(shot))
        scene["camera_selection"] = json.dumps(camera_selection)
        (output / "shot_manifest.json").write_text(
            json.dumps(
                {
                    "shot": asdict(shot),
                    "source_frames": sources.tolist(),
                    "source_sampling": "nearest timestamp per stage; no gap filling",
                    "camera_selection": camera_selection,
                    "stages": [
                        s.to_dict(shot.fps, manifest["fps"]) for s in timeline.stages
                    ],
                    "bundle": str(bundle),
                    "ball_source": manifest["ball_source"],
                },
                indent=2,
            )
        )
        scene.frame_set(0)
    except BaseException:
        shutil.rmtree(output)
        raise
    return scene


def main(argv: list[str]) -> None:
    defaults = RayShot()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--time-mode", choices=("continuous", "replay"), default=defaults.time_mode
    )
    parser.add_argument(
        "--camera-id", help="default: most visible frames, then camera ID"
    )
    parser.add_argument("--two-d-seconds", type=float, help="default: native speed")
    parser.add_argument(
        "--pullback-seconds", type=float, default=defaults.pullback_seconds
    )
    parser.add_argument("--playback-seconds", type=float, help="default: native speed")
    parser.add_argument(
        "--continuous-two-d-fraction",
        type=float,
        default=defaults.continuous_two_d_fraction,
    )
    parser.add_argument(
        "--continuous-pullback-fraction",
        type=float,
        default=defaults.continuous_pullback_fraction,
    )
    parser.add_argument("--frames", help="start:end, inclusive")
    parser.add_argument("--samples", type=int, default=defaults.samples)
    parser.add_argument("--resolution-percentage", type=int, default=100)
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.resolution_percentage <= 100
        or args.samples < 1
        or args.frame_step < 1
    ):
        parser.error("Require resolution 1..100, positive samples/frame-step")
    scene = build(
        args.bundle.resolve(),
        args.output.resolve(),
        RayShot(
            samples=args.samples,
            camera_id=args.camera_id,
            time_mode=args.time_mode,
            two_d_seconds=args.two_d_seconds,
            pullback_seconds=args.pullback_seconds,
            playback_seconds=args.playback_seconds,
            continuous_two_d_fraction=args.continuous_two_d_fraction,
            continuous_pullback_fraction=args.continuous_pullback_fraction,
        ),
    )
    if args.frames:
        start, end = map(int, args.frames.split(":"))
        if not scene.frame_start <= start <= end <= scene.frame_end:
            parser.error("--frames must be inside the shot")
        scene.frame_start, scene.frame_end = start, end
    scene.render.resolution_percentage = args.resolution_percentage
    scene.frame_step = args.frame_step
    scene.render.filepath = str(args.output.resolve() / "frames/#####")
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve() / "ball_rays.blend"))
    if args.render:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__" and "--" in sys.argv:
    main(sys.argv[sys.argv.index("--") + 1 :])

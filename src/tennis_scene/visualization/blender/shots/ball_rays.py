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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).parent))
from camera_planes import camera_frustum, image_plane_point  # noqa: E402
from common import (  # noqa: E402
    Nodes,
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


@dataclass(frozen=True)
class RayShot:
    fps: int = 30
    width: int = 1920
    height: int = 1080
    samples: int = 32
    plate_start: int = 90
    rays_start: int = 150
    ball_start: int = 210
    zoom_start: int = 255
    playback_start: int = 315
    frustum_depth_m: float = 1.5
    ray_radius_m: float = 0.012
    ball_radius_m: float = 0.05
    trail_count: int = 16
    trail_stride: int = 2
    camera_colors: tuple[str, ...] = (
        "#4CC9F0",
        "#F4A261",
        "#8AC926",
        "#C77DFF",
        "#FFCD70",
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
    coords = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    n.set(sep.inputs[0], coords.outputs["Generated"])
    stripe = n.math(
        "LESS_THAN", n.math("FRACT", n.math("MULTIPLY", sep.outputs["Z"], 64.0)), 0.6
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


def _key_segment(
    obj: bpy.types.Object,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    frame: int,
) -> None:
    obj.location = origin.tolist()
    # Calculate in float64 before storing Blender's float32 quaternion. The
    # float32 to_track_quat path can move a 40m endpoint by more than 1e-5m.
    w = math.sqrt(max(0.0, (1.0 + float(direction[2])) / 2.0))
    obj.rotation_quaternion = (
        Quaternion((w, -direction[1] / (2.0 * w), direction[0] / (2.0 * w), 0.0))
        if w > 1e-10
        else Quaternion((0.0, 1.0, 0.0, 0.0))
    )
    obj.scale = (1.0, 1.0, length)
    for channel in ("location", "rotation_quaternion", "scale"):
        obj.keyframe_insert(channel, frame=frame)


def _source_frames(manifest: dict[str, Any], shot: RayShot) -> np.ndarray:
    start, stop = manifest["frames"]
    playback_count = math.ceil((stop - start) * shot.fps / manifest["fps"])
    # Nearest source sample at each output timestamp; timing stays in seconds.
    playback = np.minimum(
        np.floor(np.arange(playback_count) * manifest["fps"] / shot.fps + 0.5).astype(
            int
        )
        + start,
        stop - 1,
    )
    return np.r_[np.full(shot.playback_start, manifest["t0"], int), playback]


def _progress(frame: int, start: int, stop: int) -> float:
    return float(np.clip((frame - start) / (stop - start), 0.0, 1.0))


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


def _marker(
    name: str, *, cross: bool, material: bpy.types.Material
) -> bpy.types.Object:
    if cross:
        a, b = 0.025, 0.003
        vertices = [
            (-a, -b, 0),
            (a, -b, 0),
            (a, b, 0),
            (-a, b, 0),
            (-b, -a, 0),
            (b, -a, 0),
            (b, a, 0),
            (-b, a, 0),
        ]
        obj = _object(name, vertices, [(0, 1, 2, 3), (4, 5, 6, 7)])
    else:
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
    source_frames: np.ndarray,
    shot: RayShot,
) -> None:
    cam = manifest["cameras"][view]
    color = _rgb(shot.camera_colors[view])
    root, plane = camera_frustum(cam, color=color, depth=shot.frustum_depth_m)
    plate_mat, n, surface = _new_material(f"Plate_{view}", transparent=True)
    texture = n.new("ShaderNodeTexImage")
    texture.name = "PlateSequence"
    texture.image = bpy.data.images.load(
        str(bundle / cam["plate_dir"] / f"{manifest['frames'][0]:05d}.jpg")
    )
    texture.image.source = "SEQUENCE"
    texture.image.colorspace_settings.name = "sRGB"
    texture.image_user.frame_duration = max(len(source_frames), manifest["frames"][1])
    texture.image_user.frame_start = 0
    texture.image_user.use_auto_refresh = True
    opacity = n.new("ShaderNodeValue")
    opacity.name = "Opacity"
    n.set(
        surface,
        n.mix_shader(
            opacity.outputs[0],
            n.new("ShaderNodeBsdfTransparent").outputs[0],
            n.emission(texture.outputs["Color"]),
        ),
    )
    plane.data.materials.clear()
    plane.data.materials.append(plate_mat)
    ring_mat = _material(f"DetectionMaterial_{view}", color, strength=2.0)
    cross_mat = _material(f"ReprojectionMaterial_{view}", (1.0, 1.0, 1.0), strength=2.0)
    ring = _marker(f"Detection_{view}", cross=False, material=ring_mat)
    cross = _marker(f"Reprojection_{view}", cross=True, material=cross_mat)
    ring.parent = cross.parent = root
    ray_mat = _material(f"RayMaterial_{view}", color, strength=2.0)
    ray = _cylinder(f"Ray_{view}", shot.ray_radius_m, ray_mat)
    residual_mat = _material(f"ResidualMaterial_{view}", _rgb("#FFD166"), strength=3.0)
    residual = _cylinder(f"Residual_{view}", 0.003, residual_mat)
    ray_valid, ring_valid, cross_valid, residual_valid = [], [], [], []
    for frame, src in enumerate(source_frames):
        t = int(src - manifest["frames"][0])
        state = int(arrays["ray_state"][view, t])
        exists = state != 0
        growth = _progress(frame, shot.rays_start, shot.ball_start)
        ray_valid.append(exists and frame >= shot.rays_start)
        ring_valid.append(exists and frame >= shot.plate_start)
        uv = arrays["reprojection_uv"][view, t]
        cross_valid.append(bool(np.isfinite(uv).all()) and frame >= shot.zoom_start)
        _key_value(
            plate_mat,
            "Opacity",
            _progress(frame, shot.plate_start, shot.rays_start),
            frame,
        )
        _key_value(
            ring_mat,
            "Opacity",
            _progress(frame, shot.plate_start, shot.rays_start),
            frame,
        )
        _key_value(
            cross_mat,
            "Opacity",
            _progress(frame, shot.zoom_start, shot.playback_start),
            frame,
        )
        # Blender uses the absolute numeric filename, rather than an offset
        # relative to the first loaded JPEG. Keep its time range unclamped.
        texture.image_user.frame_offset = int(src) - frame - 1
        texture.image_user.keyframe_insert("frame_offset", frame=frame)
        if exists:
            _key_segment(
                ray,
                arrays["ray_origin"][view],
                arrays["ray_direction"][view, t],
                float(arrays["ray_length"][view, t]) * growth,
                frame,
            )
            ring.location = image_plane_point(
                cam, arrays["detection_uv"][view, t], depth=shot.frustum_depth_m
            )
            ring.location.z += 0.005
            ring.keyframe_insert("location", frame=frame)
        if np.isfinite(uv).all():
            cross.location = image_plane_point(cam, uv, depth=shot.frustum_depth_m)
            cross.location.z += 0.008
            cross.keyframe_insert("location", frame=frame)
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
        point = arrays["ball_position"][t]
        closest = arrays["closest_point"][view, t]
        has_residual = bool(np.isfinite(closest).all())
        residual_valid.append(
            has_residual and shot.ball_start <= frame < shot.zoom_start
        )
        if has_residual:
            delta = closest - point
            length = float(np.linalg.norm(delta))
            if length > 1e-12:
                _key_segment(residual, point, delta / length, length, frame)
            else:
                _key_segment(residual, point, np.array([0.0, 0.0, 1.0]), 0.0, frame)
        pulse = math.sin(math.pi * _progress(frame, shot.ball_start, shot.zoom_start))
        _key_value(residual_mat, "Opacity", pulse, frame)
    for obj, valid in (
        (ray, ray_valid),
        (ring, ring_valid),
        (cross, cross_valid),
        (residual, residual_valid),
    ):
        _key_visibility(obj, np.array(valid))


def _ball(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    source_frames: np.ndarray,
    shot: RayShot,
    output: Path,
) -> None:
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
            frame >= shot.ball_start and valid[src - manifest["frames"][0]]
            for frame, src in enumerate(source_frames)
        ]
    )
    for frame, src in enumerate(source_frames):
        if timeline_valid[frame]:
            ball.location = positions[src - manifest["frames"][0]].tolist()
            ball.keyframe_insert("location", frame=frame)
        _key_value(
            material,
            "Opacity",
            _progress(frame, shot.ball_start, shot.ball_start + 15),
            frame,
        )
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
                frame >= shot.playback_start and past >= 0 and valid[past : t + 1].all()
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
    _key_visibility(
        trail, timeline_valid & (np.arange(len(source_frames)) >= shot.playback_start)
    )


def _camera(
    manifest: dict[str, Any],
    source_frames: np.ndarray,
    arrays: dict[str, np.ndarray],
    shot: RayShot,
) -> None:
    data = bpy.data.cameras.new("ShotCamera")
    camera = bpy.data.objects.new("ShotCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    data.lens = 32.0
    data.clip_start, data.clip_end = 0.05, 1000.0
    radius = max(
        45.0, float(np.linalg.norm(arrays["ray_origin"][:, :2], axis=1).max()) + 30.0
    )
    focus_cam = manifest["cameras"][0]
    rotation = np.array(focus_cam["R"])
    plane = Vector(
        np.array(focus_cam["center"])
        + rotation.T @ np.array([0.0, 0.0, shot.frustum_depth_m])
    )
    close = plane + Vector(rotation.T @ np.array([0.0, 0.0, 2.2]))
    focus = Vector((0.0, 0.0, 1.0))
    for frame in range(len(source_frames)):
        angle = math.radians(-65.0 + 20.0 * frame / max(1, len(source_frames) - 1))
        overview = Vector(
            (radius * math.cos(angle), radius * math.sin(angle), radius * 0.7)
        )
        if frame < shot.zoom_start:
            location, target = overview, focus
        elif frame < shot.playback_start:
            t = _progress(frame, shot.zoom_start, shot.playback_start - 1)
            location, target = overview.lerp(close, t), focus.lerp(plane, t)
        else:
            t = _progress(frame, shot.playback_start, len(source_frames) - 1)
            # Cut back to the reconstruction at playback start so brief source
            # outliers/missing frames remain visible; then pull out slowly.
            location, target = overview * (0.9 + 0.1 * t), focus
        camera.location = location
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = _look_rotation(target - location, 0.0)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)


def _constant_keys() -> None:
    # Shader, sequence, visibility and source geometry must switch together;
    # missing observations are never interpolated into the next source frame.
    for action in bpy.data.actions:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for point in curve.keyframe_points:
                            point.interpolation = "CONSTANT"


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
            "closest_point",
            "ball_position",
            "reprojection_uv",
            "detection_uv",
        )
    }
    sources = _source_frames(manifest, shot)
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
            _build_view(bundle, manifest, arrays, view, sources, shot)
        _ball(arrays, manifest, sources, shot, output)
        _camera(manifest, sources, arrays, shot)
        _constant_keys()
        _build_compositor(scene, name="RaysComposite")
        scene["rays_bundle_schema"] = manifest["schema"]
        scene["source_frames"] = sources.tolist()
        scene["ray_shot"] = json.dumps(asdict(shot))
        (output / "shot_manifest.json").write_text(
            json.dumps(
                {
                    "shot": asdict(shot),
                    "source_frames": sources.tolist(),
                    "source_sampling": "nearest source timestamp at 30fps; no gap filling",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--frames", help="start:end, inclusive")
    parser.add_argument("--samples", type=int, default=32)
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
        args.bundle.resolve(), args.output.resolve(), RayShot(samples=args.samples)
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

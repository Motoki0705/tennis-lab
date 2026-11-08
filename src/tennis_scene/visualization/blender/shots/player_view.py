"""Opaque player views from a prepared C1/C2 package (Blender Python only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import Nodes, _net_geometry, _pc2_frame0, court_image, net_shader


def material(name, color):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    tree = value.node_tree
    tree.nodes.clear()
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Alpha"].default_value = 1
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Emission Strength"].default_value = 0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return value


def mesh(name, vertices, faces, mat):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices.tolist(), [], faces.tolist())
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    data.materials.append(mat)
    return obj


def configure_gpu(scene, backend):
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = backend
    preferences.get_devices()
    devices = []
    for device in preferences.devices:
        device.use = device.type == backend
        if device.use:
            devices.append({"name": device.name, "type": device.type})
    if not devices:
        raise RuntimeError(f"No {backend} GPU; CPU fallback is forbidden")
    scene.cycles.device = "GPU"
    return devices


def build(package, viewer_id, backend):
    info = json.loads((package / "manifest.json").read_text())
    if info["schema"] != "tennis_scene_player_view_blender_v1":
        raise ValueError("Expected prepared player-view Blender input")
    for name, expected in info["files"].items():
        with (package / name).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Shot input digest differs: {name}")
    matches = [p for p in info["players"] if p["player_id"] == viewer_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown/duplicate viewer ID {viewer_id}")
    viewer = matches[0]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = info["config"]["samples"]
    scene.cycles.use_denoising = True
    devices = configure_gpu(scene, backend)
    scene.render.use_motion_blur = False
    fps = info["fps"]
    scene.render.fps = round(fps)
    scene.render.fps_base = scene.render.fps / fps
    camera_config = info["camera_config"]
    scene.render.resolution_x = camera_config["width"]
    scene.render.resolution_y = camera_config["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.frame_start, scene.frame_end = 0, info["num_frames"] - 1
    cam_data = bpy.data.cameras.new("SavedHeadCamera")
    cam_data.sensor_fit, cam_data.sensor_width = "HORIZONTAL", 36
    cam_data.lens = viewer["lens_mm"]
    cam_data.clip_start = camera_config["near_m"]
    cam_data.clip_end = camera_config["far_m"]
    camera = bpy.data.objects.new("SavedHeadCamera", cam_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_arrays = dict(np.load(package / viewer["camera"]))
    faces = np.load(package / "smpl_faces.npy")
    players, caches = [], []
    gray = material("OpaqueGrayPlayer", info["config"]["player_color"])
    for player in info["players"]:
        cache = package / player["pc2"]
        obj = mesh(f"PlayerID{player['player_id']}", _pc2_frame0(cache), faces, gray)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj["player_id"] = player["player_id"]
        modifier = obj.modifiers.new("SavedPC2", "MESH_CACHE")
        modifier.cache_format, modifier.filepath = "PC2", str(cache.resolve())
        modifier.frame_start = 0
        modifier.forward_axis, modifier.up_axis = "POS_Y", "POS_Z"
        modifier.interpolation = "NONE"
        modifier.use_time_override, modifier.time_mode = True, "FRAME"
        players.append(obj)
        with cache.open("rb") as handle:
            _, _, points, _, _, samples = struct.unpack("<12siiffi", handle.read(32))
        caches.append(
            np.memmap(
                cache, dtype="<f4", mode="r", offset=32, shape=(samples, points, 3)
            )
        )
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=info["config"]["ball_radius_m"], segments=24, ring_count=12
    )
    ball = bpy.context.object
    ball.name = "OpaqueGreenBall"
    ball.data.materials.append(
        material("OpaqueGreenBall", info["config"]["ball_color"])
    )
    for polygon in ball.data.polygons:
        polygon.use_smooth = True
    court = info["court"]
    xmin, xmax, ymin, ymax = court["runoff_xy"]
    court_mat = material("Court", (0.08, 0.24, 0.34))
    ground = mesh(
        "Court",
        np.array([[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0]]),
        np.array([[0, 1, 2, 3]]),
        court_mat,
    )
    uv = ground.data.uv_layers.new()
    for entry, value in zip(uv.data, [(0, 0), (1, 0), (1, 1), (0, 1)], strict=True):
        entry.uv = value
    texture = court_mat.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = court_image(
        court,
        court_color=(0.08, 0.24, 0.34),
        runoff_color=(0.055, 0.12, 0.14),
        px_per_m=60,
    )
    court_mat.node_tree.links.new(
        texture.outputs["Color"],
        court_mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"],
    )
    verts, net_faces, tape = _net_geometry(court)
    net_mat = bpy.data.materials.new("Net")
    net_mat.use_nodes = True
    net_mat.node_tree.nodes.clear()
    output = net_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    net_mat.node_tree.links.new(
        net_shader(Nodes(net_mat.node_tree)), output.inputs["Surface"]
    )
    net = mesh("Net", verts, net_faces, net_mat)
    attribute = net.data.attributes.new("tape", "FLOAT", "POINT")
    attribute.data.foreach_set("value", tape.astype(np.float32))
    for side in (-1, 1):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.04,
            depth=court["net"]["height_post"],
            location=(
                side * court["net"]["post_x"],
                0,
                court["net"]["height_post"] / 2,
            ),
        )
        bpy.context.object.data.materials.append(
            material(f"Post{side}", (0.25, 0.25, 0.25))
        )
    world = bpy.data.worlds.new("PlayerViewWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.13,
        0.16,
        0.20,
        1,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7
    scene.world = world
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 2
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(30), math.radians(-20), math.radians(-35))
    return info, scene, camera_arrays, players, caches, ball, devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--player-id", required=True, type=int)
    parser.add_argument("--gpu-backend", choices=("OPTIX", "CUDA"), default="OPTIX")
    parser.add_argument("--source-frames", nargs="+", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(f"Render output already exists: {args.output}")
    info, scene, cameras, players, caches, ball, devices = build(
        args.package, args.player_id, args.gpu_backend
    )
    source_frames = np.load(args.package / "source_frames.npy")
    selected = (
        source_frames.tolist() if args.source_frames is None else args.source_frames
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(source_frames.tolist())
    ):
        raise ValueError(
            "Render frame selection must be unique and inside the source interval"
        )
    valid = np.load(args.package / "player_smpl_valid.npy")
    ball_valid = np.load(args.package / "ball_3d_valid.npy")
    ball_positions = np.load(args.package / "ball_positions.npy")
    args.output.mkdir(parents=True)
    (args.output / "frames").mkdir()
    records = []
    static = [
        o for o in scene.objects if o.type == "MESH" and o not in players and o != ball
    ]
    for output_index, source_frame in enumerate(selected):
        index = int(source_frame - source_frames[0])
        scene.frame_set(index)
        camera_valid = bool(cameras["valid"][index])
        for obj in static:
            obj.hide_render = not camera_valid
        if camera_valid:
            transform = Matrix(cameras["rotation"][index].tolist()).to_4x4()
            transform.translation = cameras["location"][index].tolist()
            scene.camera.matrix_world = transform
        for p, obj in enumerate(players):
            obj.hide_render = not (
                camera_valid and valid[p, index] and obj["player_id"] != args.player_id
            )
            obj.hide_viewport = False  # keep geometry evaluation available to the audit
            obj.modifiers["SavedPC2"].eval_frame = index
        ball.hide_render = not (camera_valid and ball_valid[index])
        if ball_valid[index]:
            ball.location = ball_positions[index].tolist()
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
            (0.13, 0.16, 0.20, 1) if camera_valid else (0.005, 0.005, 0.005, 1)
        )
        bpy.context.view_layer.update()
        errors = []
        graph = bpy.context.evaluated_depsgraph_get()
        for p, obj in enumerate(players):
            evaluated = obj.evaluated_get(graph)
            actual = np.empty((len(evaluated.data.vertices), 3), np.float32)
            evaluated.data.vertices.foreach_get("co", actual.ravel())
            error = float(np.max(np.abs(actual - caches[p][index])))
            if error > 1e-5:
                raise ValueError(
                    f"PC2/frame mismatch ID{obj['player_id']} source {source_frame}: {error}"
                )
            errors.append(error)
        if scene.render.use_motion_blur or any(
            o.animation_data for o in players + [ball, scene.camera]
        ):
            raise ValueError("Unexpected interpolation/motion blur/animation")
        frame_path = args.output / "frames" / f"{output_index:05d}.png"
        scene.render.filepath = str(frame_path.resolve())
        bpy.ops.render.render(write_still=True)
        records.append(
            {
                "output_index": output_index,
                "source_frame": int(source_frame),
                "cache_index": index,
                "camera_valid": camera_valid,
                "camera_reason": int(cameras["reason"][index]),
                "camera_matrix_world": [list(row) for row in scene.camera.matrix_world]
                if camera_valid
                else None,
                "player_hidden": [bool(o.hide_render) for o in players],
                "ball_hidden": bool(ball.hide_render),
                "ball_position": list(ball.location) if ball_valid[index] else None,
                "pc2_max_error_m": errors,
            }
        )
        # Publish only records whose PNG has completed, so interrupted runs remain identifiable.
        report = {
            "schema": "tennis_scene_player_view_render_v1",
            "package_sha256": hashlib.sha256(
                (args.package / "manifest.json").read_bytes()
            ).hexdigest(),
            "player_id": args.player_id,
            "blender_version": bpy.app.version_string,
            "engine": scene.render.engine,
            "gpu_devices": devices,
            "motion_blur": scene.render.use_motion_blur,
            "material": {
                name: {
                    "alpha": bpy.data.materials[name]
                    .node_tree.nodes.get("Principled BSDF")
                    .inputs["Alpha"]
                    .default_value,
                    "emission_strength": bpy.data.materials[name]
                    .node_tree.nodes.get("Principled BSDF")
                    .inputs["Emission Strength"]
                    .default_value,
                }
                for name in ("OpaqueGrayPlayer", "OpaqueGreenBall")
            },
            "frames": records,
            "complete": len(records) == len(selected),
        }
        temporary = args.output / ".render.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(args.output / "render.json")
    print(f"PLAYER_VIEW_COMPLETE ID{args.player_id}: {len(records)} frames", flush=True)


if __name__ == "__main__":
    main()

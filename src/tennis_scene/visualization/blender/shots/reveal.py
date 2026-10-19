"""Build and render the "input video -> grid -> court -> players" reveal shot.

Runs inside Blender 5.2 (its bundled Python + numpy); it does not import the
repository. The input is a bundle written by
``python -m src.tennis_scene.scripts.export_blender``. Usage, bundle layout and
verified invariants: ``src/tennis_scene/visualization/blender/export/README.md``.

How the shot works: the camera reproduces the calibrated input camera exactly,
and every surface initially emits the input frame sampled at its own screen
position (``Window`` texture coordinates), so the first frames are the input
video. Three reveal fronts then sweep away from the camera along the ground
(distance along the camera's ground-projected forward axis, perturbed by 4D
noise so the edge wobbles instead of forming concentric rings): the grid world,
then the court and net, then players and ball. Players and the net add a
height term so they materialize bottom-up. After the reveal the camera leaves
the input viewpoint to show that the scene is really 3D.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    Nodes,
    _build_compositor,
    _disk,
    _distance_fade,
    _dome,
    _grid_lines,
    _group_socket,
    _key_visibility,
    _look_rotation,
    _net_geometry,
    _new_material,
    _pc2_frame0,
    _write_pc2,
    ball_shader,
    court_image,
    net_shader,
    player_shader,
)

BUNDLE_SCHEMA = "tennis_scene_blender_bundle_v1"
COLLECTION = "TennisReveal"


@dataclass(frozen=True)
class Shot:
    """Timing (scene frames) and look parameters of the reveal shot."""

    reveal_start: int = 150
    grid_frames: int = 120
    court_delay: int = 22
    actor_delay: int = 48
    front_start_m: float = 1.0
    front_end_m: float = 215.0
    front_power: float = 2.0
    mask_width_m: float = 2.5
    band_width_m: float = 0.7
    band_strength: float = 2.5
    warp_large_m: float = 7.0
    warp_small_m: float = 1.6
    actor_height_gain: float = 6.0
    net_height_gain: float = 10.0
    move_start: int = 420
    move_end_location: tuple[float, float, float] = (-24.0, -7.0, 13.0)
    move_end_target: tuple[float, float, float] = (0.0, 1.5, 0.0)
    move_look_distance_m: float = 24.0
    ball_radius_m: float = 0.05
    trail_count: int = 16
    trail_stride: int = 2
    samples: int = 32
    # Monochrome world (grid, dome, reveal band, net); only the court, players
    # and ball carry color.
    world_color: tuple[float, float, float] = (0.012, 0.012, 0.012)
    grid_color: tuple[float, float, float] = (0.4, 0.4, 0.4)
    band_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    court_color: tuple[float, float, float] = (0.03, 0.12, 0.34)
    runoff_color: tuple[float, float, float] = (0.018, 0.075, 0.095)
    player_colors: tuple[tuple[float, float, float], ...] = (
        (0.80, 0.16, 0.08),  # renderer #E76F51 (linear, saturated for EEVEE)
        (0.03, 0.34, 0.27),  # renderer #2A9D8F
    )
    ball_color: tuple[float, float, float] = (0.06, 0.85, 0.12)


# --------------------------------------------------------------------------
# Reveal field
# --------------------------------------------------------------------------


def build_params_group(shot: Shot, num_frames: int) -> bpy.types.NodeTree:
    """Shared animated fronts (meters) and noise time, keyed once per frame."""
    group = bpy.data.node_groups.new("RevealParams", "ShaderNodeTree")
    out = group.nodes.new("NodeGroupOutput")
    names = ("grid_front", "court_front", "actor_front", "time")
    for name in names:
        _group_socket(group, name, "OUTPUT")
        value = group.nodes.new("ShaderNodeValue")
        value.name = name
        group.links.new(value.outputs[0], out.inputs[name])

    starts = {
        "grid_front": shot.reveal_start,
        "court_front": shot.reveal_start + shot.court_delay,
        "actor_front": shot.reveal_start + shot.actor_delay,
    }
    for frame in range(num_frames):
        for name, start in starts.items():
            u = min(max((frame - start) / shot.grid_frames, 0.0), 1.0)
            front = shot.front_start_m + (shot.front_end_m - shot.front_start_m) * (
                u**shot.front_power
            )
            if frame < start:
                front = -50.0  # nothing revealed yet (well behind the camera)
            socket = group.nodes[name].outputs[0]
            socket.default_value = front
            socket.keyframe_insert("default_value", frame=frame)
        time_socket = group.nodes["time"].outputs[0]
        time_socket.default_value = frame / 60.0
        time_socket.keyframe_insert("default_value", frame=frame)
    return group


def build_field_group(
    shot: Shot, origin: Vector, forward: Vector
) -> bpy.types.NodeTree:
    """Mask/band of one front at a world position.

    ``s = dot(P.xy - origin.xy, forward.xy) + P.z * height_gain + warp(P, t)``;
    ``mask = smoothstep(0, width, front - s)``, ``band = exp(-((front - s)/bw)^2)``.
    """
    group = bpy.data.node_groups.new("RevealField", "ShaderNodeTree")
    for name, kind in (
        ("Position", "NodeSocketVector"),
        ("Front", "NodeSocketFloat"),
        ("HeightGain", "NodeSocketFloat"),
        ("Time", "NodeSocketFloat"),
    ):
        _group_socket(group, name, "INPUT", kind)
    for name in ("Mask", "Band"):
        _group_socket(group, name, "OUTPUT")
    n = Nodes(group)
    gin = group.nodes.new("NodeGroupInput")
    gout = group.nodes.new("NodeGroupOutput")
    sep = n.new("ShaderNodeSeparateXYZ")
    group.links.new(gin.outputs["Position"], sep.inputs[0])

    s = n.math(
        "ADD",
        n.math("MULTIPLY", n.math("SUBTRACT", sep.outputs[0], origin.x), forward.x),
        n.math("MULTIPLY", n.math("SUBTRACT", sep.outputs[1], origin.y), forward.y),
    )
    s = n.math("ADD", s, n.math("MULTIPLY", sep.outputs[2], gin.outputs["HeightGain"]))

    def noise(scale: float, speed: float, amplitude: float) -> Any:
        tex = group.nodes.new("ShaderNodeTexNoise")
        tex.noise_dimensions = "4D"
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Detail"].default_value = 1.0
        group.links.new(gin.outputs["Position"], tex.inputs["Vector"])
        n.set(tex.inputs["W"], n.math("MULTIPLY", gin.outputs["Time"], speed))
        centered = n.math("SUBTRACT", tex.outputs["Fac"], 0.5)
        return n.math("MULTIPLY", centered, 2.0 * amplitude)

    warp = n.math(
        "ADD",
        noise(0.035, 0.35, shot.warp_large_m),
        noise(0.22, 0.9, shot.warp_small_m),
    )
    delta = n.math("SUBTRACT", gin.outputs["Front"], n.math("ADD", s, warp))

    ramp = n.new(
        "ShaderNodeMapRange", **{"From Min": 0.0, "From Max": shot.mask_width_m}
    )
    ramp.interpolation_type = "SMOOTHSTEP"
    ramp.clamp = True
    group.links.new(delta, ramp.inputs["Value"])
    group.links.new(ramp.outputs[0], gout.inputs["Mask"])

    scaled = n.math("DIVIDE", delta, shot.band_width_m)
    band = n.math(
        "EXPONENT", n.math("MULTIPLY", n.math("MULTIPLY", scaled, scaled), -1.0)
    )
    group.links.new(band, gout.inputs["Band"])
    return group


@dataclass
class Reveal:
    params: bpy.types.NodeTree
    field: bpy.types.NodeTree
    plate: bpy.types.Image
    num_frames: int


def _field(n: Nodes, reveal: Reveal, front: str, height_gain: float) -> tuple[Any, Any]:
    params = n.tree.nodes.new("ShaderNodeGroup")
    params.node_tree = reveal.params
    geometry = n.tree.nodes.new("ShaderNodeNewGeometry")
    field = n.tree.nodes.new("ShaderNodeGroup")
    field.node_tree = reveal.field
    n.set(field.inputs["Position"], geometry.outputs["Position"])
    n.set(field.inputs["Front"], params.outputs[front])
    n.set(field.inputs["Time"], params.outputs["time"])
    field.inputs["HeightGain"].default_value = height_gain
    return field.outputs["Mask"], field.outputs["Band"]


def _plate_shader(n: Nodes, reveal: Reveal) -> Any:
    coords = n.tree.nodes.new("ShaderNodeTexCoord")
    tex = n.tree.nodes.new("ShaderNodeTexImage")
    tex.image = reveal.plate
    tex.extension = "EXTEND"
    tex.interpolation = "Linear"
    user = tex.image_user
    user.frame_duration = reveal.num_frames
    user.frame_start = 0
    # Sequence file number = scene frame - frame_start + frame_offset + 1; the
    # plate numbering starts at 00000 == scene frame 0 (verified by rendering
    # frame 100 against plate/00100.jpg).
    user.frame_offset = -1
    user.use_auto_refresh = True
    n.tree.links.new(coords.outputs["Window"], tex.inputs["Vector"])
    return n.emission(tex.outputs["Color"])


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------


def ground_material(
    shot: Shot,
    reveal: Reveal,
    court_image: bpy.types.Image,
    runoff: list[float],
    cam: Vector,
) -> bpy.types.Material:
    material, n, surface = _new_material("RevealGround", transparent=False)
    geometry = n.tree.nodes.new("ShaderNodeNewGeometry")
    position = geometry.outputs["Position"]
    grid_mask, grid_band = _field(n, reveal, "grid_front", 0.0)
    court_mask, court_band = _field(n, reveal, "court_front", 0.0)

    minor = n.math(
        "MULTIPLY",
        _grid_lines(n, position, 1.0, 0.025),
        _distance_fade(n, position, cam, 12.0, 45.0),
    )
    major = n.math(
        "MULTIPLY",
        _grid_lines(n, position, 5.0, 0.045),
        _distance_fade(n, position, cam, 30.0, 150.0),
    )
    lines = n.math(
        "ADD", n.math("MULTIPLY", minor, 0.9), n.math("MULTIPLY", major, 2.2)
    )
    grid_emit = n.add_shader(
        n.emission((*shot.world_color, 1.0)),
        n.emission(n.rgb_scale(shot.grid_color, lines)),
    )

    # Court surface: an image over the run-off rectangle in court XY.
    xmin, xmax, ymin, ymax = runoff
    sep = n.new("ShaderNodeSeparateXYZ")
    n.set(sep.inputs[0], position)
    u = n.math("DIVIDE", n.math("SUBTRACT", sep.outputs[0], xmin), xmax - xmin)
    v = n.math("DIVIDE", n.math("SUBTRACT", sep.outputs[1], ymin), ymax - ymin)
    uv = n.new("ShaderNodeCombineXYZ")
    n.set(uv.inputs[0], u)
    n.set(uv.inputs[1], v)
    tex = n.tree.nodes.new("ShaderNodeTexImage")
    tex.image = court_image
    tex.extension = "CLIP"
    tex.interpolation = "Cubic"
    n.tree.links.new(uv.outputs[0], tex.inputs["Vector"])
    inside = n.math(
        "MULTIPLY",
        n.math("MULTIPLY", n.math("GREATER_THAN", u, 0.0), n.math("LESS_THAN", u, 1.0)),
        n.math("MULTIPLY", n.math("GREATER_THAN", v, 0.0), n.math("LESS_THAN", v, 1.0)),
    )
    court_bsdf = n.new(
        "ShaderNodeBsdfPrincipled", Roughness=1.0, **{"Specular IOR Level": 0.0}
    )
    n.set(court_bsdf.inputs["Base Color"], tex.outputs["Color"])
    n.set(court_bsdf.inputs["Emission Color"], (1.0, 1.0, 1.0, 1.0))
    # Lines are the only texels brighter than 0.5 in red (surface colors < 0.35).
    channels = n.new("ShaderNodeSeparateColor")
    n.set(channels.inputs[0], tex.outputs["Color"])
    line_mask = n.new("ShaderNodeMapRange", **{"From Min": 0.5, "From Max": 0.9})
    n.set(line_mask.inputs["Value"], channels.outputs[0])
    n.set(
        court_bsdf.inputs["Emission Strength"],
        n.math("MULTIPLY", line_mask.outputs[0], 0.35),
    )

    shader = n.mix_shader(grid_mask, _plate_shader(n, reveal), grid_emit)
    shader = n.mix_shader(
        n.math("MULTIPLY", court_mask, inside), shader, court_bsdf.outputs[0]
    )
    glow = n.math(
        "ADD",
        n.math("MULTIPLY", grid_band, shot.band_strength),
        n.math(
            "MULTIPLY", n.math("MULTIPLY", court_band, inside), shot.band_strength * 0.4
        ),
    )
    shader = n.add_shader(shader, n.emission(n.rgb_scale(shot.band_color, glow)))
    n.tree.links.new(shader, surface)
    return material


def backdrop_material(shot: Shot, reveal: Reveal) -> bpy.types.Material:
    material, n, surface = _new_material("RevealBackdrop", transparent=False)
    geometry = n.tree.nodes.new("ShaderNodeNewGeometry")
    grid_mask, grid_band = _field(n, reveal, "grid_front", 0.0)
    sep = n.new("ShaderNodeSeparateXYZ")
    n.set(sep.inputs[0], geometry.outputs["Normal"])
    # Horizon glow fading to black overhead (dome normals point inward).
    height = n.math("ABSOLUTE", sep.outputs[2])
    ramp = n.new(
        "ShaderNodeMapRange",
        **{"From Min": 0.0, "From Max": 0.35, "To Min": 1.0, "To Max": 0.0},
    )
    ramp.interpolation_type = "SMOOTHSTEP"
    n.set(ramp.inputs["Value"], height)
    # Faint latitude/longitude grid so the enclosing space reads as "digital".
    position = geometry.outputs["Position"]
    pos_sep = n.new("ShaderNodeSeparateXYZ")
    n.set(pos_sep.inputs[0], position)
    azimuth = n.math("ARCTAN2", pos_sep.outputs[1], pos_sep.outputs[0])
    lat_long = n.new("ShaderNodeCombineXYZ")
    n.set(lat_long.inputs[0], n.math("MULTIPLY", azimuth, 170.0))  # arc length (m)
    n.set(lat_long.inputs[1], pos_sep.outputs[2])
    dome_lines = n.math(
        "MULTIPLY",
        _grid_lines(n, lat_long.outputs[0], 12.0, 0.35),
        n.math("ADD", n.math("MULTIPLY", ramp.outputs[0], 0.25), 0.05),
    )
    sky = n.add_shader(
        n.emission((*shot.world_color, 1.0)),
        n.emission(n.rgb_scale((0.07, 0.07, 0.07), ramp.outputs[0])),
    )
    sky = n.add_shader(sky, n.emission(n.rgb_scale(shot.grid_color, dome_lines)))
    shader = n.mix_shader(grid_mask, _plate_shader(n, reveal), sky)
    shader = n.add_shader(
        shader,
        n.emission(
            n.rgb_scale(
                shot.band_color,
                n.math("MULTIPLY", grid_band, shot.band_strength * 0.03),
            )
        ),
    )
    n.tree.links.new(shader, surface)
    return material


def _staged_material(
    name: str,
    shot: Shot,
    reveal: Reveal,
    front: str,
    height_gain: float,
    look: Any,
    n: Nodes,
    surface: Any,
) -> None:
    """plate -> transparent (after grid front) -> ``look`` (after ``front``)."""
    band_color = shot.band_color
    grid_mask, _ = _field(n, reveal, "grid_front", 0.0)
    mask, band = _field(n, reveal, front, height_gain)
    transparent = n.new("ShaderNodeBsdfTransparent").outputs[0]
    staged = n.mix_shader(mask, transparent, look)
    shader = n.mix_shader(grid_mask, _plate_shader(n, reveal), staged)
    shader = n.add_shader(
        shader,
        n.emission(
            n.rgb_scale(
                band_color,
                n.math(
                    "MULTIPLY",
                    n.math("MULTIPLY", band, grid_mask),
                    shot.band_strength * 0.8,
                ),
            )
        ),
    )
    n.tree.links.new(shader, surface)


def player_material(shot: Shot, reveal: Reveal, index: int) -> bpy.types.Material:
    material, n, surface = _new_material(f"RevealPlayer{index}", transparent=True)
    color = shot.player_colors[index % len(shot.player_colors)]
    look = player_shader(n, color)
    _staged_material(
        material.name,
        shot,
        reveal,
        "actor_front",
        shot.actor_height_gain,
        look,
        n,
        surface,
    )
    return material


def net_material(shot: Shot, reveal: Reveal) -> bpy.types.Material:
    material, n, surface = _new_material("RevealNet", transparent=True)
    net_look = net_shader(n)
    _staged_material(
        material.name,
        shot,
        reveal,
        "court_front",
        shot.net_height_gain,
        net_look,
        n,
        surface,
    )
    return material


def post_material(shot: Shot, reveal: Reveal) -> bpy.types.Material:
    material, n, surface = _new_material("RevealNetPost", transparent=True)
    look = n.new(
        "ShaderNodeBsdfPrincipled",
        **{"Base Color": (0.05, 0.05, 0.05, 1.0), "Roughness": 1.0},
    ).outputs[0]
    _staged_material(
        material.name,
        shot,
        reveal,
        "court_front",
        shot.net_height_gain,
        look,
        n,
        surface,
    )
    return material


def ball_material(
    shot: Shot, reveal: Reveal, name: str, *, fade_attribute: str | None
) -> bpy.types.Material:
    material, n, surface = _new_material(name, transparent=True)
    look = ball_shader(n, shot.ball_color, fade_attribute=fade_attribute)
    _staged_material(
        material.name,
        shot,
        reveal,
        "actor_front",
        0.0,
        look,
        n,
        surface,
    )
    return material


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _link(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.data.collections[COLLECTION].objects.link(obj)
    return obj


def _mesh_object(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    material: bpy.types.Material,
    *,
    smooth: bool,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    if smooth:
        mesh.shade_smooth()
    mesh.materials.append(material)
    return _link(bpy.data.objects.new(name, mesh))


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


def animate_camera(camera: bpy.types.Object, shot: Shot, num_frames: int) -> None:
    """Hold the calibrated pose until ``move_start``, then orbit out and up."""
    q0 = camera.matrix_world.to_quaternion()
    p0 = camera.matrix_world.translation.copy()
    d0 = q0 @ Vector((0.0, 0.0, -1.0))
    base = d0.to_track_quat("-Z", "Y")
    roll_q = base.inverted() @ q0
    roll = 2.0 * math.atan2(roll_q.z, roll_q.w)
    if (_look_rotation(d0, roll).rotation_difference(q0).angle) > 1e-5:
        raise RuntimeError("Calibrated camera rotation is not a pure look-at + roll.")
    t0 = p0 + d0 * shot.move_look_distance_m
    t1 = Vector(shot.move_end_target)
    p1 = Vector(shot.move_end_location)

    def cyl(p: Vector) -> tuple[float, float, float]:
        return math.atan2(p.y, p.x), math.hypot(p.x, p.y), p.z

    a0, r0, z0 = cyl(p0)
    a1, r1, z1 = cyl(p1)
    if a1 - a0 > math.pi:
        a1 -= 2.0 * math.pi
    elif a0 - a1 > math.pi:
        a1 += 2.0 * math.pi

    camera.rotation_mode = "QUATERNION"
    span = max(num_frames - 1 - shot.move_start, 1)
    for frame in range(num_frames):
        u = min(max((frame - shot.move_start) / span, 0.0), 1.0)
        e = u * u * u * (u * (6.0 * u - 15.0) + 10.0)  # smootherstep
        a, r, z = a0 + (a1 - a0) * e, r0 + (r1 - r0) * e, z0 + (z1 - z0) * e
        position = Vector((r * math.cos(a), r * math.sin(a), z))
        target = t0.lerp(t1, e)
        if frame <= shot.move_start:
            position, rotation = p0, q0
        else:
            rotation = _look_rotation(target - position, roll * (1.0 - e))
        camera.location = position
        camera.rotation_quaternion = rotation
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)


# --------------------------------------------------------------------------
# Scene assembly
# --------------------------------------------------------------------------


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.node_groups,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.collections,
        bpy.data.actions,
        bpy.data.worlds,
    ):
        for block in list(collection):
            collection.remove(block)


def load_manifest(bundle: Path) -> dict[str, Any]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported bundle schema {manifest.get('schema')!r}.")
    return manifest


def build(bundle: Path, output: Path, shot: Shot) -> bpy.types.Scene:
    """Build the full reveal scene from ``bundle``; caches go to ``output``."""
    manifest = load_manifest(bundle)
    output.mkdir(parents=True, exist_ok=True)
    num_frames = int(manifest["num_frames"])
    _clear_scene()
    scene = bpy.context.scene
    scene.collection.children.link(bpy.data.collections.new(COLLECTION))

    fps = float(manifest["fps"])
    if not math.isfinite(fps) or not 1.0 <= fps <= 240.0:
        raise ValueError(f"Bundle fps must be finite and in [1, 240], got {fps}.")
    scene.render.fps = round(fps)
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start, scene.frame_end = 0, num_frames - 1
    scene.render.resolution_x = int(manifest["width"])
    scene.render.resolution_y = int(manifest["height"])
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = shot.samples
    scene.render.filter_size = 0.75
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    # Camera (exact calibrated pose).
    blender_cam = manifest["camera"]["blender"]
    cam_data = bpy.data.cameras.new("InputCamera")
    cam_data.sensor_fit = blender_cam["sensor_fit"]
    cam_data.sensor_width = blender_cam["sensor_width_mm"]
    cam_data.lens = blender_cam["lens_mm"]
    cam_data.clip_start, cam_data.clip_end = 0.1, 1000.0
    camera = _link(bpy.data.objects.new("InputCamera", cam_data))
    camera.matrix_world = (
        Matrix.Translation(Vector(blender_cam["location"]))
        @ Matrix(blender_cam["rotation_matrix"]).to_4x4()
    )
    scene.camera = camera
    cam_pos = Vector(blender_cam["location"])
    forward = Matrix(blender_cam["rotation_matrix"]) @ Vector((0.0, 0.0, -1.0))
    forward_xy = Vector((forward.x, forward.y, 0.0)).normalized()

    # Plate sequence.
    plate_first = bundle / manifest["plate"]["pattern"].format(frame=0)
    plate = bpy.data.images.load(str(plate_first))
    plate.source = "SEQUENCE"
    plate.colorspace_settings.name = "sRGB"

    reveal = Reveal(
        params=build_params_group(shot, num_frames),
        field=build_field_group(shot, cam_pos, forward_xy),
        plate=plate,
        num_frames=num_frames,
    )

    court = manifest["court"]
    ground_verts, ground_faces = _disk(160.0, 128)
    _mesh_object(
        "Ground",
        ground_verts,
        ground_faces,
        ground_material(
            shot,
            reveal,
            court_image(
                court, court_color=shot.court_color, runoff_color=shot.runoff_color
            ),
            court["runoff_xy"],
            cam_pos,
        ),
        smooth=False,
    )
    dome_verts, dome_faces = _dome(170.0, 24, 128)
    dome = _mesh_object(
        "Backdrop", dome_verts, dome_faces, backdrop_material(shot, reveal), smooth=True
    )
    # Camera-only: a closed dome would otherwise shadow the sun and occlude
    # the world ambient for everything inside it.
    dome.visible_shadow = False
    dome.visible_diffuse = False
    dome.visible_glossy = False

    net_verts, net_faces, tape = _net_geometry(court)
    net = _mesh_object(
        "Net", net_verts, net_faces, net_material(shot, reveal), smooth=False
    )
    attribute = net.data.attributes.new("tape", "FLOAT", "POINT")
    attribute.data.foreach_set("value", tape.astype(np.float32))
    post_mat = post_material(shot, reveal)
    for side in (-1.0, 1.0):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.04,
            depth=court["net"]["height_post"],
            location=(
                side * court["net"]["post_x"],
                0.0,
                court["net"]["height_post"] / 2.0,
            ),
        )
        post = bpy.context.active_object
        for coll in post.users_collection:
            coll.objects.unlink(post)
        _link(post)
        post.data.materials.append(post_mat)

    # Players: base mesh = first cached frame, animation via Mesh Cache (PC2).
    faces = np.load(bundle / manifest["smpl_faces"])
    player_valid = np.load(bundle / manifest["player_valid"])
    for player in manifest["players"]:
        index = int(player["index"])
        pc2 = bundle / player["pc2"]
        obj = _mesh_object(
            f"Player{index}",
            _pc2_frame0(pc2),
            faces,
            player_material(shot, reveal, index),
            smooth=True,
        )
        mod = obj.modifiers.new("SceneMotion", "MESH_CACHE")
        mod.cache_format = "PC2"
        mod.filepath = str(pc2)
        mod.frame_start = 0.0
        mod.forward_axis, mod.up_axis = "POS_Y", "POS_Z"
        _key_visibility(obj, player_valid[index])

    _build_ball(manifest, bundle, output, shot, reveal, num_frames)

    # Lighting: world ambient (hidden behind the dome) + sun.
    world = bpy.data.worlds.new("RevealWorld")
    scene.world = world
    wn = Nodes(world.node_tree)
    world.node_tree.nodes.clear()
    background = wn.new(
        "ShaderNodeBackground", Color=(0.05, 0.05, 0.05, 1.0), Strength=1.0
    )
    world_out = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs[0], world_out.inputs["Surface"])
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.5
    sun_data.angle = math.radians(1.5)
    sun = _link(bpy.data.objects.new("Sun", sun_data))
    sun.rotation_euler = (math.radians(38.0), 0.0, math.radians(-150.0))

    _build_compositor(scene, name="RevealComposite")
    animate_camera(camera, shot, num_frames)
    scene.frame_set(0)
    return scene


def _build_ball(
    manifest: dict[str, Any],
    bundle: Path,
    output: Path,
    shot: Shot,
    reveal: Reveal,
    num_frames: int,
) -> None:
    ball_info = manifest["ball"]
    positions = np.load(bundle / ball_info["positions"])
    valid = np.load(bundle / ball_info["valid"])

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=shot.ball_radius_m, segments=24, ring_count=12
    )
    ball = bpy.context.active_object
    ball.name = "Ball"
    for coll in ball.users_collection:
        coll.objects.unlink(ball)
    _link(ball)
    ball.data.materials.append(
        ball_material(shot, reveal, "RevealBall", fade_attribute=None)
    )
    for frame in range(num_frames):
        if valid[frame]:
            ball.location = positions[frame].tolist()
            ball.keyframe_insert("location", frame=frame)
    _key_visibility(ball, valid)

    # Trail: trail_count shrinking spheres at past positions, baked to PC2.
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, subdivisions=2)
    template = bpy.context.active_object
    unit = np.array([v.co[:] for v in template.data.vertices])
    tri = np.array([p.vertices[:] for p in template.data.polygons])
    bpy.data.objects.remove(template, do_unlink=True)
    count, per = shot.trail_count, len(unit)
    scales = shot.ball_radius_m * 0.8 * (1.0 - np.arange(count) / count)
    frames = np.zeros((num_frames, count * per, 3), dtype=np.float32)
    for frame in range(num_frames):
        for k in range(count):
            src = frame - (k + 1) * shot.trail_stride
            ok = src >= 0 and valid[src : frame + 1].all()
            center = (
                positions[src]
                if ok
                else (positions[frame] if valid[frame] else np.zeros(3))
            )
            frames[frame, k * per : (k + 1) * per] = center + unit * (
                scales[k] if ok else 0.0
            )
    trail_pc2 = output / "ball_trail.pc2"
    _write_pc2(trail_pc2, frames)
    trail_faces = np.concatenate([tri + k * per for k in range(count)])
    trail = _mesh_object(
        "BallTrail",
        frames[0],
        trail_faces,
        ball_material(shot, reveal, "RevealTrail", fade_attribute="age"),
        smooth=True,
    )
    age = np.repeat(1.0 - np.arange(count) / count, per).astype(np.float32) * 0.8
    attr = trail.data.attributes.new("age", "FLOAT", "POINT")
    attr.data.foreach_set("value", age)
    mod = trail.modifiers.new("Trail", "MESH_CACHE")
    mod.cache_format = "PC2"
    mod.filepath = str(trail_pc2)
    mod.frame_start = 0.0
    _key_visibility(trail, valid)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--frames", default=None, help="start:end inclusive subset")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--resolution-percentage", type=int, default=100)
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.resolution_percentage <= 100:
        parser.error("--resolution-percentage must be in [1, 100]")
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be positive")
    if args.frame_step < 1:
        parser.error("--frame-step must be positive")
    shot = Shot() if args.samples is None else Shot(samples=args.samples)
    scene = build(args.bundle, args.output, shot)
    scene.render.resolution_percentage = args.resolution_percentage
    scene.frame_step = args.frame_step
    if args.frames is not None:
        start, end = (int(v) for v in args.frames.split(":"))
        if not scene.frame_start <= start <= end <= scene.frame_end:
            parser.error("--frames must be an inclusive range inside the scene")
        scene.frame_start, scene.frame_end = start, end
    scene.render.filepath = str(args.output / "frames" / "#####")
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "tennis_reveal.blend"))
    if args.render:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__" and "--" in sys.argv:
    main(sys.argv[sys.argv.index("--") + 1 :])

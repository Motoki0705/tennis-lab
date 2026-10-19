"""Shot-independent Blender helpers (Blender's Python + numpy only).

Import this sibling module from a shot script after adding its directory to
sys.path. Reveal timing, shader staging and camera animation stay in reveal.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Quaternion, Vector


class Nodes:
    """Terse node/link construction for shader and compositor trees."""

    def __init__(self, tree: bpy.types.NodeTree) -> None:
        self.tree = tree

    def new(self, kind: str, **inputs: Any) -> bpy.types.Node:
        node = self.tree.nodes.new(kind)
        for name, value in inputs.items():
            self.set(node.inputs[name], value)
        return node

    def set(self, socket: bpy.types.NodeSocket, value: Any) -> None:
        if isinstance(value, bpy.types.NodeSocket):
            self.tree.links.new(value, socket)
        else:
            socket.default_value = value

    def math(self, op: str, a: Any, b: Any = 0.0, *, clamp: bool = False) -> Any:
        node = self.tree.nodes.new("ShaderNodeMath")
        node.operation = op
        node.use_clamp = clamp
        self.set(node.inputs[0], a)
        self.set(node.inputs[1], b)
        return node.outputs[0]

    def mix_shader(self, fac: Any, a: Any, b: Any) -> Any:
        node = self.tree.nodes.new("ShaderNodeMixShader")
        self.set(node.inputs[0], fac)
        self.tree.links.new(a, node.inputs[1])
        self.tree.links.new(b, node.inputs[2])
        return node.outputs[0]

    def add_shader(self, a: Any, b: Any) -> Any:
        node = self.tree.nodes.new("ShaderNodeAddShader")
        self.tree.links.new(a, node.inputs[0])
        self.tree.links.new(b, node.inputs[1])
        return node.outputs[0]

    def emission(self, color: Any, strength: Any = 1.0) -> Any:
        node = self.new("ShaderNodeEmission", Strength=strength)
        self.set(node.inputs["Color"], color)
        return node.outputs[0]

    def rgb_scale(self, color: tuple[float, float, float], factor: Any) -> Any:
        node = self.tree.nodes.new("ShaderNodeVectorMath")
        node.operation = "SCALE"
        node.inputs[0].default_value = color
        self.set(node.inputs["Scale"], factor)
        return node.outputs[0]


def _group_socket(
    group: bpy.types.NodeTree, name: str, in_out: str, kind: str = "NodeSocketFloat"
) -> None:
    group.interface.new_socket(name=name, in_out=in_out, socket_type=kind)


def _new_material(
    name: str, *, transparent: bool
) -> tuple[bpy.types.Material, Nodes, Any]:
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    material.surface_render_method = "DITHERED"
    material.use_transparent_shadow = transparent
    return material, Nodes(tree), output.inputs["Surface"]


def _grid_lines(n: Nodes, position: Any, cell: float, half_width: float) -> Any:
    """1 on world-space grid lines of pitch ``cell`` (meters), else 0."""
    sep = n.new("ShaderNodeSeparateXYZ")
    n.set(sep.inputs[0], position)

    def axis(value: Any) -> Any:
        folded = n.math(
            "ABSOLUTE",
            n.math("SUBTRACT", n.math("FRACT", n.math("DIVIDE", value, cell)), 0.5),
        )
        # distance (m) to the nearest line = (0.5 - folded) * cell
        dist = n.math("MULTIPLY", n.math("SUBTRACT", 0.5, folded), cell)
        ramp = n.new("ShaderNodeMapRange", **{"From Min": half_width, "From Max": 0.0})
        ramp.interpolation_type = "SMOOTHSTEP"
        n.set(ramp.inputs["Value"], dist)
        return ramp.outputs[0]

    return n.math("MAXIMUM", axis(sep.outputs[0]), axis(sep.outputs[1]))


def _distance_fade(
    n: Nodes, position: Any, origin: Vector, near: float, far: float
) -> Any:
    vec = n.tree.nodes.new("ShaderNodeVectorMath")
    vec.operation = "DISTANCE"
    n.set(vec.inputs[0], position)
    vec.inputs[1].default_value = origin
    ramp = n.new(
        "ShaderNodeMapRange",
        **{"From Min": near, "From Max": far, "To Min": 1.0, "To Max": 0.0},
    )
    ramp.interpolation_type = "SMOOTHSTEP"
    n.set(ramp.inputs["Value"], vec.outputs["Value"])
    return ramp.outputs[0]


def _disk(radius: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rings = np.linspace(0.0, 1.0, 24)[1:] ** 2 * radius  # dense near the center
    verts = [(0.0, 0.0, 0.0)]
    for r in rings:
        verts += [(r * math.cos(a), r * math.sin(a), 0.0) for a in angles]
    faces: list[tuple[int, ...]] = []
    for i in range(segments):
        faces.append((0, 1 + i, 1 + (i + 1) % segments))
    for ring in range(len(rings) - 1):
        base, nxt = 1 + ring * segments, 1 + (ring + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((base + i, nxt + i, nxt + j, base + j))
    return np.asarray(verts), np.asarray(faces, dtype=object)


def _dome(radius: float, rings: int, segments: int) -> tuple[np.ndarray, np.ndarray]:
    verts = []
    for ring in range(rings + 1):
        phi = (ring / rings) * (math.pi / 2.0) - math.radians(4.0)  # dip below horizon
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            verts.append(
                (
                    radius * math.cos(phi) * math.cos(a),
                    radius * math.cos(phi) * math.sin(a),
                    radius * math.sin(phi),
                )
            )
    faces = []
    for ring in range(rings):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = ring * segments, (ring + 1) * segments
            faces.append((a + i, a + j, b + j, b + i))
    return np.asarray(verts), np.asarray(faces)


def court_image(
    court: dict[str, Any],
    *,
    court_color: tuple[float, float, float],
    runoff_color: tuple[float, float, float],
    px_per_m: float = 100.0,
) -> bpy.types.Image:
    """Opaque court surface colors with anti-aliased white lines."""
    xmin, xmax, ymin, ymax = court["runoff_xy"]
    width = int(round((xmax - xmin) * px_per_m))
    height = int(round((ymax - ymin) * px_per_m))
    xs = xmin + (np.arange(width) + 0.5) / px_per_m
    ys = ymin + (np.arange(height) + 0.5) / px_per_m
    gx, gy = np.meshgrid(xs, ys)
    lines = np.zeros_like(gx)
    half = 0.05 / 2.0
    for (ax, ay, _), (bx, by, _) in court["ground_lines"]:
        dx, dy = bx - ax, by - ay
        t = np.clip(((gx - ax) * dx + (gy - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        dist = np.hypot(gx - (ax + t * dx), gy - (ay + t * dy))
        lines = np.maximum(lines, np.clip((half - dist) * px_per_m + 0.5, 0.0, 1.0))
    xd = max(abs(p[0]) for seg in court["ground_lines"] for p in seg)
    yb = max(abs(p[1]) for seg in court["ground_lines"] for p in seg)
    inside = (np.abs(gx) <= xd) & (np.abs(gy) <= yb)
    rgb = np.where(inside[..., None], court_color, runoff_color)
    rgb = rgb * (1.0 - lines[..., None]) + 0.92 * lines[..., None]
    # Opaque: an alpha-coded line mask would be premultiplied into the color
    # when the generated image is packed into the .blend.
    rgba = np.concatenate([rgb, np.ones_like(lines)[..., None]], axis=-1)
    rgba = rgba.astype(np.float32)
    image = bpy.data.images.new(
        "CourtSurface", width, height, alpha=False, float_buffer=True
    )
    image.colorspace_settings.name = "Linear Rec.709"
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    return image


def _net_geometry(court: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = np.asarray(court["net"]["top"], dtype=np.float64)
    # Sheet (tape 0) and tape strip (tape 1) use separate vertices so the
    # attribute does not interpolate across the sheet.
    verts, faces, tape = [], [], []
    for x, h in top:
        verts += [(x, 0.0, 0.0), (x, 0.0, h - 0.065), (x, 0.0, h - 0.065), (x, 0.0, h)]
        tape += [0.0, 0.0, 1.0, 1.0]
    for i in range(len(top) - 1):
        a, b = 4 * i, 4 * (i + 1)
        faces.append((a, b, b + 1, a + 1))
        faces.append((a + 2, b + 2, b + 3, a + 3))
    return np.asarray(verts), np.asarray(faces), np.asarray(tape)


def _pc2_frame0(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.uint8, count=32)
    points = int(np.frombuffer(header[16:20].tobytes(), dtype="<i4")[0])
    data = np.fromfile(path, dtype="<f4", count=points * 3, offset=32)
    return data.reshape(points, 3)


def _write_pc2(path: Path, frames: np.ndarray) -> None:
    import struct

    frames = np.asarray(frames, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                "<12siiffi",
                b"POINTCACHE2\0",
                1,
                frames.shape[1],
                0.0,
                1.0,
                frames.shape[0],
            )
        )
        handle.write(np.ascontiguousarray(frames).tobytes())


def _look_rotation(direction: Vector, roll: float) -> Quaternion:
    return direction.to_track_quat("-Z", "Y") @ Quaternion((0.0, 0.0, 1.0), roll)


def _key_visibility(obj: bpy.types.Object, valid: np.ndarray) -> None:
    previous: bool | None = None
    for frame, is_valid in enumerate(valid.tolist()):
        if is_valid != previous:
            obj.hide_render = not is_valid
            obj.hide_viewport = not is_valid
            obj.keyframe_insert("hide_render", frame=frame)
            obj.keyframe_insert("hide_viewport", frame=frame)
            previous = is_valid


def _build_compositor(scene: bpy.types.Scene, *, name: str) -> None:
    tree = bpy.data.node_groups.new(name, "CompositorNodeTree")
    tree.interface.new_socket(
        name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
    )
    layers = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Bloom"
    glare.inputs["Quality"].default_value = "High"
    glare.inputs["Threshold"].default_value = 1.0
    glare.inputs["Strength"].default_value = 0.3
    glare.inputs["Size"].default_value = 0.6
    out = tree.nodes.new("NodeGroupOutput")
    tree.links.new(layers.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], out.inputs["Image"])
    scene.compositing_node_group = tree
    scene.render.use_compositing = True


def player_shader(n: Nodes, color: tuple[float, float, float]) -> Any:
    """Matte player color with the shared silhouette rim, without reveal staging."""
    # Matte flat color; a faint same-hue rim keeps the silhouette readable
    # against the dark world without a glossy "material" look.
    bsdf = n.new(
        "ShaderNodeBsdfPrincipled", Roughness=1.0, **{"Specular IOR Level": 0.0}
    )
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    facing = n.new("ShaderNodeLayerWeight", Blend=0.35)
    rim = n.math("POWER", facing.outputs["Facing"], 3.0)
    look = n.add_shader(
        bsdf.outputs[0],
        n.emission(n.rgb_scale(color, n.math("MULTIPLY", rim, 0.6))),
    )
    return look


def net_shader(n: Nodes) -> Any:
    """Translucent sheet and white tape selected by the mesh's ``tape`` attribute."""
    # The mesh cells are sub-pixel at shot distances, so the net is a dark
    # translucent sheet; the white tape is a per-vertex attribute (1 = tape).
    tape = n.new("ShaderNodeAttribute")
    tape.attribute_name = "tape"
    tape_band = n.new("ShaderNodeMapRange", **{"From Min": 0.45, "From Max": 0.55})
    n.set(tape_band.inputs["Value"], tape.outputs["Fac"])
    sheet = n.mix_shader(
        0.6,
        n.new("ShaderNodeBsdfTransparent").outputs[0],
        n.new(
            "ShaderNodeBsdfPrincipled",
            **{"Base Color": (0.015, 0.015, 0.015, 1.0), "Roughness": 1.0},
        ).outputs[0],
    )
    net_look = n.mix_shader(
        tape_band.outputs[0],
        sheet,
        n.new(
            "ShaderNodeBsdfPrincipled",
            **{"Base Color": (0.9, 0.9, 0.9, 1.0), "Roughness": 1.0},
        ).outputs[0],
    )
    return net_look


def ball_shader(
    n: Nodes, color: tuple[float, float, float], *, fade_attribute: str | None
) -> Any:
    """Emissive ball/trail look with optional per-vertex opacity."""
    look = n.emission((*color, 1.0), 3.0)
    if fade_attribute is not None:
        attr = n.new("ShaderNodeAttribute")
        attr.attribute_name = fade_attribute
        look = n.mix_shader(
            attr.outputs["Fac"], n.new("ShaderNodeBsdfTransparent").outputs[0], look
        )
    return look

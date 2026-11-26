"""Event overlays for the plate-free, fixed-camera court shot."""

from __future__ import annotations

import math
from typing import Any

import bpy
import numpy as np

from common import _key_visibility, _new_material

PLAYBACK_BUNDLE_SCHEMA = "tennis_scene_blender_playback_v1"
EVENT_HOLD_SECONDS = 0.4
EVENT_STYLES = {
    "shot": ("SHOT", (1.0, 0.36, 0.025)),
    "bounce": ("BOUNCE", (0.025, 0.67, 1.0)),
    "net": ("NET CONTACT", (1.0, 0.10, 0.46)),
}


def event_ring_name(kind: str, frame: int) -> str:
    return f"BallEvent.{kind}.{frame}"


def event_badge_name(kind: str) -> str:
    return f"EventBadge.{kind}"


def _emissive_material(
    name: str, color: tuple[float, float, float], *, fade: bool = False
) -> bpy.types.Material:
    material, nodes, surface = _new_material(name, transparent=True)
    transparent = nodes.new("ShaderNodeBsdfTransparent").outputs[0]
    shader = nodes.emission((*color, 1.0), 2.0 if fade else 1.0)
    if fade:
        info = nodes.new("ShaderNodeObjectInfo")
        shader = nodes.mix_shader(info.outputs["Alpha"], transparent, shader)
    # Annotation graphics should cast no shadows on the reconstructed court.
    shadow_ray = nodes.new("ShaderNodeLightPath").outputs["Is Shadow Ray"]
    nodes.tree.links.new(nodes.mix_shader(shadow_ray, shader, transparent), surface)
    return material


def _add_ring(
    collection: bpy.types.Collection,
    camera: bpy.types.Object,
    event: dict[str, Any],
    material: bpy.types.Material,
    visible: np.ndarray,
    last_frame: int,
) -> None:
    kind, frame = event["kind"], event["frame"]
    name = event_ring_name(kind, frame)
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.065
    curve.bevel_resolution = 2
    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    spline = curve.splines.new("POLY")
    spline.points.add(len(angles) - 1)
    for point, angle in zip(spline.points, angles, strict=True):
        point.co = (math.cos(angle), math.sin(angle), 0.0, 1.0)
    spline.use_cyclic_u = True
    curve.materials.append(material)
    ring = bpy.data.objects.new(name, curve)
    collection.objects.link(ring)
    ring.location = event["position"]
    if kind != "bounce":
        ring.rotation_euler = camera.rotation_euler
    ring.scale = (0.32,) * 3
    ring.color[3] = 1.0
    ring.keyframe_insert("scale", frame=frame)
    ring.keyframe_insert("color", index=3, frame=frame)
    if last_frame > frame:
        ring.scale = (0.78,) * 3
        ring.color[3] = 0.08
        ring.keyframe_insert("scale", frame=last_frame)
        ring.keyframe_insert("color", index=3, frame=last_frame)
    _key_visibility(ring, visible)


def _badge_quad(
    name: str,
    collection: bpy.types.Collection,
    camera: bpy.types.Object,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    depth: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [(0, 0, 0), (width, 0, 0), (width, height, 0), (0, height, 0)],
        [],
        [(0, 1, 2, 3)],
    )
    obj = bpy.data.objects.new(name, mesh)
    mesh.materials.append(material)
    collection.objects.link(obj)
    obj.parent = camera
    obj.location = (x, y, depth)
    return obj


def _add_badges(
    scene: bpy.types.Scene,
    collection: bpy.types.Collection,
    camera: bpy.types.Object,
    active: dict[str, np.ndarray],
) -> None:
    # Fixed slots keep simultaneous event types legible above the court.
    corners = camera.data.view_frame(scene=scene)
    left = min(corner.x for corner in corners)
    right = max(corner.x for corner in corners)
    bottom = min(corner.y for corner in corners)
    top = max(corner.y for corner in corners)
    width, height = right - left, top - bottom
    badge_height = height * 0.055
    gap = width * 0.009
    widths = {
        kind: width * (0.045 + 0.010 * len(label))
        for kind, (label, _) in EVENT_STYLES.items()
    }
    x = right - width * 0.038 - sum(widths.values()) - gap * (len(widths) - 1)
    y = top - height * 0.092
    background = _emissive_material("EventBadgeBackground", (0.011, 0.016, 0.024))
    text_color = _emissive_material("EventBadgeText", (0.91, 0.95, 1.0))
    for kind, (label, color) in EVENT_STYLES.items():
        badge_width = widths[kind]
        visible = active[kind]
        if visible.any():
            name = event_badge_name(kind)
            panel = _badge_quad(
                name,
                collection,
                camera,
                x=x,
                y=y,
                width=badge_width,
                height=badge_height,
                depth=-2.0,
                material=background,
            )
            accent = _badge_quad(
                f"{name}.Accent",
                collection,
                camera,
                x=x,
                y=y,
                width=badge_height * 0.12,
                height=badge_height,
                depth=-1.99,
                material=_emissive_material(f"{name}Color", color),
            )
            font = bpy.data.curves.new(f"{name}.Label", "FONT")
            font.body = label
            font.align_y = "CENTER"
            font.size = badge_height * 0.60
            font.materials.append(text_color)
            text = bpy.data.objects.new(f"{name}.Label", font)
            collection.objects.link(text)
            text.parent = camera
            text.location = (x + badge_height * 0.37, y + badge_height * 0.5, -1.98)
            for obj in (panel, accent, text):
                _key_visibility(obj, visible)
        x += badge_width + gap


def build_ball_events(
    scene: bpy.types.Scene,
    collection: bpy.types.Collection,
    events: list[dict[str, Any]],
    fps: float,
    num_frames: int,
) -> None:
    """Trigger each label and optional world-space ring at its source frame."""
    camera = scene.camera
    active = {kind: np.zeros(num_frames, dtype=bool) for kind in EVENT_STYLES}
    hold_frames = math.ceil(EVENT_HOLD_SECONDS * fps)
    ring_materials: dict[str, bpy.types.Material] = {}
    for event in events:
        kind, start = event["kind"], event["frame"]
        if kind not in EVENT_STYLES or not 0 <= start < num_frames:
            raise ValueError(f"Invalid playback event {kind!r} at frame {start}")
        stop = min(start + hold_frames, num_frames)
        active[kind][start:stop] = True
        if event["position"] is None:
            continue
        if kind not in ring_materials:
            ring_materials[kind] = _emissive_material(
                f"BallEventColor.{kind}", EVENT_STYLES[kind][1], fade=True
            )
        visible = np.zeros(num_frames, dtype=bool)
        visible[start:stop] = True
        _add_ring(collection, camera, event, ring_materials[kind], visible, stop - 1)
    _add_badges(scene, collection, camera, active)

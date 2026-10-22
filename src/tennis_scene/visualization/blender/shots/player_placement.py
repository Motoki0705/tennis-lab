"""Build the "triangulated player -> GVHMR body placement" Blender shot.

Runs inside Blender 5.2 (its bundled Python + numpy); it does not import the
repository.  The input is a ``tennis_scene_blender_placement_v1`` bundle written
by ``python -m src.tennis_scene.scripts.export_blender_placement``.

The shot shows the transform encoded by the placement trace,
``world = hip_position + scale * Rz(yaw) @ u_t``, as object keys on one mesh:

* ``Body`` is the PC2 mesh ``body_local.pc2`` (``u_t``), deformed by object
  scale / Z rotation / location.
* ``SourceJoints`` are 17 orange spheres parented to ``Body`` so they follow the
  same transform (``joints_local``).
* ``TargetJoints`` are the white triangulated COCO17 points in world space, with
  radii proportional to ``target_weights`` and the ``coco17_skeleton`` lines.
* ``Residuals`` are the yellow SourceJoint -> TargetJoint segments.

Stages are configured by :class:`PlacementShot`.  The output scene always runs
at 30 fps; the replay stage maps each output frame's elapsed time to the input
fps (``start + round(k * fps / 30)``) so a 60 fps source plays for its original
duration instead of at half speed.  During the staged part ``t0`` is held fixed
(the Mesh Cache ``frame_start`` key selects the same cache frame).  Invalid
frames hide ``Body`` and ``SourceJoints`` and are never interpolated over.

The camera is placed diagonally in front of the body: the horizontal forward is
``Rz(yaw_t0) @ (joints_local[11] - joints_local[12]) x +Z`` and the camera sits
``camera_azimuth_offset_deg`` (default 35) away from it.  This t0 forward remains
the azimuth reference for the whole shot; during replay the camera follows the
moving hip and adds only the slow orbit, never the per-frame body yaw/pose.  The
plate camera's calibration is kept for the intro image, but the shot's view
azimuth does not depend on the plate camera position.

``intro_plate=True`` prepends 3 seconds of the plate video as a full-frame
background (the same emissive Window-texture composition used by ``reveal.py``)
and shifts every stage later.  The plate sequence is re-sampled to 30 fps from
``manifest["frames"]["start"]``; a missing source frame is rejected instead of
being looped, duplicated or frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

BUNDLE_SCHEMA = "tennis_scene_blender_placement_v1"
OUTPUT_FPS = 30
TOLERANCE_M = 1e-4
COLLECTION = "TennisPlacement"
INTRO_PLATE_OBJECT = "IntroPlate"
JOINT_COUNT = 17

# Colors are specified as sRGB hex values and converted to linear for shaders.
def _srgb_channel_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _srgb_hex(value: str) -> tuple[float, float, float]:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected a #RRGGBB color, got {value!r}.")
    return tuple(
        _srgb_channel_to_linear(int(text[index : index + 2], 16) / 255.0)
        for index in (0, 2, 4)
    )


def _tuple3(value: Any, *, name: str) -> tuple[float, float, float]:
    items = tuple(value)
    if len(items) != 3:
        raise ValueError(f"{name} must have three components.")
    result = tuple(float(item) for item in items)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite.")
    return result  # type: ignore[return-value]


def _round_half_up(value: float) -> int:
    """Deterministic rounding used by the frame resampling (not banker's)."""
    return int(math.floor(value + 0.5))


def _blender() -> tuple[Any, Any]:
    """Import Blender and the sibling common module only when running there."""
    import bpy
    import common

    return bpy, common


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PlacementShot:
    """Timing (30 fps output frames) and look parameters of the placement shot.

    The six staged intervals are ``court_frames`` (0-60 equivalent),
    ``appear_frames`` (60-120), ``scale_frames`` (120-180), ``yaw_frames``
    (180-240), ``position_frames`` (240-300) and ``final_frames`` (300-330);
    the replay follows immediately.  Intervals are half-open and start at zero,
    so the default boundaries are exactly ``0, 60, 120, 180, 240, 300, 330``.
    """

    intro_plate: bool = False
    intro_seconds: float = 3.0
    court_frames: int = 60
    appear_frames: int = 60
    scale_frames: int = 60
    yaw_frames: int = 60
    position_frames: int = 60
    final_frames: int = 30
    display_offset_m: tuple[float, float, float] = (2.5, 0.0, 0.0)
    camera_distance_m: float = 9.0
    camera_height_m: float = 4.2
    camera_target_height_m: float = 0.2
    camera_azimuth_offset_deg: float = 35.0
    camera_orbit_deg: float = 45.0
    court_px_per_m: float = 100.0
    samples: int = 32
    world_color: tuple[float, float, float] = (0.012, 0.012, 0.012)
    court_color: tuple[float, float, float] = (0.03, 0.12, 0.34)
    runoff_color: tuple[float, float, float] = (0.018, 0.075, 0.095)
    body_color: tuple[float, float, float] = (0.62, 0.66, 0.72)
    source_joint_color: tuple[float, float, float] = _srgb_hex("#F4A261")
    target_joint_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    skeleton_color: tuple[float, float, float] = (0.9, 0.9, 0.9)
    residual_color: tuple[float, float, float] = _srgb_hex("#FFD166")
    source_joint_radius_m: float = 0.035
    target_joint_radius_m: float = 0.05
    skeleton_radius_m: float = 0.008
    residual_radius_m: float = 0.006

    def __post_init__(self) -> None:
        for name in (
            "court_frames",
            "appear_frames",
            "scale_frames",
            "yaw_frames",
            "position_frames",
            "final_frames",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(self.intro_seconds, bool)
            or not math.isfinite(float(self.intro_seconds))
            or self.intro_seconds <= 0.0
        ):
            raise ValueError("intro_seconds must be finite and positive.")
        if _round_half_up(self.intro_seconds * OUTPUT_FPS) < 1:
            raise ValueError("intro_seconds must cover at least one output frame.")
        object.__setattr__(
            self, "display_offset_m", _tuple3(self.display_offset_m, name="display_offset_m")
        )
        for name in (
            "world_color",
            "court_color",
            "runoff_color",
            "body_color",
            "source_joint_color",
            "target_joint_color",
            "skeleton_color",
            "residual_color",
        ):
            object.__setattr__(self, name, _tuple3(getattr(self, name), name=name))
        for name in (
            "camera_distance_m",
            "camera_height_m",
            "camera_target_height_m",
            "court_px_per_m",
            "source_joint_radius_m",
            "target_joint_radius_m",
            "skeleton_radius_m",
            "residual_radius_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(float(self.camera_azimuth_offset_deg)):
            raise ValueError("camera_azimuth_offset_deg must be finite.")
        if not math.isfinite(float(self.camera_orbit_deg)):
            raise ValueError("camera_orbit_deg must be finite.")
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("samples must be a positive integer.")

    @property
    def intro_frames(self) -> int:
        return _round_half_up(self.intro_seconds * OUTPUT_FPS)


@dataclass(frozen=True)
class PlacementTimeline:
    """Output-frame mapping derived from the shot and the bundle window."""

    shot: PlacementShot
    fps: float
    start: int
    stop: int
    t0: int
    intro_source_frames: tuple[int, ...]
    stages: tuple[tuple[str, int, int], ...]
    replay_source_frames: tuple[int, ...]
    frame_start: int
    frame_end: int

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("Timeline has no stages.")
        expected = self.frame_start
        for name, begin, end in self.stages:
            if begin != expected or end <= begin:
                raise ValueError(f"Stage {name!r} is not contiguous: {begin}:{end}.")
            expected = end
        if expected != self.frame_end + 1:
            raise ValueError("Timeline stages do not cover the scene frame range.")
        if self.stage_bounds()["replay"][1] - self.stage_bounds()["replay"][0] != len(
            self.replay_source_frames
        ):
            raise ValueError("Replay stage and source frame mapping disagree.")

    def stage_bounds(self) -> dict[str, tuple[int, int]]:
        return {name: (begin, end) for name, begin, end in self.stages}

    @property
    def intro_frames(self) -> int:
        return len(self.intro_source_frames)

    def stage_at(self, frame: int) -> str:
        if not self.frame_start <= frame <= self.frame_end:
            raise ValueError(f"Frame {frame} is outside {self.frame_start}:{self.frame_end}.")
        for name, begin, end in self.stages:
            if begin <= frame < end:
                return name
        raise ValueError(f"Frame {frame} is not covered by any stage.")

    def source_frame_at(self, frame: int) -> int:
        """Bundle source frame displayed at ``frame`` (``t0`` before replay)."""
        stage = self.stage_at(frame)
        if stage != "replay":
            return self.t0
        begin, _ = self.stage_bounds()["replay"]
        return self.replay_source_frames[frame - begin]

    def pc2_frame_at(self, frame: int) -> int:
        """Zero-based ``body_local.pc2`` frame used at ``frame``."""
        return self.source_frame_at(frame) - self.start


def replay_source_indices(start: int, stop: int, fps: float) -> tuple[int, ...]:
    """Map 30 fps output frames to source frames by their real elapsed time.

    Output frame ``k`` is at ``k / 30`` seconds after ``start``, so the source
    frame is ``start + round(k * fps / 30)``.  The output frame count is
    ``round((stop - start) * 30 / fps)`` (at least one), which keeps the replay
    duration within the 30 fps rounding error of the original.  The tuple is
    not stretched to force the last source frame: the final output frame may
    land on an earlier source frame when the original duration ends between
    output samples.  A mapped index that would leave ``[start, stop)`` is
    clamped to the last source frame explicitly; no interpolation or gap
    filling is ever performed.
    """
    length = stop - start
    if length <= 0:
        raise ValueError("frames must satisfy start < stop.")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("input fps must be finite and positive.")
    count = max(1, _round_half_up(length * OUTPUT_FPS / fps))
    indices = []
    for index in range(count):
        offset = _round_half_up(index * fps / OUTPUT_FPS)
        # The last output sample can fall just after the clip because the
        # output grid is rounded to whole 30 fps frames.  Clamp instead of
        # looping, holding an unrelated frame or reading out of bounds.
        offset = min(max(offset, 0), length - 1)
        indices.append(start + offset)
    return tuple(indices)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject a bundle manifest this shot cannot interpret unambiguously."""
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a mapping.")
    if manifest["schema"] != BUNDLE_SCHEMA:
        raise ValueError(
            f"Unsupported bundle schema {manifest['schema']!r}; "
            f"expected {BUNDLE_SCHEMA!r}."
        )
    fps = float(manifest["fps"])
    if not math.isfinite(fps) or not 1.0 <= fps <= 240.0:
        raise ValueError(f"Bundle fps must be finite and in [1, 240], got {fps}.")
    width, height = int(manifest["width"]), int(manifest["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Bundle width and height must be positive.")
    frames = manifest["frames"]
    start, stop = int(frames["start"]), int(frames["stop"])
    if not 0 <= start < stop:
        raise ValueError(f"frames must satisfy 0 <= start < stop, got {start}:{stop}.")
    source_frames = tuple(int(frame) for frame in frames["source_frames"])
    if source_frames != tuple(range(start, stop)):
        raise ValueError(
            "frames.source_frames must be the contiguous interval start..stop-1."
        )
    t0 = int(manifest["t0"])
    if not start <= t0 < stop:
        raise ValueError(f"t0={t0} must lie inside frames {start}:{stop}.")
    scale = float(manifest["scale"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Bundle scale must be finite and positive.")
    if not isinstance(manifest["files"], dict):
        raise ValueError("Bundle manifest must declare its files.")
    if not isinstance(manifest["plate"], dict):
        raise ValueError("Bundle manifest must declare its plate.")


def validate_t0(valid: np.ndarray, t0: int, start: int, stop: int) -> None:
    """Reject a representative frame that is outside the window or invalid."""
    if not start <= t0 < stop:
        raise ValueError(f"t0={t0} must lie inside frames {start}:{stop}.")
    index = t0 - start
    if index < 0 or index >= len(valid):
        raise ValueError("t0 is outside the valid mask.")
    if not bool(valid[index]):
        raise ValueError(
            f"t0={t0} is not a valid placement frame; "
            "the staged shot has no final placement to show."
        )


def build_timeline(
    manifest: dict[str, Any], shot: PlacementShot
) -> PlacementTimeline:
    """Assign every output frame to a stage and an explicit source frame."""
    validate_manifest(manifest)
    fps = float(manifest["fps"])
    start = int(manifest["frames"]["start"])
    stop = int(manifest["frames"]["stop"])
    t0 = int(manifest["t0"])
    intro_sources: tuple[int, ...] = ()
    stages: list[tuple[str, int, int]] = []
    cursor = 0
    if shot.intro_plate:
        required = int(math.ceil(shot.intro_seconds * fps - 1e-9))
        if stop - start < required:
            raise ValueError(
                f"intro_plate needs {required} source frames "
                f"({shot.intro_seconds:g}s at {fps:g}fps) from "
                f"frames.start={start}, but frames.stop={stop} covers only "
                f"{stop - start}."
            )
        intro_sources = tuple(
            start + _round_half_up(index * fps / OUTPUT_FPS)
            for index in range(shot.intro_frames)
        )
        if intro_sources[-1] >= stop:
            raise ValueError(
                "intro_plate requires source frames beyond the bundle window."
            )
        stages.append(("intro", 0, shot.intro_frames))
        cursor = shot.intro_frames
    for name, length in (
        ("court", shot.court_frames),
        ("appear", shot.appear_frames),
        ("scale", shot.scale_frames),
        ("yaw", shot.yaw_frames),
        ("position", shot.position_frames),
        ("final", shot.final_frames),
    ):
        stages.append((name, cursor, cursor + length))
        cursor += length
    replay = replay_source_indices(start, stop, fps)
    replay_stop = cursor + len(replay)
    stages.append(("replay", cursor, replay_stop))
    return PlacementTimeline(
        shot=shot,
        fps=fps,
        start=start,
        stop=stop,
        t0=t0,
        intro_source_frames=intro_sources,
        stages=tuple(stages),
        replay_source_frames=replay,
        frame_start=0,
        frame_end=replay_stop - 1,
    )


def body_visible_at(timeline: PlacementTimeline, frame: int, valid: np.ndarray) -> bool:
    """The single validity mask used by ``Body`` and every ``SourceJoint``."""
    timeline.stage_at(frame)
    appear_start, _ = timeline.stage_bounds()["appear"]
    if frame < appear_start:
        return False
    replay_start, _ = timeline.stage_bounds()["replay"]
    if frame < replay_start:
        index = timeline.t0 - timeline.start
    else:
        index = timeline.source_frame_at(frame) - timeline.start
    if not 0 <= index < len(valid):
        raise ValueError("Visibility mask does not cover the mapped source frame.")
    return bool(valid[index])


def body_transform_at(
    timeline: PlacementTimeline,
    frame: int,
    *,
    scale: float,
    t0_yaw: float,
    t0_hip: np.ndarray,
    anchor: np.ndarray,
    hip_position: np.ndarray,
    yaw_correction: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Object scale / Z rotation / location at one output frame."""
    bounds = timeline.stage_bounds()
    stage = timeline.stage_at(frame)
    if stage in {"intro", "court", "appear"}:
        return 1.0, 0.0, np.asarray(anchor, np.float64).copy()
    if stage == "scale":
        begin, end = bounds["scale"]
        progress = (frame - begin) / (end - begin)
        return 1.0 + (scale - 1.0) * progress, 0.0, np.asarray(anchor, np.float64).copy()
    if stage == "yaw":
        begin, end = bounds["yaw"]
        progress = (frame - begin) / (end - begin)
        return scale, t0_yaw * progress, np.asarray(anchor, np.float64).copy()
    if stage == "position":
        begin, end = bounds["position"]
        progress = (frame - begin) / (end - begin)
        location = anchor + (np.asarray(t0_hip, np.float64) - anchor) * progress
        return scale, t0_yaw, np.asarray(location, np.float64)
    if stage == "final":
        return scale, t0_yaw, np.asarray(t0_hip, np.float64).copy()
    if stage == "replay":
        index = timeline.source_frame_at(frame) - timeline.start
        return (
            scale,
            float(yaw_correction[index]),
            np.asarray(hip_position[index], np.float64).copy(),
        )
    raise ValueError(f"Unknown stage {stage!r}.")


def load_manifest(bundle: Path) -> dict[str, Any]:
    """Read the JSON manifest without interpreting its arrays."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def check_output_available(output: Path) -> None:
    """Refuse to overwrite a shot directory; failures stay failures."""
    if output.exists():
        raise FileExistsError(f"Shot output already exists: {output}")


def require_plate_frames(
    bundle: Path, manifest: dict[str, Any], timeline: PlacementTimeline
) -> None:
    """Reject missing intro plate frames instead of looping or duplicating."""
    if not timeline.shot.intro_plate:
        return
    plate = manifest["plate"]
    pattern = str(plate["pattern"])
    if "{frame" not in pattern:
        raise ValueError("plate.pattern must contain a {frame...} placeholder.")
    declared = tuple(int(frame) for frame in plate["frames"])
    missing: list[int] = []
    for source_frame in timeline.intro_source_frames:
        if source_frame not in declared:
            missing.append(source_frame)
            continue
        path = bundle / pattern.format(frame=source_frame)
        if not path.is_file():
            missing.append(source_frame)
    if missing:
        raise ValueError(
            "intro_plate requires plate frames that are not present in the "
            f"bundle: {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
            + f" (first={timeline.intro_source_frames[0]}, "
            f"last={timeline.intro_source_frames[-1]})."
        )


# --------------------------------------------------------------------------
# Blender scene construction
# --------------------------------------------------------------------------


def _pc2_header(path: Path) -> tuple[int, int]:
    header = np.fromfile(path, dtype=np.uint8, count=32)
    if header.size != 32:
        raise ValueError(f"{path} is too short to be a PC2 file.")
    points = int(np.frombuffer(header[16:20].tobytes(), dtype="<i4")[0])
    samples = int(np.frombuffer(header[28:32].tobytes(), dtype="<i4")[0])
    return points, samples


def _mesh_object(
    bpy: Any,
    collection: Any,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    material: Any,
    *,
    smooth: bool,
) -> Any:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.asarray(vertices).tolist(), [], np.asarray(faces).tolist())
    mesh.update()
    if smooth:
        mesh.shade_smooth()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _primitive_mesh(bpy: Any, kind: str, **kwargs: Any) -> Any:
    operator = getattr(bpy.ops.mesh, kind)
    operator(**kwargs)
    obj = bpy.context.active_object
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    mesh.shade_smooth()
    return mesh


def _shared_object(bpy: Any, collection: Any, name: str, mesh: Any) -> Any:
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _emission_material(
    common: Any, name: str, color: tuple[float, float, float], strength: float
) -> Any:
    material, nodes, surface = common._new_material(name, transparent=False)
    nodes.tree.links.new(nodes.emission((*color, 1.0), strength), surface)
    return material


def _body_material(common: Any, shot: PlacementShot) -> tuple[Any, Any]:
    material, nodes, surface = common._new_material(
        "PlacementBody", transparent=True
    )
    look = common.player_shader(nodes, shot.body_color)
    opacity = nodes.tree.nodes.new("ShaderNodeValue")
    opacity.name = "BodyOpacity"
    opacity.outputs[0].default_value = 1.0
    transparent = nodes.new("ShaderNodeBsdfTransparent").outputs[0]
    nodes.tree.links.new(
        nodes.mix_shader(opacity.outputs[0], transparent, look), surface
    )
    return material, opacity


def _court_material(common: Any, image: Any) -> Any:
    material, nodes, surface = common._new_material(
        "PlacementCourt", transparent=False
    )
    coordinates = nodes.new("ShaderNodeTexCoord")
    texture = nodes.tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.extension = "CLIP"
    texture.interpolation = "Cubic"
    nodes.tree.links.new(coordinates.outputs["UV"], texture.inputs["Vector"])
    shader = nodes.new(
        "ShaderNodeBsdfPrincipled", Roughness=1.0, **{"Specular IOR Level": 0.0}
    )
    nodes.set(shader.inputs["Base Color"], texture.outputs["Color"])
    nodes.set(shader.inputs["Emission Color"], (1.0, 1.0, 1.0, 1.0))
    nodes.set(shader.inputs["Emission Strength"], 0.18)
    nodes.tree.links.new(shader.outputs[0], surface)
    return material


def _net_material(common: Any) -> Any:
    material, nodes, surface = common._new_material("PlacementNet", transparent=True)
    nodes.tree.links.new(common.net_shader(nodes), surface)
    return material


def _dark_material(common: Any, name: str, color: tuple[float, float, float]) -> Any:
    material, nodes, surface = common._new_material(name, transparent=False)
    shader = nodes.new(
        "ShaderNodeBsdfPrincipled",
        **{"Base Color": (*color, 1.0), "Roughness": 1.0},
    )
    nodes.tree.links.new(shader.outputs[0], surface)
    return material


def _plate_material(common: Any, image: Any, intro_frames: int) -> Any:
    material, nodes, surface = common._new_material(
        "PlacementIntroPlate", transparent=False
    )
    coordinates = nodes.new("ShaderNodeTexCoord")
    texture = nodes.tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.extension = "EXTEND"
    texture.interpolation = "Linear"
    user = texture.image_user
    user.frame_duration = intro_frames
    user.frame_start = 0
    # Sequence file number = scene frame - frame_start + frame_offset + 1.
    user.frame_offset = -1
    user.use_auto_refresh = True
    nodes.tree.links.new(coordinates.outputs["Window"], texture.inputs["Vector"])
    nodes.tree.links.new(nodes.emission(texture.outputs["Color"]), surface)
    return material


def body_forward(yaw: float, joints_local: np.ndarray) -> np.ndarray:
    """Horizontal body-forward direction at ``yaw``.

    ``forward = Rz(yaw) @ (joints_local[11] - joints_local[12]) x +Z`` using
    the COCO17 left/right hips (11/12).  A zero or non-finite horizontal hip
    difference is rejected instead of falling back to the plate camera.
    """
    if not math.isfinite(float(yaw)):
        raise ValueError("Body yaw must be finite to determine its forward direction.")
    delta = np.asarray(joints_local[11], np.float64) - np.asarray(
        joints_local[12], np.float64
    )
    if not np.isfinite(delta).all():
        raise ValueError(
            "Left/right hip difference is non-finite; cannot determine the "
            "body forward direction."
        )
    world = _rotation_z(float(yaw)) @ delta
    forward = np.cross(world, np.asarray([0.0, 0.0, 1.0]))
    horizontal = np.asarray(forward[:2], np.float64)
    norm = float(np.linalg.norm(horizontal))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError(
            "Left/right hip difference has no horizontal component; cannot "
            "determine the body forward direction."
        )
    return np.asarray([horizontal[0] / norm, horizontal[1] / norm, 0.0])


def _camera_direction(
    shot: PlacementShot, forward: np.ndarray, progress: float
) -> np.ndarray:
    """Unit horizontal direction from the hip to the camera.

    The camera sits ``camera_azimuth_offset_deg`` away from the body's forward
    direction; during replay the slow orbit adds ``camera_orbit_deg * progress``.
    """
    vector = np.asarray(forward, np.float64)
    horizontal = np.asarray(vector[:2], np.float64)
    norm = float(np.linalg.norm(horizontal))
    if (
        not np.isfinite(horizontal).all()
        or not math.isfinite(norm)
        or norm <= 1e-9
    ):
        raise ValueError("Camera forward direction must be finite and horizontal.")
    unit = np.asarray([horizontal[0] / norm, horizontal[1] / norm, 0.0])
    angle = math.radians(
        shot.camera_azimuth_offset_deg + shot.camera_orbit_deg * float(progress)
    )
    direction = _rotation_z(angle) @ unit
    direction[2] = 0.0
    return direction


def _camera_pose(
    shot: PlacementShot,
    forward: np.ndarray,
    hip: np.ndarray,
    progress: float,
) -> tuple[Any, Any]:
    from mathutils import Vector

    direction = _camera_direction(shot, forward, progress)
    target = Vector(
        (
            float(hip[0]),
            float(hip[1]),
            float(hip[2]) + shot.camera_target_height_m,
        )
    )
    position = Vector(
        (
            float(hip[0]) + shot.camera_distance_m * float(direction[0]),
            float(hip[1]) + shot.camera_distance_m * float(direction[1]),
            shot.camera_height_m,
        )
    )
    common = _blender()[1]
    return position, common._look_rotation(target - position, 0.0)


def _write_intro_plate(
    bundle: Path,
    output: Path,
    manifest: dict[str, Any],
    timeline: PlacementTimeline,
) -> Path:
    pattern = str(manifest["plate"]["pattern"])
    destination = output / "intro_plate"
    destination.mkdir()
    for index, source_frame in enumerate(timeline.intro_source_frames):
        shutil.copyfile(
            bundle / pattern.format(frame=source_frame),
            destination / f"{index:05d}.jpg",
        )
    return destination


def _build_court(
    bpy: Any, common: Any, collection: Any, manifest: dict[str, Any], shot: PlacementShot
) -> None:
    court = manifest["court"]
    xmin, xmax, ymin, ymax = court["runoff_xy"]
    vertices = np.asarray(
        [[xmin, ymin, 0.0], [xmax, ymin, 0.0], [xmax, ymax, 0.0], [xmin, ymax, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    image = common.court_image(
        court,
        court_color=shot.court_color,
        runoff_color=shot.runoff_color,
        px_per_m=shot.court_px_per_m,
    )
    ground = _mesh_object(
        bpy, collection, "Court", vertices, faces, _court_material(common, image),
        smooth=False,
    )
    uv_layer = ground.data.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set(
        "uv", np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32).ravel()
    )

    net_vertices, net_faces, tape = common._net_geometry(court)
    net = _mesh_object(
        bpy, collection, "Net", net_vertices, net_faces, _net_material(common),
        smooth=False,
    )
    attribute = net.data.attributes.new("tape", "FLOAT", "POINT")
    attribute.data.foreach_set("value", np.asarray(tape, np.float32))
    post_material = _dark_material(common, "PlacementNetPost", (0.05, 0.05, 0.05))
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
        for owner in post.users_collection:
            owner.objects.unlink(post)
        collection.objects.link(post)
        post.data.materials.append(post_material)


def _build_lighting(bpy: Any, common: Any, scene: Any, shot: PlacementShot) -> None:
    world = bpy.data.worlds.new("PlacementWorld")
    scene.world = world
    world.node_tree.nodes.clear()
    nodes = common.Nodes(world.node_tree)
    background = nodes.new(
        "ShaderNodeBackground", Color=(*shot.world_color, 1.0), Strength=1.0
    )
    output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs[0], output.inputs["Surface"])
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 3.5
    light.angle = math.radians(1.5)
    sun = bpy.data.objects.new("Sun", light)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(38.0), 0.0, math.radians(-150.0))


def _build_body(
    bpy: Any,
    common: Any,
    collection: Any,
    scene: Any,
    manifest: dict[str, Any],
    arrays: dict[str, Any],
    timeline: PlacementTimeline,
    bundle: Path,
    output: Path,
    shot: PlacementShot,
    camera: Any,
) -> None:
    from mathutils import Matrix

    files = manifest["files"]
    pc2_path = bundle / str(files["body_local"])
    faces = arrays["smpl_faces"]
    base_vertices = common._pc2_frame0(pc2_path)
    body_material, opacity = _body_material(common, shot)
    body = _mesh_object(
        bpy, collection, "Body", base_vertices, faces, body_material, smooth=True
    )
    body.rotation_mode = "XYZ"
    cache = body.modifiers.new("BodyCache", "MESH_CACHE")
    cache.cache_format = "PC2"
    cache.filepath = str(pc2_path)
    cache.time_mode = "FRAME"
    cache.play_mode = "SCENE"
    cache.forward_axis, cache.up_axis = "POS_Y", "POS_Z"
    cache.interpolation = "LINEAR"
    cache.frame_start = 0.0
    cache.frame_scale = 1.0

    t0_index = timeline.t0 - timeline.start
    t0_hip = np.asarray(arrays["hip_position"][t0_index], np.float64)
    t0_yaw = float(arrays["yaw_correction"][t0_index])
    anchor = t0_hip + np.asarray(shot.display_offset_m, np.float64)
    t0_forward = body_forward(t0_yaw, arrays["joints_local"][t0_index])
    scale = float(manifest["scale"])
    valid = arrays["valid"]
    joints_local = arrays["joints_local"]
    hip_position = arrays["hip_position"]
    yaw_correction = arrays["yaw_correction"]
    target_joints = arrays["target_joints"]
    target_weights = arrays["target_weights"]
    has_target = np.isfinite(target_joints).all(-1) & (target_weights > 0.0)

    source_mesh = _primitive_mesh(
        bpy, "primitive_uv_sphere_add", segments=16, ring_count=8, radius=1.0
    )
    source_material = _emission_material(
        common, "PlacementSourceJoint", shot.source_joint_color, 3.0
    )
    source_mesh.materials.append(source_material)
    target_mesh = _primitive_mesh(
        bpy, "primitive_uv_sphere_add", segments=16, ring_count=8, radius=1.0
    )
    target_material = _emission_material(
        common, "PlacementTargetJoint", shot.target_joint_color, 2.5
    )
    target_mesh.materials.append(target_material)
    bone_mesh = _primitive_mesh(
        bpy, "primitive_cylinder_add", vertices=8, radius=1.0, depth=1.0
    )
    bone_material = _emission_material(
        common, "PlacementTargetBone", shot.skeleton_color, 1.5
    )
    bone_mesh.materials.append(bone_material)
    residual_mesh = _primitive_mesh(
        bpy, "primitive_cylinder_add", vertices=8, radius=1.0, depth=1.0
    )
    residual_material = _emission_material(
        common, "PlacementResidual", shot.residual_color, 3.0
    )
    residual_mesh.materials.append(residual_material)

    source_joints = []
    for joint in range(JOINT_COUNT):
        obj = _shared_object(
            bpy, collection, f"SourceJoint{joint:02d}", source_mesh
        )
        obj.parent = body
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.rotation_mode = "XYZ"
        source_joints.append(obj)
    target_joints_objects = []
    for joint in range(JOINT_COUNT):
        obj = _shared_object(
            bpy, collection, f"TargetJoint{joint:02d}", target_mesh
        )
        obj.rotation_mode = "XYZ"
        target_joints_objects.append(obj)

    skeleton = [tuple(int(value) for value in edge) for edge in manifest["coco17_skeleton"]]
    bones = [
        _shared_object(bpy, collection, f"TargetBone{index:02d}", bone_mesh)
        for index in range(len(skeleton))
    ]
    residuals = [
        _shared_object(bpy, collection, f"Residual{index:02d}", residual_mesh)
        for index in range(JOINT_COUNT)
    ]
    for obj in bones + residuals:
        obj.rotation_mode = "QUATERNION"

    total_frames = timeline.frame_end + 1
    body_visibility = np.zeros(total_frames, bool)
    for frame in range(total_frames):
        body_visibility[frame] = body_visible_at(timeline, frame, valid)

    joint_visibility = np.zeros((JOINT_COUNT, total_frames), bool)
    for frame in range(total_frames):
        if not body_visibility[frame]:
            continue
        index = timeline.pc2_frame_at(frame)
        finite = np.isfinite(joints_local[index]).all(-1)
        joint_visibility[:, frame] = finite
    common._key_visibility(body, body_visibility)
    for joint, obj in enumerate(source_joints):
        common._key_visibility(obj, joint_visibility[joint])

    opacity_frames = (
        (0, 0.5),
        (timeline.stage_bounds()["final"][0], 0.5),
        (timeline.stage_bounds()["replay"][0], 1.0),
    )
    for frame, value in opacity_frames:
        opacity.outputs[0].default_value = value
        opacity.outputs[0].keyframe_insert("default_value", frame=frame)

    target_visibility = np.zeros((JOINT_COUNT, total_frames), bool)
    for frame in range(total_frames):
        if frame < timeline.stage_bounds()["court"][0]:
            continue
        index = timeline.pc2_frame_at(frame)
        target_visibility[:, frame] = has_target[index]
    for joint, obj in enumerate(target_joints_objects):
        common._key_visibility(obj, target_visibility[joint])

    for frame in range(total_frames):
        body_scale, body_yaw, body_location = body_transform_at(
            timeline,
            frame,
            scale=scale,
            t0_yaw=t0_yaw,
            t0_hip=t0_hip,
            anchor=anchor,
            hip_position=hip_position,
            yaw_correction=yaw_correction,
        )
        body.scale = (body_scale, body_scale, body_scale)
        body.rotation_euler = (0.0, 0.0, body_yaw)
        body.location = body_location.tolist()
        body.keyframe_insert("scale", frame=frame)
        body.keyframe_insert("rotation_euler", frame=frame)
        body.keyframe_insert("location", frame=frame)
        cache.frame_start = frame - timeline.pc2_frame_at(frame)
        cache.keyframe_insert("frame_start", frame=frame)

        source_index = timeline.pc2_frame_at(frame)
        local = joints_local[source_index]
        joint_radius = shot.source_joint_radius_m / body_scale
        for joint, obj in enumerate(source_joints):
            point = local[joint]
            obj.location = (
                point.tolist() if np.isfinite(point).all() else (0.0, 0.0, 0.0)
            )
            obj.scale = (joint_radius, joint_radius, joint_radius)
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("scale", frame=frame)

        target_index = timeline.pc2_frame_at(frame)
        target = target_joints[target_index]
        weights = target_weights[target_index]
        for joint, obj in enumerate(target_joints_objects):
            radius = shot.target_joint_radius_m * float(weights[joint])
            point = target[joint]
            obj.location = (
                point.tolist() if np.isfinite(point).all() else (0.0, 0.0, 0.0)
            )
            obj.scale = (radius, radius, radius)
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("scale", frame=frame)

        source_world = body_location + body_scale * (
            _rotation_z(body_yaw) @ np.nan_to_num(local, nan=0.0).T
        ).T
        for index, (first, second) in enumerate(skeleton):
            begin = target[first]
            end = target[second]
            visible = bool(has_target[target_index, first] and has_target[target_index, second])
            _key_segment(
                bones[index],
                begin if visible else None,
                end if visible else None,
                shot.skeleton_radius_m,
                frame,
            )
        for joint, obj in enumerate(residuals):
            visible = bool(
                body_visibility[frame]
                and np.isfinite(local[joint]).all()
                and has_target[target_index, joint]
            )
            _key_segment(
                obj,
                source_world[joint] if visible else None,
                target[joint] if visible else None,
                shot.residual_radius_m,
                frame,
            )

    _animate_camera(
        bpy,
        common,
        scene,
        timeline,
        shot,
        camera,
        t0_forward,
        t0_hip,
        hip_position,
    )
    _build_intro_plate(
        bpy,
        common,
        collection,
        bundle,
        output,
        manifest,
        timeline,
        shot,
        camera,
    )


def _key_segment(
    obj: Any, start: np.ndarray | None, end: np.ndarray | None, radius: float, frame: int
) -> None:
    from mathutils import Vector

    if start is None or end is None:
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert("hide_render", frame=frame)
        obj.keyframe_insert("hide_viewport", frame=frame)
        return
    begin = Vector((float(start[0]), float(start[1]), float(start[2])))
    finish = Vector((float(end[0]), float(end[1]), float(end[2])))
    direction = finish - begin
    length = direction.length
    if length < 1e-9:
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert("hide_render", frame=frame)
        obj.keyframe_insert("hide_viewport", frame=frame)
        return
    obj.hide_render = False
    obj.hide_viewport = False
    obj.location = ((begin + finish) * 0.5).to_tuple()
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.scale = (radius, radius, length)
    obj.keyframe_insert("hide_render", frame=frame)
    obj.keyframe_insert("hide_viewport", frame=frame)
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("rotation_quaternion", frame=frame)
    obj.keyframe_insert("scale", frame=frame)


def _animate_camera(
    bpy: Any,
    common: Any,
    scene: Any,
    timeline: PlacementTimeline,
    shot: PlacementShot,
    camera: Any,
    t0_forward: np.ndarray,
    t0_hip: np.ndarray,
    hip_position: np.ndarray,
) -> None:
    replay_start = timeline.stage_bounds()["replay"][0]
    replay_stop = timeline.stage_bounds()["replay"][1]
    replay_count = replay_stop - replay_start
    camera.rotation_mode = "QUATERNION"
    for frame in range(timeline.frame_end + 1):
        if frame < replay_start:
            hip = t0_hip
            progress = 0.0
        else:
            progress = (frame - replay_start) / max(replay_count - 1, 1)
            index = timeline.source_frame_at(frame) - timeline.start
            hip = np.asarray(hip_position[index], np.float64)
        # The orbit is anchored to the t0 body forward for the whole shot; the
        # per-frame body yaw/pose must not steer the camera.
        position, rotation = _camera_pose(shot, t0_forward, hip, progress)
        camera.location = position
        camera.rotation_quaternion = rotation
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)
    scene.camera = camera


def _build_intro_plate(
    bpy: Any,
    common: Any,
    collection: Any,
    bundle: Path,
    output: Path,
    manifest: dict[str, Any],
    timeline: PlacementTimeline,
    shot: PlacementShot,
    camera: Any,
) -> None:
    from mathutils import Matrix

    if not shot.intro_plate:
        return
    directory = _write_intro_plate(bundle, output, manifest, timeline)
    image = bpy.data.images.load(str(directory / "00000.jpg"))
    image.source = "SEQUENCE"
    image.colorspace_settings.name = "sRGB"
    material = _plate_material(common, image, timeline.intro_frames)
    sensor_width = float(camera.data.sensor_width)
    half_width = 0.5 * sensor_width / float(camera.data.lens) * 1.02
    half_height = half_width * float(manifest["height"]) / float(manifest["width"])
    vertices = np.asarray(
        [
            [-half_width, -half_height, -1.0],
            [half_width, -half_height, -1.0],
            [half_width, half_height, -1.0],
            [-half_width, half_height, -1.0],
        ],
        dtype=np.float64,
    )
    plate = _mesh_object(
        bpy,
        collection,
        INTRO_PLATE_OBJECT,
        vertices,
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        material,
        smooth=False,
    )
    plate.parent = camera
    plate.matrix_parent_inverse = Matrix.Identity(4)
    visibility = np.zeros(timeline.frame_end + 1, bool)
    visibility[: timeline.intro_frames] = True
    common._key_visibility(plate, visibility)


def _load_arrays(
    bundle: Path, manifest: dict[str, Any], timeline: PlacementTimeline
) -> dict[str, Any]:
    files = manifest["files"]
    required = (
        "joints_local",
        "target_joints",
        "target_weights",
        "hip_position",
        "yaw_correction",
        "valid",
        "placed_vertices_t0",
        "smpl_faces",
        "body_local",
    )
    missing = [name for name in required if name not in files]
    if missing:
        raise ValueError(f"Bundle manifest is missing file entries: {missing}.")
    body_local_file = str(files["body_local"])
    arrays = {
        name: np.load(bundle / str(files[name]))
        for name in required
        if name != "body_local"
    }
    frame_count = timeline.stop - timeline.start
    joints = np.asarray(arrays["joints_local"], np.float64)
    target = np.asarray(arrays["target_joints"], np.float64)
    weights = np.asarray(arrays["target_weights"], np.float64)
    hip = np.asarray(arrays["hip_position"], np.float64)
    yaw = np.asarray(arrays["yaw_correction"], np.float64)
    valid = np.asarray(arrays["valid"], bool)
    faces = np.asarray(arrays["smpl_faces"])
    placed = np.asarray(arrays["placed_vertices_t0"], np.float64)
    if joints.shape != (frame_count, JOINT_COUNT, 3):
        raise ValueError(f"joints_local has shape {joints.shape}, expected "
                         f"{(frame_count, JOINT_COUNT, 3)}.")
    if target.shape != (frame_count, JOINT_COUNT, 3):
        raise ValueError(f"target_joints has shape {target.shape}.")
    if weights.shape != (frame_count, JOINT_COUNT):
        raise ValueError(f"target_weights has shape {weights.shape}.")
    if hip.shape != (frame_count, 3) or yaw.shape != (frame_count,):
        raise ValueError("hip_position/yaw_correction shapes disagree with frames.")
    if valid.shape != (frame_count,):
        raise ValueError(f"valid has shape {valid.shape}.")
    if not np.isfinite(joints).all():
        raise ValueError("joints_local contains non-finite coordinates.")
    if not np.isfinite(hip).all() or not np.isfinite(yaw).all():
        raise ValueError("hip_position/yaw_correction contain non-finite values.")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("target_weights must be finite and non-negative.")
    if not np.isfinite(placed).all():
        raise ValueError("placed_vertices_t0 contains non-finite values.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"smpl_faces has shape {faces.shape}.")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("smpl_faces must be an integer array.")
    if (weights > 0.0)[~np.isfinite(target).all(-1)].any():
        raise ValueError(
            "target_joints has non-finite coordinates where target_weights is positive."
        )
    validate_t0(valid, timeline.t0, timeline.start, timeline.stop)
    pc2_path = bundle / body_local_file
    points, samples = _pc2_header(pc2_path)
    if points != len(placed) or samples != frame_count:
        raise ValueError(
            f"body_local.pc2 has {samples} samples x {points} points, expected "
            f"{frame_count} x {len(placed)}."
        )
    if faces.size and (faces.min() < 0 or int(faces.max()) >= points):
        raise ValueError("smpl_faces reference vertices beyond body_local.pc2.")
    if len(placed) == 0:
        raise ValueError("placed_vertices_t0 is empty.")
    cached = np.memmap(
        pc2_path, dtype="<f4", mode="r", offset=32, shape=(frame_count, points, 3)
    )
    invalid_frames = ~valid
    if invalid_frames.any() and not np.all(np.asarray(cached[invalid_frames]) == 0.0):
        raise ValueError("body_local.pc2 must be zero on invalid frames.")
    if not np.isfinite(np.asarray(cached[valid])).all():
        raise ValueError("body_local.pc2 contains non-finite valid vertices.")
    del cached
    arrays.update(
        joints_local=joints,
        target_joints=target,
        target_weights=weights,
        hip_position=hip,
        yaw_correction=yaw,
        valid=valid,
        smpl_faces=faces,
        placed_vertices_t0=placed,
    )
    return arrays


def _build_scene(
    bpy: Any,
    common: Any,
    scene: Any,
    manifest: dict[str, Any],
    arrays: dict[str, Any],
    timeline: PlacementTimeline,
    bundle: Path,
    output: Path,
    shot: PlacementShot,
) -> None:
    scene.render.fps = OUTPUT_FPS
    scene.render.fps_base = 1.0
    scene.frame_start, scene.frame_end = timeline.frame_start, timeline.frame_end
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
    collection = bpy.data.collections.new(COLLECTION)
    scene.collection.children.link(collection)

    blender_camera = manifest["camera"]["blender"]
    camera_data = bpy.data.cameras.new("InputCamera")
    camera_data.sensor_fit = blender_camera["sensor_fit"]
    camera_data.sensor_width = blender_camera["sensor_width_mm"]
    camera_data.lens = blender_camera["lens_mm"]
    camera_data.clip_start, camera_data.clip_end = 0.1, 1000.0
    camera = bpy.data.objects.new("InputCamera", camera_data)
    collection.objects.link(camera)

    _build_court(bpy, common, collection, manifest, shot)
    _build_lighting(bpy, common, scene, shot)

    _build_body(
        bpy,
        common,
        collection,
        scene,
        manifest,
        arrays,
        timeline,
        bundle,
        output,
        shot,
        camera,
    )
    common._build_compositor(scene, name="PlacementComposite")
    scene["placement_bundle"] = str(bundle.resolve())
    scene["placement_shot"] = json.dumps(asdict(shot))
    scene["placement_timeline"] = json.dumps(
        {
            "fps": timeline.fps,
            "start": timeline.start,
            "stop": timeline.stop,
            "t0": timeline.t0,
            "stages": [[name, begin, end] for name, begin, end in timeline.stages],
            "replay_source_frames": list(timeline.replay_source_frames),
            "intro_source_frames": list(timeline.intro_source_frames),
        }
    )


def build(bundle: Path, output: Path, shot: PlacementShot) -> Any:
    """Build the placement scene from ``bundle``; caches live under ``output``."""
    bpy, common = _blender()
    manifest = load_manifest(bundle)
    timeline = build_timeline(manifest, shot)
    check_output_available(output)
    require_plate_frames(bundle, manifest, timeline)
    arrays = _load_arrays(bundle, manifest, timeline)
    validate_t0(arrays["valid"], timeline.t0, timeline.start, timeline.stop)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    output.mkdir(parents=True)
    _build_scene(
        bpy, common, scene, manifest, arrays, timeline, bundle, output, shot
    )
    scene.frame_set(0)
    return scene


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intro-plate", action="store_true")
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
    shot = PlacementShot(intro_plate=args.intro_plate)
    if args.samples is not None:
        shot = replace(shot, samples=args.samples)
    scene = build(args.bundle, args.output, shot)
    scene.render.resolution_percentage = args.resolution_percentage
    scene.frame_step = args.frame_step
    if args.frames is not None:
        start, end = (int(value) for value in args.frames.split(":"))
        if not scene.frame_start <= start <= end <= scene.frame_end:
            parser.error("--frames must be an inclusive range inside the scene")
        scene.frame_start, scene.frame_end = start, end
    scene.render.filepath = str(args.output / "frames" / "#####")
    bpy, _ = _blender()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "tennis_placement.blend"))
    if args.render:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__" and "--" in sys.argv:
    main(sys.argv[sys.argv.index("--") + 1 :])

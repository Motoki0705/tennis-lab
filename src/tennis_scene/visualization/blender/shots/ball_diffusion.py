"""Display exported concept geometry in the final state of a saved rays shot.

Blender bundled Python only. Numerical fitting, noise and mesh caches are
prepared by the exporter. The input .blend and its delivery remain read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ball_rays import _key_value, _material, _object, _rgb  # noqa: E402
from common import _key_visibility, _new_material, _pc2_frame0  # noqa: E402


def verify_assets(bundle: Path, manifest: dict) -> None:
    for name, digest in manifest["assets"].items():
        with (bundle / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != digest:
                raise ValueError(f"Bundle asset changed: {name}")


def _sphere(name: str, radius: float, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def _badge(bundle: Path, camera: bpy.types.Object) -> None:
    badge = _object(
        "ConceptBadge", np.load(bundle / "badge_vertices.npy").tolist(), [(0, 1, 2, 3)]
    )
    badge.parent = camera
    uv = badge.data.uv_layers.new()
    for loop, value in zip(uv.data, [(0, 0), (1, 0), (1, 1), (0, 1)], strict=True):
        loop.uv = value
    material, nodes, surface = _new_material("ConceptBadge", transparent=True)
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(bundle / "badge.png"))
    nodes.set(
        surface,
        nodes.mix_shader(
            texture.outputs["Alpha"],
            nodes.new("ShaderNodeBsdfTransparent").outputs[0],
            nodes.emission(texture.outputs["Color"], 1.0),
        ),
    )
    material.surface_render_method = "BLENDED"
    material.use_transparency_overlap = False
    badge.data.materials.append(material)
    badge["label"] = "構想"


def build(
    bundle: Path, source_blend: Path, output: Path, *, samples: int | None
) -> bpy.types.Scene:
    if output.exists():
        raise FileExistsError(f"Shot output already exists: {output}")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema"] != "tennis_scene_ball_diffusion_concept_v1":
        raise ValueError("Unsupported concept bundle")
    verify_assets(bundle, manifest)
    with source_blend.open("rb") as file:
        if (
            hashlib.file_digest(file, "sha256").hexdigest()
            != manifest["camera_continuity"]["source_blend_sha256"]
        ):
            raise ValueError("Source blend changed since export")
    bpy.ops.wm.open_mainfile(filepath=str(source_blend), load_ui=False)
    scene = bpy.context.scene
    # Relocate platform-specific external references before evaluating the end.
    for obj in list(scene.objects):
        for modifier in obj.modifiers:
            if modifier.type == "MESH_CACHE":
                modifier.filepath = str(
                    source_blend.parent
                    / Path(modifier.filepath.replace("\\", "/")).name
                )
    scene.frame_set(manifest["camera_continuity"]["source_output_frame"])
    matrix = np.array(scene.camera.matrix_world)
    if np.max(np.abs(matrix - manifest["camera_continuity"]["matrix_world"])) > 1e-5:
        raise AssertionError(
            "Source final camera disagrees with exported overview pose"
        )
    if abs(scene.camera.data.lens - manifest["camera_continuity"]["lens_mm"]) > 1e-5:
        raise AssertionError("Source final lens disagrees with manifest")
    snapshot = {
        "camera_matrix_world": matrix.tolist(),
        "camera_lens": scene.camera.data.lens,
        "source_blend_sha256": manifest["camera_continuity"]["source_blend_sha256"],
        "source_output_frame": scene.frame_current,
        "frusta": {
            c["id"]: np.array(
                bpy.data.objects[f"CameraFrustum_{c['id']}"].matrix_world
            ).tolist()
            for c in manifest["cameras"]
        },
    }
    # Animation removal preserves the evaluated last-state values.
    for obj in list(scene.objects):
        obj.animation_data_clear()
        if obj.data is not None:
            obj.data.animation_data_clear()
        if obj.name in ("Ball", "BallTrail") or obj.name.startswith(
            ("Ray_", "Detection_")
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        if material.node_tree:
            material.node_tree.animation_data_clear()
    for view, cam in enumerate(manifest["cameras"]):
        texture = bpy.data.materials[f"Plate_{view}"].node_tree.nodes["PlateSequence"]
        texture.image = bpy.data.images.load(str(bundle / cam["plate"]))
        texture.image.source = "FILE"
    config = manifest["config"]
    output.mkdir(parents=True)
    try:
        scene.render.fps, scene.render.fps_base = manifest["fps"], 1.0
        scene.frame_start, scene.frame_end = 0, manifest["num_frames"] - 1
        scene.render.resolution_x, scene.render.resolution_y = (
            config["width"],
            config["height"],
        )
        scene.eevee.taa_render_samples = (
            config["samples"] if samples is None else samples
        )
        condition = np.load(bundle / "condition_position.npy")
        state = np.load(bundle / "condition_state.npy")
        trajectory = np.load(bundle / "trajectory.npy")
        colors = ["#FFD166", "#4CC9F0", "#E63946"]
        materials = [
            _material(f"ConditionMaterial_{i}", _rgb(color), strength=1.5)
            for i, color in enumerate(colors)
        ]
        templates = []
        for i in range(3):
            radius = config["gap_radius_m"] if i == 0 else config["condition_radius_m"]
            obj = _sphere(f"ConditionTemplate_{i}", radius, materials[i])
            if i == 0:
                wire = obj.modifiers.new("MissingPositionIsSchematic", "WIREFRAME")
                wire.thickness = 0.025
            templates.append(obj)
        for index, kind in enumerate(state):
            obj = templates[int(kind)].copy()
            obj.name = f"Condition_{index}"
            scene.collection.objects.link(obj)
            obj.location = (
                trajectory[index] if kind == 0 else condition[index]
            ).tolist()
            obj["condition_state"] = int(kind)
            obj["origin"] = "schematic" if kind == 0 else "measured"
        for template in templates:
            bpy.data.objects.remove(template, do_unlink=True)
        curve = _object(
            "CompletionCurve",
            _pc2_frame0(bundle / "process.pc2").tolist(),
            np.load(bundle / "tube_faces.npy").tolist(),
        )
        modifier = curve.modifiers.new("ExportedProcess", "MESH_CACHE")
        modifier.cache_format, modifier.filepath, modifier.frame_start = (
            "PC2",
            str(bundle / "process.pc2"),
            0.0,
        )
        material = _material("CompletionMaterial", _rgb("#BC8FFF"), strength=2.0)
        curve.data.materials.append(material)
        _key_visibility(curve, np.load(bundle / "curve_visible.npy"))
        ball = _sphere(
            "CompletedBall",
            config["ball_radius_m"],
            _material("CompletedBallMaterial", _rgb("#E8FF70"), strength=2.5),
        )
        for frame, (position, alpha) in enumerate(
            zip(
                np.load(bundle / "ball_position.npy"),
                np.load(bundle / "alpha.npy"),
                strict=True,
            )
        ):
            ball.location = position.tolist()
            ball.keyframe_insert("location", frame=frame)
            color = (1 - alpha) * np.array(_rgb("#BC8FFF")) + alpha * np.array(
                _rgb("#E8FF70")
            )
            _key_value(material, "Color", (*color, 1.0), frame)
        _key_visibility(ball, np.load(bundle / "ball_visible.npy"))
        _badge(bundle, scene.camera)
        for action in bpy.data.actions:
            for layer in action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for curve in bag.fcurves:
                            for point in curve.keyframe_points:
                                point.interpolation = "CONSTANT"
        scene["concept_bundle_schema"] = manifest["schema"]
        scene["concept_origins"] = json.dumps(manifest["origins"])
        scene["source_continuity"] = json.dumps(snapshot)
        (output / "shot_manifest.json").write_text(
            json.dumps(
                {
                    "schema": manifest["schema"],
                    "source_continuity": snapshot,
                    "stages": manifest["stages"],
                    "origins": manifest["origins"],
                    "samples": scene.eevee.taa_render_samples,
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
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--resolution-percentage", type=int, default=100)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    if (
        not 1 <= args.resolution_percentage <= 100
        or args.frame_step < 1
        or (args.samples is not None and args.samples < 1)
    ):
        parser.error("Require resolution 1..100 and positive samples/frame-step")
    bundle, output = args.bundle.resolve(), args.output.resolve()
    scene = build(bundle, args.source_blend.resolve(), output, samples=args.samples)
    scene.render.resolution_percentage = args.resolution_percentage
    scene.frame_step = args.frame_step
    scene.render.filepath = str(output / "frames/#####")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "ball_diffusion.blend"))
    if args.render:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__" and "--" in sys.argv:
    main(sys.argv[sys.argv.index("--") + 1 :])

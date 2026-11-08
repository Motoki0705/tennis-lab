"""Measure Blender Z/Position and linear premultiplied Combined on known geometry.

Uses bpy 4.5.3 on Colab, Cycles CPU, one camera, two tiny frames. This is not a
production renderer and does not assume that Z and Position are antialiased alike.
"""

import argparse
import json
from pathlib import Path

import bpy
import numpy as np


def read_image(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    a = np.asarray(image.pixels[:], dtype=np.float32).reshape(
        image.size[1], image.size[0], 4
    )
    bpy.data.images.remove(image)
    return a[::-1].copy()  # Blender bottom-up -> NHT top-down


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 2
    scene.render.resolution_x, scene.render.resolution_y = 32, 18
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    layer = scene.view_layers[0]
    layer.use_pass_z = True
    layer.use_pass_position = True
    data = bpy.data.cameras.new("camera")
    data.sensor_fit, data.sensor_width, data.lens = "HORIZONTAL", 36, 22.5  # fx=20
    data.clip_start, data.clip_end = 0.1, 100
    camera = bpy.data.objects.new("camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera  # identity camera looks along -Z
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0.13, 0.07, -4))
    plane = bpy.context.object
    plane.scale = (2.1, 1.3, 1)
    material = bpy.data.materials.new("constant-linear-emission")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.25, 0.5, 0.75, 1)
    surface = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs[0], surface.inputs["Surface"])
    plane.data.materials.append(material)
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    render = nodes.new("CompositorNodeRLayers")
    for name in ("Image", "Depth", "Position"):
        file = nodes.new("CompositorNodeOutputFile")
        file.base_path = str(output)
        file.format.file_format = "OPEN_EXR"
        file.format.color_depth = "32"
        file.format.color_mode = "RGBA"
        file.file_slots[0].path = name + "-"
        scene.node_tree.links.new(render.outputs[name], file.inputs[0])
    scene.frame_set(1)
    bpy.ops.render.render()
    rgba = read_image(output / "Image-0001.exr")
    depth = read_image(output / "Depth-0001.exr")[..., 0]
    position = read_image(output / "Position-0001.exr")[..., :3]
    valid = rgba[..., 3] > 0.999
    ray_error = float(
        np.max(np.abs(depth[valid] - np.linalg.norm(position[valid], axis=-1)))
    )
    z_error = float(np.max(np.abs(depth[valid] - 4)))
    if min(ray_error, z_error) > 1e-3:
        raise AssertionError("Neither ray distance nor camera z explains Blender Z")
    yy, xx = np.mgrid[:18, :32]
    expected_ray = np.stack(((xx + 0.5 - 16) / 20, -(yy + 0.5 - 9) / 20), axis=-1)
    observed_ray = position[..., :2] / 4
    report = {
        "schema_version": 1,
        "blender_version": bpy.app.version_string,
        "engine": "CYCLES",
        "device": "CPU",
        "resolution": [32, 18],
        "camera_z_m": 4,
        "single_sample_valid_pixels": int(valid.sum()),
        "z_pass_vs_ray_distance_max_error_m": ray_error,
        "z_pass_vs_camera_z_max_error_m": z_error,
        "pixel_center_ray_max_error_px": float(
            np.max(np.abs(observed_ray[valid] - expected_ray[valid])) * 20
        ),
        "z_pass_interpretation": "ray_distance" if ray_error < z_error else "camera_z",
    }
    scene.cycles.samples = 16
    scene.frame_set(2)
    bpy.ops.render.render()
    rgba = read_image(output / "Image-0002.exr")
    alpha = rgba[..., 3:]
    expected = alpha * np.asarray([0.25, 0.5, 0.75])
    color_error = float(np.max(np.abs(rgba[..., :3] - expected)))
    report["combined_vs_premultiplied_linear_max_error"] = color_error
    report["partial_coverage_pixels"] = int(((alpha > 0) & (alpha < 1)).sum())
    if color_error > 1e-3 or report["partial_coverage_pixels"] == 0:
        raise AssertionError(
            "Combined did not verify premultiplied linear color at edges"
        )
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

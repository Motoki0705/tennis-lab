"""Reusable camera frusta and image planes for Blender shots.

Kept beside common.py so new shots can reuse these without changing the frozen
R2 helpers. No repository imports; coordinates use OpenCV camera axes.
"""

from __future__ import annotations

from typing import Any

import bpy
import numpy as np
from mathutils import Vector

from common import _new_material


def image_plane_point(
    camera: dict[str, Any], uv_px: Any, *, depth: float = 1.5
) -> Vector:
    """OpenCV pixels -> camera-local image plane, using original width/height."""
    width, height = camera["width"], camera["height"]
    corners = (
        np.array(
            [
                [0.0, 0.0, 1.0],
                [width, 0.0, 1.0],
                [width, height, 1.0],
                [0.0, height, 1.0],
            ]
        )
        @ np.linalg.inv(np.array(camera["K"])).T
        * depth
    )
    u, v = np.array(uv_px) / [width, height]
    return Vector(
        corners[0] + u * (corners[1] - corners[0]) + v * (corners[3] - corners[0])
    )


def camera_frustum(
    camera: dict[str, Any], *, color: tuple[float, float, float], depth: float = 1.5
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Return (court-space camera root, sequence-ready image plane).

    The root's local axes are OpenCV camera axes (+X right, +Y down, +Z
    forward). Children, including image_plane_point(), use those axes. No ball
    geometry or triangulation is computed by this display helper.
    """
    from mathutils import Matrix

    name = camera["id"]
    root = bpy.data.objects.new(f"CameraFrustum_{name}", None)
    bpy.context.scene.collection.objects.link(root)
    transform = np.eye(4)
    transform[:3, :3] = np.array(camera["R"]).T
    transform[:3, 3] = camera["center"]
    root.matrix_world = Matrix(transform)
    corners = [
        image_plane_point(camera, pixel, depth=depth)
        for pixel in [
            (0, 0),
            (camera["width"], 0),
            (camera["width"], camera["height"]),
            (0, camera["height"]),
        ]
    ]
    mesh = bpy.data.meshes.new(f"ImagePlane_{name}")
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    mesh.update()
    plane = bpy.data.objects.new(f"ImagePlane_{name}", mesh)
    bpy.context.scene.collection.objects.link(plane)
    plane.parent = root
    uv = mesh.uv_layers.new()
    for loop, value in zip(
        uv.data, [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)], strict=True
    ):
        loop.uv = value
    material, n, surface = _new_material(f"FrustumMaterial_{name}", transparent=False)
    n.set(surface, n.emission((*color, 1.0), 1.5))
    curve = bpy.data.curves.new(f"FrustumLines_{name}", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth, curve.bevel_resolution = 0.012, 2
    segments = [((0.0, 0.0, 0.0), corner) for corner in corners]
    segments += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    for a, b in segments:
        line = curve.splines.new("POLY")
        line.points.add(1)
        line.points[0].co = (*a, 1.0)
        line.points[1].co = (*b, 1.0)
    wires = bpy.data.objects.new(f"FrustumLines_{name}", curve)
    bpy.context.scene.collection.objects.link(wires)
    wires.parent = root
    wires.data.materials.append(material)
    return root, plane

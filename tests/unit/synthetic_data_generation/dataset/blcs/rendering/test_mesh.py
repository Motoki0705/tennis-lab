"""Focused tests for depth-correct BLCS mesh rasterization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.synthetic_data_generation.dataset.blcs.mesh_asset import BLCSBallMesh
from src.synthetic_data_generation.dataset.blcs.rendering.mesh import (
    BLCSMeshRasterizer,
    _ray_triangle_intersections,
)
from src.synthetic_data_generation.dataset.blcs.timeline import build_blcs_plans
from src.synthetic_data_generation.dataset.runtime import BackgroundArrays


def test_mesh_rasterizer_emits_metric_surface_depth_and_instance_pixels(
    tmp_path: Path,
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    plan = build_blcs_plans(
        (blcs_trajectory_factory("trajectory-0", frame_count=1),),
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=7,
        chunk_size_frames=1,
    )[0]
    mesh = _octahedron_mesh(tmp_path / "mesh.glb", radius=1.0)
    rasterizer = BLCSMeshRasterizer(
        mesh=mesh,
        device="cpu",
        ambient=1.0,
        diffuse=0.0,
    )
    camera = plan.camera_rig.cameras[0].scene_camera
    background = _background(camera.camera_id, camera.width, camera.height, depth=100.0)

    delta = rasterizer.render_sample(
        plan=plan,
        source_frame_index=0,
        camera_index=0,
        background=background,
    )

    assert len(delta.pixel_indices) > 0
    assert np.all(delta.depth > 0.0)
    assert np.all(delta.depth < 100.0)
    np.testing.assert_array_equal(delta.instance_ids, 1)
    np.testing.assert_array_equal(delta.alpha, 1.0)
    assert np.all(delta.rgb[:, 0] > delta.rgb[:, 1])


def test_nht_background_depth_occludes_mesh_pixels(
    tmp_path: Path,
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    plan = build_blcs_plans(
        (blcs_trajectory_factory("trajectory-0", frame_count=1),),
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=7,
        chunk_size_frames=1,
    )[0]
    rasterizer = BLCSMeshRasterizer(
        mesh=_octahedron_mesh(tmp_path / "mesh.glb", radius=1.0),
        device="cpu",
    )
    camera = plan.camera_rig.cameras[0].scene_camera
    foreground = rasterizer.render_sample(
        plan=plan,
        source_frame_index=0,
        camera_index=0,
        background=_background(
            camera.camera_id,
            camera.width,
            camera.height,
            depth=100.0,
        ),
    )
    occluded = rasterizer.render_sample(
        plan=plan,
        source_frame_index=0,
        camera_index=0,
        background=_background(
            camera.camera_id,
            camera.width,
            camera.height,
            depth=0.1,
        ),
    )

    assert len(foreground.pixel_indices) > 0
    assert len(occluded.pixel_indices) == 0


def test_triangle_z_buffer_selects_the_nearest_mesh_surface() -> None:
    vertices = torch.tensor(
        (
            (-1.0, -1.0, 2.0),
            (1.0, -1.0, 2.0),
            (0.0, 1.0, 2.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=torch.float32,
    )
    faces = torch.tensor(((0, 1, 2), (3, 4, 5)), dtype=torch.int64)

    depths, face_indices, barycentric = _ray_triangle_intersections(
        torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32),
        vertices,
        faces,
    )

    assert depths.tolist() == [1.0]
    assert face_indices.tolist() == [1]
    np.testing.assert_allclose(barycentric.numpy().sum(axis=1), 1.0, atol=1.0e-6)


def _octahedron_mesh(path: Path, *, radius: float) -> BLCSBallMesh:
    path.write_bytes(b"test mesh source")
    vertices = radius * np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(
        (
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        ),
        dtype=np.int64,
    )
    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    colors = np.repeat(
        np.asarray(((0.8, 0.1, 0.05),), dtype=np.float32),
        len(vertices),
        axis=0,
    )
    return BLCSBallMesh(
        vertices_m=vertices,
        faces=faces,
        normals=normals.astype(np.float32),
        colors_linear_rgb=colors,
        source_vertex_count=len(vertices),
        source_face_count=len(faces),
        source_path=path.resolve(),
    )


def _background(
    camera_id: str,
    width: int,
    height: int,
    *,
    depth: float,
) -> BackgroundArrays:
    return BackgroundArrays(
        camera_id=camera_id,
        rgb=np.zeros((height, width, 3), dtype=np.float32),
        alpha=np.ones((height, width, 1), dtype=np.float32),
        depth=np.full((height, width, 1), depth, dtype=np.float32),
    )

"""Tests for src/utils/rendering/mesh_renderer.py."""

from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.rendering.mesh_renderer import MeshRenderer, MeshStyle

Float32Array: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]

def make_camera_K(width: int = 64, height: int = 64, f: float = 64.0) -> Float32Array:
    return np.array(  # type: ignore[no-any-return]
        [[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def make_square_mesh(z: float, half_size: float = 0.4) -> Float32Array:
    """Two triangles forming a square centred on the optical axis at depth z."""
    return np.array(  # type: ignore[no-any-return]
        [
            [-half_size, -half_size, z],
            [half_size, -half_size, z],
            [half_size, half_size, z],
            [-half_size, half_size, z],
        ],
        dtype=np.float32,
    )


SQUARE_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


class TestMeshRendererOverlay:
    def test_draws_mesh_in_image_center(self) -> None:
        renderer = MeshRenderer(SQUARE_FACES, MeshStyle(color=(1.0, 0.0, 0.0), alpha=1.0))
        image: UInt8Array = np.zeros((64, 64, 3), dtype=np.uint8)
        out = renderer.render_overlay(image, make_square_mesh(z=1.0), make_camera_K())

        assert out.shape == image.shape
        assert out.dtype == np.uint8
        # Center pixel covered by the mesh: red channel dominant.
        center = out[32, 32]
        assert center[0] > 100
        assert center[1] == 0 and center[2] == 0
        # Corner pixel untouched.
        assert (out[0, 0] == 0).all()
        # Input image is not modified in place.
        assert np.all(image == 0)

    def test_painter_occlusion_near_mesh_wins(self) -> None:
        # Two stacked squares: red at z=1 (near), green at z=2 (far).
        vertices = np.concatenate(
            [make_square_mesh(z=1.0), make_square_mesh(z=2.0)], axis=0
        )
        faces = np.concatenate([SQUARE_FACES, SQUARE_FACES + 4], axis=0)
        style = MeshStyle(color=(1.0, 0.0, 0.0), alpha=1.0, ambient=1.0, diffuse=0.0)
        renderer = MeshRenderer(faces, style)
        image: UInt8Array = np.zeros((64, 64, 3), dtype=np.uint8)

        out = renderer.render_overlay(image, vertices, make_camera_K())
        # The near square fully covers the far one at the image center; with
        # ambient-only shading the pixel must be exactly the base color.
        assert out[32, 32, 0] == 255

    def test_behind_camera_faces_skipped(self) -> None:
        renderer = MeshRenderer(SQUARE_FACES)
        image: UInt8Array = np.full((64, 64, 3), 7, dtype=np.uint8)
        out = renderer.render_overlay(image, make_square_mesh(z=-1.0), make_camera_K())
        np.testing.assert_array_equal(out, image)

    def test_alpha_blending(self) -> None:
        style = MeshStyle(color=(1.0, 1.0, 1.0), alpha=0.5, ambient=1.0, diffuse=0.0)
        renderer = MeshRenderer(SQUARE_FACES, style)
        image: UInt8Array = np.zeros((64, 64, 3), dtype=np.uint8)
        out = renderer.render_overlay(image, make_square_mesh(z=1.0), make_camera_K())
        # 50% blend of white mesh over black image.
        assert 120 <= out[32, 32, 0] <= 135

    def test_shading_depends_on_orientation(self) -> None:
        style = MeshStyle(color=(1.0, 1.0, 1.0), alpha=1.0, ambient=0.0, diffuse=1.0)
        renderer = MeshRenderer(SQUARE_FACES, style)
        K = make_camera_K()
        image: UInt8Array = np.zeros((64, 64, 3), dtype=np.uint8)

        # Frontal square: normals along z -> full intensity.
        frontal = renderer.render_overlay(image, make_square_mesh(z=1.0), K)

        # Tilted square (rotated ~60 deg about y): lower intensity.
        verts = make_square_mesh(z=0.0)
        angle = np.deg2rad(60.0)
        rot = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float32,
        )
        tilted_verts = verts @ rot.T + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        tilted = renderer.render_overlay(image, tilted_verts, K)

        assert frontal[32, 32, 0] > tilted[32, 32, 0] > 0

    def test_invalid_inputs_raise(self) -> None:
        renderer = MeshRenderer(SQUARE_FACES)
        image: UInt8Array = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="vertices_cam"):
            renderer.render_overlay(
                image, np.zeros((4, 2), dtype=np.float32), make_camera_K()
            )
        with pytest.raises(ValueError, match="K must"):
            renderer.render_overlay(
                image, make_square_mesh(1.0), np.eye(4, dtype=np.float32)
            )
        with pytest.raises(ValueError, match="image must"):
            renderer.render_overlay(
                np.zeros((8, 8), dtype=np.uint8), make_square_mesh(1.0), make_camera_K()
            )

    def test_invalid_faces_raise(self) -> None:
        with pytest.raises(ValueError, match="faces must"):
            MeshRenderer(np.zeros((4, 2), dtype=np.int64))


class TestMeshRenderer3D:
    def test_render_3d_adds_collection(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        renderer = MeshRenderer(SQUARE_FACES)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        before = len(ax.collections)
        renderer.render_3d(ax, make_square_mesh(z=1.0))
        assert len(ax.collections) == before + 1
        plt.close(fig)

    def test_render_3d_invalid_vertices(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        renderer = MeshRenderer(SQUARE_FACES)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with pytest.raises(ValueError, match="vertices must"):
            renderer.render_3d(ax, np.zeros((4, 2), dtype=np.float32))
        plt.close(fig)

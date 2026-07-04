"""Triangle-mesh renderer for camera-view overlays and 3D axes.

This module renders triangle meshes (e.g. SMPL bodies) without a GPU
rasterizer dependency:

- ``MeshRenderer.render_overlay``: paint a mesh given in *camera coordinates*
  (x right, y down, z forward) onto an image using pinhole intrinsics ``K``.
  Uses the painter's algorithm (depth-sorted ``cv2.fillConvexPoly``) with a
  Lambertian headlight shading, so no pytorch3d / pyrender is required.
- ``MeshRenderer.render_3d``: draw the mesh into a matplotlib 3D axis as a
  ``Poly3DCollection``.

Example:
    >>> renderer = MeshRenderer(faces)
    >>> out = renderer.render_overlay(frame_rgb, vertices_cam, K)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray


@dataclass
class MeshStyle:
    """Style configuration for mesh rendering.

    Attributes:
        color: Base mesh color as RGB in [0, 1].
        alpha: Blend factor of the mesh over the input image (overlay only).
        ambient: Ambient light intensity in [0, 1].
        diffuse: Diffuse (headlight) intensity; ``ambient + diffuse <= 1``
            keeps shading within the base color range.
        alpha_3d: Face alpha used by :meth:`MeshRenderer.render_3d`.
    """

    color: tuple[float, float, float] = (0.65, 0.74, 0.86)
    alpha: float = 0.9
    ambient: float = 0.45
    diffuse: float = 0.55
    alpha_3d: float = 0.6


class MeshRenderer:
    """Render a fixed-topology triangle mesh (overlay and matplotlib 3D)."""

    def __init__(
        self,
        faces: NDArray[np.integer],
        style: MeshStyle | None = None,
    ) -> None:
        """
        Args:
            faces: Triangle vertex indices, shape (F, 3).
            style: Rendering style; defaults to :class:`MeshStyle`.
        """
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
        self.faces: NDArray[np.int64] = faces.astype(np.int64)
        self.style = style or MeshStyle()

    def _face_shading(
        self,
        triangles: NDArray[np.float32],
        color: tuple[float, float, float],
    ) -> NDArray[np.float32]:
        """Per-face RGB colors (F, 3) from a Lambertian headlight model.

        The light direction is the camera viewing axis, and the absolute value
        of the normal-light dot product is used so the result does not depend
        on the face winding convention.
        """
        e1 = triangles[:, 1] - triangles[:, 0]
        e2 = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normals = normals / norms

        # Headlight along +z (camera view axis); |cos| makes it winding-agnostic.
        cos_theta = np.abs(normals[:, 2])
        intensity = self.style.ambient + self.style.diffuse * cos_theta  # (F,)
        return intensity[:, None] * np.asarray(color, dtype=np.float32)[None, :]

    def render_overlay(
        self,
        image: NDArray[np.uint8],
        vertices_cam: NDArray[np.float32],
        K: NDArray[np.float32],
        *,
        color: tuple[float, float, float] | None = None,
        alpha: float | None = None,
    ) -> NDArray[np.uint8]:
        """Render the mesh over an image from the camera viewpoint.

        Args:
            image: Input image (H, W, 3) uint8 (any channel order; the mesh
                color is interpreted in the same order as the image).
            vertices_cam: Vertices in camera coordinates (V, 3); x right,
                y down, z forward (towards the scene).
            K: Pinhole intrinsics (3, 3).
            color: Optional per-call override of the style color.
            alpha: Optional per-call override of the style alpha.

        Returns:
            New image (H, W, 3) uint8 with the mesh blended in. Faces with any
            vertex at or behind the camera plane (z <= 1e-6) are skipped.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
        vertices_cam = np.asarray(vertices_cam, dtype=np.float32)
        if vertices_cam.ndim != 2 or vertices_cam.shape[1] != 3:
            raise ValueError(
                f"vertices_cam must have shape (V, 3), got {vertices_cam.shape}"
            )
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {K.shape}")
        color = color if color is not None else self.style.color
        alpha = alpha if alpha is not None else self.style.alpha

        height, width = image.shape[:2]

        # Project all vertices with the pinhole model.
        z = vertices_cam[:, 2]
        z_safe = np.where(z > 1e-6, z, 1.0)
        u = K[0, 0] * (vertices_cam[:, 0] / z_safe) + K[0, 2]
        v = K[1, 1] * (vertices_cam[:, 1] / z_safe) + K[1, 2]
        uv = np.stack([u, v], axis=-1)  # (V, 2)

        # Valid faces: all vertices in front of the camera.
        vertex_valid = z > 1e-6
        face_valid = vertex_valid[self.faces].all(axis=1)  # (F,)

        # Skip faces fully outside the image.
        face_uv = uv[self.faces]  # (F, 3, 2)
        inside = (
            (face_uv[..., 0].max(axis=1) >= 0)
            & (face_uv[..., 0].min(axis=1) < width)
            & (face_uv[..., 1].max(axis=1) >= 0)
            & (face_uv[..., 1].min(axis=1) < height)
        )
        face_valid &= inside
        if not face_valid.any():
            return image.copy()

        faces = self.faces[face_valid]
        face_uv = face_uv[face_valid]
        triangles = vertices_cam[faces]  # (F, 3, 3)

        # Painter's algorithm: draw far faces first.
        depth = triangles[..., 2].mean(axis=1)
        order = np.argsort(-depth)

        shading = self._face_shading(triangles, color)  # (F, 3) in [0, 1]
        shading_u8 = np.clip(shading * 255.0, 0.0, 255.0).astype(np.uint8)

        mesh_layer = image.copy()
        face_uv_i32 = np.round(face_uv).astype(np.int32)
        for idx in order:
            face_color = shading_u8[idx]
            cv2.fillConvexPoly(
                mesh_layer,
                face_uv_i32[idx],
                (int(face_color[0]), int(face_color[1]), int(face_color[2])),
                lineType=cv2.LINE_AA,
            )

        out = cv2.addWeighted(mesh_layer, alpha, image, 1.0 - alpha, 0.0)
        return np.asarray(out, dtype=np.uint8)

    def render_3d(
        self,
        ax: Axes3D,
        vertices: NDArray[np.float32],
        *,
        color: str | tuple[float, float, float] | None = None,
        alpha: float | None = None,
    ) -> None:
        """Add the mesh to a matplotlib 3D axis as a ``Poly3DCollection``.

        Args:
            ax: Target 3D axis.
            vertices: Vertices (V, 3) in the axis coordinate system.
            color: Face color (matplotlib color); defaults to the style color.
            alpha: Face alpha; defaults to ``style.alpha_3d``.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape (V, 3), got {vertices.shape}")

        triangles = vertices[self.faces]
        mesh = Poly3DCollection(
            triangles,
            alpha=alpha if alpha is not None else self.style.alpha_3d,
            facecolor=color if color is not None else self.style.color,
            edgecolor="none",
            linewidths=0.0,
        )
        ax.add_collection3d(mesh)

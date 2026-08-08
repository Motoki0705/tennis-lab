"""Build SMPL-surface Gaussian assets and deform their covariance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _float_array(value: object, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return array


def _int_array(value: object, *, name: str, ndim: int) -> IntArray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array with {ndim} dimensions.")
    return array.astype(np.int64, copy=False)


def _readonly_float(array: FloatArray) -> FloatArray:
    result = np.ascontiguousarray(array, dtype=np.float64)
    result.setflags(write=False)
    return result


def _readonly_int(array: IntArray) -> IntArray:
    result = np.ascontiguousarray(array, dtype=np.int64)
    result.setflags(write=False)
    return result


def _rotation_matrices_to_quaternions(matrices: FloatArray) -> FloatArray:
    flat = matrices.reshape(-1, 3, 3)
    quaternions = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(flat):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
        else:
            diagonal = np.diag(matrix)
            axis = int(np.argmax(diagonal))
            other = (axis + 1) % 3
            last = (axis + 2) % 3
            scale = (
                np.sqrt(
                    1.0 + matrix[axis, axis] - matrix[other, other] - matrix[last, last]
                )
                * 2.0
            )
            quaternion = np.empty(4, dtype=np.float64)
            quaternion[0] = (matrix[last, other] - matrix[other, last]) / scale
            quaternion[axis + 1] = 0.25 * scale
            quaternion[other + 1] = (matrix[other, axis] + matrix[axis, other]) / scale
            quaternion[last + 1] = (matrix[last, axis] + matrix[axis, last]) / scale
        quaternion /= np.linalg.norm(quaternion)
        quaternions[index] = quaternion if quaternion[0] >= 0.0 else -quaternion
    return quaternions.reshape(*matrices.shape[:-2], 4)


@dataclass(frozen=True)
class AvatarGaussianAsset:
    """Canonical Gaussian geometry with explicit SMPL joint control weights."""

    means_m: FloatArray
    quaternions_wxyz: FloatArray
    log_scales_m: FloatArray
    opacity_logits: FloatArray
    point_joint_weights: FloatArray
    face_indices: IntArray
    barycentric_coordinates: FloatArray

    def __post_init__(self) -> None:
        means = _float_array(self.means_m, name="means_m", ndim=2)
        count = means.shape[0]
        if means.shape != (count, 3) or count == 0:
            raise ValueError("means_m must have non-empty shape [N,3].")
        quaternions = _float_array(
            self.quaternions_wxyz,
            name="quaternions_wxyz",
            ndim=2,
        )
        scales = _float_array(self.log_scales_m, name="log_scales_m", ndim=2)
        opacities = _float_array(
            self.opacity_logits,
            name="opacity_logits",
            ndim=1,
        )
        weights = _float_array(
            self.point_joint_weights,
            name="point_joint_weights",
            ndim=2,
        )
        faces = _int_array(self.face_indices, name="face_indices", ndim=1)
        barycentric = _float_array(
            self.barycentric_coordinates,
            name="barycentric_coordinates",
            ndim=2,
        )
        if quaternions.shape != (count, 4):
            raise ValueError("quaternions_wxyz must have shape [N,4].")
        if scales.shape != (count, 3):
            raise ValueError("log_scales_m must have shape [N,3].")
        if opacities.shape != (count,):
            raise ValueError("opacity_logits must have shape [N].")
        if weights.shape[0] != count or weights.shape[1] == 0:
            raise ValueError("point_joint_weights must have shape [N,J].")
        if faces.shape != (count,) or barycentric.shape != (count, 3):
            raise ValueError("Every Gaussian must have one triangle attachment.")
        if np.any(faces < 0) or np.any(barycentric < 0.0):
            raise ValueError("Triangle attachments must be non-negative.")
        if not np.allclose(barycentric.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0):
            raise ValueError("Barycentric coordinates must sum to one.")
        if np.any(weights < 0.0) or not np.allclose(
            weights.sum(axis=1),
            1.0,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError("point_joint_weights must be an explicit simplex.")
        if not np.allclose(
            np.linalg.norm(quaternions, axis=1),
            1.0,
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError("quaternions_wxyz must be normalized.")
        for name in (
            "means_m",
            "quaternions_wxyz",
            "log_scales_m",
            "opacity_logits",
            "point_joint_weights",
            "barycentric_coordinates",
        ):
            object.__setattr__(self, name, _readonly_float(getattr(self, name)))
        object.__setattr__(self, "face_indices", _readonly_int(self.face_indices))

    @property
    def gaussian_count(self) -> int:
        return int(self.means_m.shape[0])


def build_surface_gaussian_asset(
    canonical_vertices_m: object,
    *,
    faces: object,
    vertex_joint_weights: object,
    gaussian_count: int,
    seed: int,
    tangential_sigma_multiplier: float = 1.15,
    normal_sigma_ratio: float = 0.18,
    opacity: float = 0.97,
) -> AvatarGaussianAsset:
    """Sample an area-weighted, metric Gaussian shell from an SMPL mesh."""
    vertices = _float_array(
        canonical_vertices_m,
        name="canonical_vertices_m",
        ndim=2,
    )
    face_array = _int_array(faces, name="faces", ndim=2)
    joint_weights = _float_array(
        vertex_joint_weights,
        name="vertex_joint_weights",
        ndim=2,
    )
    if vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError("canonical_vertices_m must have non-empty shape [V,3].")
    if face_array.shape[1] != 3 or face_array.shape[0] == 0:
        raise ValueError("faces must have non-empty shape [F,3].")
    if np.any(face_array < 0) or np.any(face_array >= vertices.shape[0]):
        raise ValueError("faces contain an out-of-range vertex index.")
    if joint_weights.shape[0] != vertices.shape[0] or joint_weights.shape[1] == 0:
        raise ValueError("vertex_joint_weights must have shape [V,J].")
    if np.any(joint_weights < 0.0) or not np.allclose(
        joint_weights.sum(axis=1),
        1.0,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("vertex_joint_weights must be an explicit simplex.")
    if isinstance(gaussian_count, bool) or gaussian_count <= 0:
        raise ValueError("gaussian_count must be a positive integer.")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not np.isfinite(tangential_sigma_multiplier) or tangential_sigma_multiplier <= 0:
        raise ValueError("tangential_sigma_multiplier must be positive.")
    if not np.isfinite(normal_sigma_ratio) or not 0.0 < normal_sigma_ratio < 1.0:
        raise ValueError("normal_sigma_ratio must lie in (0,1).")
    if not np.isfinite(opacity) or not 0.0 < opacity < 1.0:
        raise ValueError("opacity must lie in (0,1).")

    triangles = vertices[face_array]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edges_a, edges_b)
    double_areas = np.linalg.norm(normals, axis=1)
    if np.any(double_areas <= 1.0e-12):
        raise ValueError("faces contain a degenerate triangle.")
    areas = double_areas * 0.5
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        face_array.shape[0],
        size=gaussian_count,
        replace=True,
        p=areas / areas.sum(),
    ).astype(np.int64)
    first = np.sqrt(rng.random(gaussian_count))
    second = rng.random(gaussian_count)
    barycentric = np.stack(
        (
            1.0 - first,
            first * (1.0 - second),
            first * second,
        ),
        axis=1,
    )
    selected_triangles = triangles[selected]
    means = np.einsum("nk,nkd->nd", barycentric, selected_triangles)
    point_weights = np.einsum(
        "nk,nkj->nj",
        barycentric,
        joint_weights[face_array[selected]],
    )

    tangents = edges_a[selected]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    selected_normals = normals[selected] / double_areas[selected, None]
    bitangents = np.cross(selected_normals, tangents)
    rotations = np.stack((tangents, bitangents, selected_normals), axis=-1)
    quaternions = _rotation_matrices_to_quaternions(rotations)

    tangent_sigma = (
        np.sqrt(float(areas.sum()) / (np.pi * gaussian_count))
        * tangential_sigma_multiplier
    )
    scales = np.broadcast_to(
        np.log(
            np.asarray(
                [
                    tangent_sigma,
                    tangent_sigma,
                    tangent_sigma * normal_sigma_ratio,
                ],
                dtype=np.float64,
            )
        ),
        (gaussian_count, 3),
    ).copy()
    opacity_logit = np.log(opacity / (1.0 - opacity))
    return AvatarGaussianAsset(
        means_m=means,
        quaternions_wxyz=quaternions,
        log_scales_m=scales,
        opacity_logits=np.full(gaussian_count, opacity_logit),
        point_joint_weights=point_weights,
        face_indices=selected,
        barycentric_coordinates=barycentric,
    )

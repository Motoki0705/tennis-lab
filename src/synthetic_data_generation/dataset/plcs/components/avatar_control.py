"""Geometric control algorithms for SMPL-driven Gaussian avatars."""

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


def _integer_array(value: object, *, name: str, ndim: int) -> IntArray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype.")
    return array.astype(np.int64, copy=False)


def _readonly_float(array: FloatArray) -> FloatArray:
    result = np.ascontiguousarray(array, dtype=np.float64)
    result.setflags(write=False)
    return result


def _readonly_int(array: IntArray) -> IntArray:
    result = np.ascontiguousarray(array, dtype=np.int64)
    result.setflags(write=False)
    return result


def _validate_simplex(weights: FloatArray, *, name: str) -> None:
    if weights.shape[-1] == 0:
        raise ValueError(f"{name} must have at least one weight.")
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    sums = weights.sum(axis=-1)
    # SMPL-X stores LBS weights as float32; their exact float64 sums can differ
    # from one by roughly 4.5e-8. Accept that representation error without
    # normalizing or repairing caller input.
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{name} must sum to one; no implicit normalization is used.")


def _validate_faces(
    faces: object,
    *,
    vertex_count: int,
) -> IntArray:
    face_array = _integer_array(faces, name="faces", ndim=2)
    if face_array.shape[1] != 3:
        raise ValueError("faces must have shape [F,3].")
    if face_array.size == 0:
        raise ValueError("faces must not be empty.")
    if np.any(face_array < 0) or np.any(face_array >= vertex_count):
        raise ValueError("faces contain an out-of-range vertex index.")
    return face_array


def _validate_attachments(
    *,
    face_count: int,
    face_indices: object,
    barycentric_coordinates: object,
) -> tuple[IntArray, FloatArray]:
    indices = _integer_array(face_indices, name="face_indices", ndim=1)
    barycentric = _float_array(
        barycentric_coordinates,
        name="barycentric_coordinates",
        ndim=2,
    )
    if barycentric.shape != (indices.shape[0], 3):
        raise ValueError(
            "barycentric_coordinates must have shape [N,3] matching face_indices."
        )
    if np.any(indices < 0) or np.any(indices >= face_count):
        raise ValueError("face_indices contain an out-of-range face.")
    _validate_simplex(barycentric, name="barycentric_coordinates")
    return indices, barycentric


def _validate_joint_transforms(value: object) -> FloatArray:
    transforms = _float_array(value, name="joint_transforms", ndim=4)
    if transforms.shape[-2:] != (4, 4) or transforms.shape[1] == 0:
        raise ValueError("joint_transforms must have shape [T,J,4,4].")
    expected_bottom = np.asarray([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(
        transforms[..., 3, :],
        expected_bottom,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("joint_transforms must be affine homogeneous matrices.")
    rotations = transforms[..., :3, :3]
    identities = np.einsum("...ji,...jk->...ik", rotations, rotations)
    if not np.allclose(
        identities,
        np.eye(3),
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise ValueError("Every joint transform must contain a proper rotation.")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, rtol=0.0, atol=2.0e-5):
        raise ValueError("Joint rotations must not reflect or scale.")
    return transforms


def _validate_translations(
    value: object | None,
    *,
    frame_count: int,
) -> FloatArray:
    if value is None:
        return np.zeros((frame_count, 3), dtype=np.float64)
    translations = _float_array(value, name="translations_m", ndim=2)
    if translations.shape != (frame_count, 3):
        raise ValueError("translations_m must have shape [T,3].")
    return translations


@dataclass(frozen=True)
class NeighborBlend:
    """Deterministic template-vertex neighbors and their explicit weights."""

    indices: IntArray
    weights: FloatArray

    def __post_init__(self) -> None:
        indices = _integer_array(self.indices, name="indices", ndim=2)
        weights = _float_array(self.weights, name="weights", ndim=2)
        if indices.shape != weights.shape or indices.shape[1] == 0:
            raise ValueError("Neighbor indices and weights must share non-empty [N,K].")
        if np.any(indices < 0):
            raise ValueError("Neighbor indices must be non-negative.")
        _validate_simplex(weights, name="neighbor weights")
        object.__setattr__(self, "indices", _readonly_int(indices))
        object.__setattr__(self, "weights", _readonly_float(weights))


def interpolate_face_attributes(
    vertex_attributes: object,
    *,
    faces: object,
    face_indices: object,
    barycentric_coordinates: object,
) -> FloatArray:
    """Interpolate arbitrary numeric vertex attributes at triangle attachments."""
    attributes = _float_array(
        vertex_attributes,
        name="vertex_attributes",
        ndim=2,
    )
    face_array = _validate_faces(faces, vertex_count=attributes.shape[0])
    indices, barycentric = _validate_attachments(
        face_count=face_array.shape[0],
        face_indices=face_indices,
        barycentric_coordinates=barycentric_coordinates,
    )
    triangle_attributes = attributes[face_array[indices]]
    result = np.einsum("nk,nkd->nd", barycentric, triangle_attributes)
    return _readonly_float(result)


def embed_points_on_posed_mesh(
    posed_vertices_m: object,
    *,
    faces: object,
    face_indices: object,
    barycentric_coordinates: object,
) -> FloatArray:
    """Evaluate persistent barycentric triangle attachments over posed frames."""
    vertices = _float_array(posed_vertices_m, name="posed_vertices_m", ndim=3)
    if vertices.shape[2] != 3 or vertices.shape[0] == 0:
        raise ValueError("posed_vertices_m must have shape [T,V,3].")
    face_array = _validate_faces(faces, vertex_count=vertices.shape[1])
    indices, barycentric = _validate_attachments(
        face_count=face_array.shape[0],
        face_indices=face_indices,
        barycentric_coordinates=barycentric_coordinates,
    )
    triangles = vertices[:, face_array[indices], :]
    result = np.einsum("nk,tnkd->tnd", barycentric, triangles)
    return _readonly_float(result)


def apply_joint_linear_blend_skinning(
    canonical_points_m: object,
    *,
    point_joint_weights: object,
    joint_transforms: object,
    translations_m: object | None = None,
) -> FloatArray:
    """Apply explicit per-point SMPL joint weights without a hidden fallback."""
    points = _float_array(canonical_points_m, name="canonical_points_m", ndim=2)
    if points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("canonical_points_m must have shape [N,3].")
    weights = _float_array(
        point_joint_weights,
        name="point_joint_weights",
        ndim=2,
    )
    transforms = _validate_joint_transforms(joint_transforms)
    if weights.shape != (points.shape[0], transforms.shape[1]):
        raise ValueError("point_joint_weights must have shape [N,J].")
    _validate_simplex(weights, name="point_joint_weights")
    translations = _validate_translations(
        translations_m,
        frame_count=transforms.shape[0],
    )
    blended = np.einsum("nj,tjkl->tnkl", weights, transforms)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    result = np.einsum("tnij,nj->tni", blended[..., :3, :], homogeneous)
    result += translations[:, None, :]
    return _readonly_float(result)


def apply_vertex_transform_blend(
    canonical_points_m: object,
    *,
    vertex_transforms: object,
    neighbor_blend: NeighborBlend,
    translations_m: object | None = None,
) -> FloatArray:
    """Blend nearby SMPL vertex transforms as used by HUGS-style control."""
    points = _float_array(canonical_points_m, name="canonical_points_m", ndim=2)
    transforms = _float_array(
        vertex_transforms,
        name="vertex_transforms",
        ndim=4,
    )
    if points.shape != (neighbor_blend.indices.shape[0], 3):
        raise ValueError("canonical_points_m must match NeighborBlend rows.")
    if transforms.shape[-2:] != (4, 4) or transforms.shape[0] == 0:
        raise ValueError("vertex_transforms must have shape [T,V,4,4].")
    if np.any(neighbor_blend.indices >= transforms.shape[1]):
        raise ValueError("NeighborBlend references a missing vertex transform.")
    if not np.allclose(
        transforms[..., 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("vertex_transforms must be affine homogeneous matrices.")
    selected = transforms[:, neighbor_blend.indices, :, :]
    blended = np.einsum("nk,tnkij->tnij", neighbor_blend.weights, selected)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    result = np.einsum("tnij,nj->tni", blended[..., :3, :], homogeneous)
    translations = _validate_translations(
        translations_m,
        frame_count=transforms.shape[0],
    )
    result += translations[:, None, :]
    return _readonly_float(result)


def hugs_topk_neighbor_blend(
    canonical_points_m: object,
    *,
    template_vertices_m: object,
    vertex_joint_weights: object,
    k: int = 6,
    weight_std: float = 0.1,
    confidence_threshold: float = 0.9,
) -> NeighborBlend:
    """Reproduce HUGS' top-k distance/confidence rule without PyTorch3D."""
    points = _float_array(canonical_points_m, name="canonical_points_m", ndim=2)
    vertices = _float_array(
        template_vertices_m,
        name="template_vertices_m",
        ndim=2,
    )
    joint_weights = _float_array(
        vertex_joint_weights,
        name="vertex_joint_weights",
        ndim=2,
    )
    if points.shape[1] != 3 or vertices.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("Point and vertex arrays must have non-empty shape [N,3].")
    if joint_weights.shape[0] != vertices.shape[0]:
        raise ValueError("vertex_joint_weights must match template vertices.")
    _validate_simplex(joint_weights, name="vertex_joint_weights")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(vertices):
        raise ValueError("k must be an integer within the template vertex count.")
    if not np.isfinite(weight_std) or weight_std <= 0.0:
        raise ValueError("weight_std must be positive and finite.")
    if not np.isfinite(confidence_threshold) or not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in (0,1].")

    squared_distances = np.sum(
        (points[:, None, :] - vertices[None, :, :]) ** 2,
        axis=-1,
    )
    unsorted = np.argpartition(squared_distances, kth=k - 1, axis=1)[:, :k]
    row = np.arange(points.shape[0])[:, None]
    order = np.argsort(squared_distances[row, unsorted], axis=1, kind="stable")
    indices = unsorted[row, order]
    neighbor_distances = squared_distances[row, indices]
    neighbor_joint_weights = joint_weights[indices]
    differences = np.abs(neighbor_joint_weights - neighbor_joint_weights[:, :1, :]).sum(
        axis=-1
    )
    confidence = np.exp(-differences / (2.0 * weight_std**2))
    accepted = confidence > confidence_threshold
    accepted[:, 0] = True
    weights = np.exp(-neighbor_distances) * accepted
    weights /= weights.sum(axis=1, keepdims=True)
    return NeighborBlend(indices=indices, weights=weights)

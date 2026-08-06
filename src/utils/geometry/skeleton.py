"""Skeleton / 3D pose geometry helpers (torch).

Joint-angle, torsion, torso-twist and bone-length computations over COCO-17
poses. Extracted from ``plcs`` losses so analysis scripts and other tasks can
reuse them without importing the loss module.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.geometry.angles import normalize_vector, signed_angle_around_axis
from src.utils.schema.player import (
    COCO17_BONE_LENGTH_EDGES,
    COCO17_JOINT_ANGLE_TRIPLETS,
    COCO17_TORSION_QUADRUPLETS,
    COCO17_TORSO_TWIST_JOINTS,
)


def compute_joint_angles(
    pose: Tensor,
    triplets: tuple[tuple[int, int, int], ...] = COCO17_JOINT_ANGLE_TRIPLETS,
) -> Tensor:
    """Compute interior joint angles in radians.

    For each triplet ``(a, vertex, c)``, computes the angle at ``vertex``
    between bones ``vertex -> a`` and ``vertex -> c`` using
    ``atan2(||v1 x v2||, v1 . v2)``, which is stable near 0 and pi.

    Args:
        pose: Joint positions, shape ``(..., J, 3)``.
        triplets: Joint index triplets.

    Returns:
        Tensor: Angles in radians, shape ``(..., len(triplets))``.
    """
    a_idx = [t[0] for t in triplets]
    b_idx = [t[1] for t in triplets]
    c_idx = [t[2] for t in triplets]

    vertex = pose[..., b_idx, :]
    v1 = pose[..., a_idx, :] - vertex
    v2 = pose[..., c_idx, :] - vertex

    cross_norm = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    dot = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross_norm, dot)


def compute_torsion_angles(
    pose: Tensor,
    quadruplets: tuple[tuple[int, int, int, int], ...] = COCO17_TORSION_QUADRUPLETS,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Compute signed torsion / dihedral angles in radians.

    For each quadruplet ``(a, b, c, d)``, computes the signed angle between
    plane ``(a, b, c)`` and plane ``(b, c, d)``, capturing the 3D bending
    direction of limbs.

    Args:
        pose: Joint positions, shape ``(..., J, 3)``.
        quadruplets: Joint index quadruplets.
        eps: Numerical stability epsilon.

    Returns:
        Tensor: Signed torsion angles in radians, shape
        ``(..., len(quadruplets))``.
    """
    a_idx = [q[0] for q in quadruplets]
    b_idx = [q[1] for q in quadruplets]
    c_idx = [q[2] for q in quadruplets]
    d_idx = [q[3] for q in quadruplets]

    p0 = pose[..., a_idx, :]
    p1 = pose[..., b_idx, :]
    p2 = pose[..., c_idx, :]
    p3 = pose[..., d_idx, :]

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    n1 = normalize_vector(torch.cross(b0, b1, dim=-1), eps=eps)
    n2 = normalize_vector(torch.cross(b1, b2, dim=-1), eps=eps)
    b1n = normalize_vector(b1, eps=eps)

    # Signed dihedral angle.
    m1 = torch.cross(n1, b1n, dim=-1)
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def compute_torso_twist(
    pose: Tensor,
    joints: tuple[int, int, int, int] = COCO17_TORSO_TWIST_JOINTS,
) -> Tensor:
    """Compute the shoulder-hip twist angle from a COCO-17 pose.

    The twist is the signed angle from the hip axis to the shoulder axis around
    the torso axis.

    Args:
        pose: Joint positions, shape ``(..., 17, 3)``.
        joints: ``(left_shoulder, right_shoulder, left_hip, right_hip)``.

    Returns:
        Tensor: Signed torso twist angle in radians, shape ``(...)``.
    """
    left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx = joints

    left_shoulder = pose[..., left_shoulder_idx, :]
    right_shoulder = pose[..., right_shoulder_idx, :]
    left_hip = pose[..., left_hip_idx, :]
    right_hip = pose[..., right_hip_idx, :]

    mid_shoulder = 0.5 * (left_shoulder + right_shoulder)
    mid_hip = 0.5 * (left_hip + right_hip)

    shoulder_axis = right_shoulder - left_shoulder
    hip_axis = right_hip - left_hip
    torso_axis = mid_shoulder - mid_hip

    return signed_angle_around_axis(hip_axis, shoulder_axis, torso_axis)


def compute_bone_lengths(
    pose: Tensor,
    edges: tuple[tuple[int, int], ...] = COCO17_BONE_LENGTH_EDGES,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Compute bone lengths for selected COCO body edges.

    Args:
        pose: Joint positions, shape ``(..., J, 3)``.
        edges: Bone edges as ``(joint_a, joint_b)``.
        eps: Numerical stability epsilon.

    Returns:
        Tensor: Bone lengths, shape ``(..., len(edges))``.
    """
    a_idx = [e[0] for e in edges]
    b_idx = [e[1] for e in edges]

    bone_vec = pose[..., a_idx, :] - pose[..., b_idx, :]
    return torch.clamp_min(torch.linalg.vector_norm(bone_vec, dim=-1), eps)


__all__ = [
    "compute_bone_lengths",
    "compute_joint_angles",
    "compute_torsion_angles",
    "compute_torso_twist",
]

"""Torch rotation-representation conversions (axis-angle / quaternion / matrix / 6D).

Adapted from PyTorch3D's ``pytorch3d.transforms.rotation_conversions``
(BSD-3-Clause, Copyright (c) Meta Platforms, Inc. and affiliates) so that code
which only needs rotation conversions does not require the compiled pytorch3d
package. Conventions follow PyTorch3D:

- Quaternions are ``(w, x, y, z)`` with real part first.
- Rotation matrices are right-multiplied column-vector convention,
  i.e. ``points_rotated = R @ points``.
- The 6D representation follows Zhou et al. "On the Continuity of Rotation
  Representations in Neural Networks" (CVPR 2019): the first two rows of the
  rotation matrix, flattened.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert quaternions ``(..., 4)`` (w, x, y, z) to rotation matrices ``(..., 3, 3)``."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: Tensor) -> Tensor:
    """Return ``sqrt(max(0, x))`` with a subgradient of zero at zero."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def standardize_quaternion(quaternions: Tensor) -> Tensor:
    """Flip quaternions so the real part is non-negative ``(..., 4)``."""
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotation matrices ``(..., 3, 3)`` to quaternions ``(..., 4)`` (w, x, y, z)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # clamp the denominator to avoid dividing by tiny values for degenerate rows
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # pick the candidate from the row with the largest denominator
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors ``(..., 3)`` to quaternions ``(..., 4)`` (w, x, y, z)."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # For small angles: sin(x/2) / x ≈ 1/2 - x^2 / 48 (Taylor expansion)
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )


def quaternion_to_axis_angle(quaternions: Tensor) -> Tensor:
    """Convert quaternions ``(..., 4)`` (w, x, y, z) to axis-angle vectors ``(..., 3)``."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors ``(..., 3)`` to rotation matrices ``(..., 3, 3)``."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Convert rotation matrices ``(..., 3, 3)`` to axis-angle vectors ``(..., 3)``."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Convert 6D rotation representation ``(..., 6)`` to matrices ``(..., 3, 3)``.

    Uses Gram-Schmidt orthogonalization per Zhou et al. (CVPR 2019).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Convert rotation matrices ``(..., 3, 3)`` to the 6D representation ``(..., 6)``."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:
    """Rotation matrices ``(..., 3, 3)`` about one canonical axis (``"X"|"Y"|"Z"``)."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"letter must be either X, Y or Z, got {axis}.")

    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: Tensor, convention: str) -> Tensor:
    """Convert Euler angles ``(..., 3)`` to rotation matrices ``(..., 3, 3)``.

    Args:
        euler_angles: Euler angles in radians.
        convention: Three uppercase letters from {"X", "Y", "Z"}, e.g. ``"XYZ"``.
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1), strict=True)
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

"""Differentiable COCO-17 geometry with explicit feature validity.

Positions and length thresholds are in metres. Callers remove padding first;
nonfinite poses are rejected. Invalid features have a zero storage value and
zero gradient, and must only be used with their mask. See ``utils/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.schema.player import (
    COCO17_BONE_LENGTH_EDGES,
    COCO17_JOINT_ANGLE_TRIPLETS,
    COCO17_TORSION_QUADRUPLETS,
    COCO17_TORSO_TWIST_JOINTS,
)

# Adopted values (#1539, 2026-11-01 user decision); not tuned on measured
# frequencies. Re-qualify with analyze_skeleton_validity before changing them.
# Physical length [m] and dimensionless sin(angle) are separate domain tests.
MIN_BONE_LENGTH_M: float = 1e-6
MIN_PLANE_SINE: float = 1e-6


@dataclass(frozen=True)
class SkeletonValues:
    """Values and same-shaped bool validity (twist has no feature axis).

    Invalid zeroes are storage only, not observed angles/lengths. A prediction
    must pass :meth:`require_valid` even when its target feature is missing.
    """

    values: Tensor
    valid: Tensor

    def require_valid(self, context: str) -> None:
        """Reject degenerate predictions with sample/feature indices."""
        if not bool(self.valid.all()):
            indices = (~self.valid).nonzero()[:8].tolist()
            raise ValueError(
                f"{context}: degenerate/invalid skeleton features at indices "
                f"{indices} (last index is feature, except for torso twist)."
            )


def _validate_pose(pose: Tensor) -> None:
    if pose.ndim < 2 or pose.shape[-1] != 3 or not pose.is_floating_point():
        raise ValueError(
            "pose must be floating point with shape (..., J, 3) in metres."
        )
    if not bool(torch.isfinite(pose).all()):
        raise ValueError(
            "pose must contain only finite coordinates (remove padding first)."
        )


def _length_valid(vector: Tensor) -> Tensor:
    # Domain decisions do not create a backward path through zero norms.
    with torch.no_grad():
        return torch.linalg.vector_norm(vector, dim=-1) > MIN_BONE_LENGTH_M


def _plane_sine(first: Tensor, second: Tensor) -> Tensor:
    """Dimensionless cross magnitude; zero-length inputs give diagnostic 0."""
    with torch.no_grad():

        def unit_or_zero(v: Tensor) -> Tensor:
            length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
            return v / torch.where(length > 0, length, torch.ones_like(length))

        return torch.linalg.vector_norm(
            torch.cross(unit_or_zero(first), unit_or_zero(second), dim=-1), dim=-1
        )


def _replace_invalid(
    vector: Tensor, valid: Tensor, fallback: tuple[int, int, int]
) -> Tensor:
    # Must precede norm/normalization/atan2, not merely mask their outputs.
    return torch.where(valid.unsqueeze(-1), vector, vector.new_tensor(fallback))


def _unit(vector: Tensor) -> Tensor:
    """Normalize vectors already selected/replaced into the valid domain."""
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True)


def _result(values: Tensor, valid: Tensor) -> SkeletonValues:
    return SkeletonValues(torch.where(valid, values, torch.zeros_like(values)), valid)


def compute_joint_angles(
    pose: Tensor,
    triplets: tuple[tuple[int, int, int], ...] = COCO17_JOINT_ANGLE_TRIPLETS,
) -> SkeletonValues:
    """Interior angles [rad], shape ``(..., A)``; both bones must be valid.

    Nonzero collinear bones have valid interior angles 0/pi. No claim is made
    about a unique derivative at these nonsmooth endpoints.
    """
    _validate_pose(pose)
    vertex = pose[..., [t[1] for t in triplets], :]
    v1 = pose[..., [t[0] for t in triplets], :] - vertex
    v2 = pose[..., [t[2] for t in triplets], :] - vertex
    valid = _length_valid(v1) & _length_valid(v2)
    v1 = _replace_invalid(v1, valid, (1, 0, 0))
    v2 = _replace_invalid(v2, valid, (0, 1, 0))
    cross_norm = torch.linalg.vector_norm(torch.cross(v1, v2, dim=-1), dim=-1)
    dot = (v1 * v2).sum(dim=-1)
    return _result(torch.atan2(cross_norm, dot), valid)


def _torsion_bones(
    pose: Tensor, quadruplets: tuple[tuple[int, int, int, int], ...]
) -> tuple[Tensor, Tensor, Tensor]:
    p0, p1, p2, p3 = (pose[..., [q[i] for q in quadruplets], :] for i in range(4))
    return p1 - p0, p2 - p1, p3 - p2


def compute_torsion_angles(
    pose: Tensor,
    quadruplets: tuple[tuple[int, int, int, int], ...] = COCO17_TORSION_QUADRUPLETS,
) -> SkeletonValues:
    """Signed dihedrals [rad], shape ``(..., A)``.

    All three bones and both normalized cross products must be valid. The
    signed branch near +/-pi remains in the domain when the planes exist.
    """
    _validate_pose(pose)
    b0, b1, b2 = _torsion_bones(pose, quadruplets)
    valid = (
        _length_valid(b0)
        & _length_valid(b1)
        & _length_valid(b2)
        & (_plane_sine(b0, b1) > MIN_PLANE_SINE)
        & (_plane_sine(b1, b2) > MIN_PLANE_SINE)
    )
    b0 = _unit(_replace_invalid(b0, valid, (1, 0, 0)))
    b1 = _unit(_replace_invalid(b1, valid, (0, 1, 0)))
    b2 = _unit(_replace_invalid(b2, valid, (0, 0, 1)))
    n1 = _unit(torch.cross(b0, b1, dim=-1))
    n2 = _unit(torch.cross(b1, b2, dim=-1))
    x = (n1 * n2).sum(dim=-1)
    y = (torch.cross(n1, b1, dim=-1) * n2).sum(dim=-1)
    return _result(torch.atan2(y, x), valid)


def _torso_axes(
    pose: Tensor, joints: tuple[int, int, int, int]
) -> tuple[Tensor, Tensor, Tensor]:
    ls, rs, lh, rh = (pose[..., i, :] for i in joints)
    return rh - lh, rs - ls, 0.5 * (ls + rs) - 0.5 * (lh + rh)


def compute_torso_twist(
    pose: Tensor,
    joints: tuple[int, int, int, int] = COCO17_TORSO_TWIST_JOINTS,
) -> SkeletonValues:
    """Hip-to-shoulder signed twist [rad], shape ``(...)``.

    Hip, shoulder and torso axes must have valid length; hip/shoulder axes
    must each define a plane with the torso axis (valid transverse projection).
    """
    _validate_pose(pose)
    hip, shoulder, torso = _torso_axes(pose, joints)
    valid = (
        _length_valid(hip)
        & _length_valid(shoulder)
        & _length_valid(torso)
        & (_plane_sine(hip, torso) > MIN_PLANE_SINE)
        & (_plane_sine(shoulder, torso) > MIN_PLANE_SINE)
    )
    hip = _unit(_replace_invalid(hip, valid, (1, 0, 0)))
    shoulder = _unit(_replace_invalid(shoulder, valid, (0, 1, 0)))
    axis = _unit(_replace_invalid(torso, valid, (0, 0, 1)))
    hip_proj = _unit(hip - (hip * axis).sum(dim=-1, keepdim=True) * axis)
    shoulder_proj = _unit(shoulder - (shoulder * axis).sum(dim=-1, keepdim=True) * axis)
    x = (hip_proj * shoulder_proj).sum(dim=-1)
    y = (torch.cross(hip_proj, shoulder_proj, dim=-1) * axis).sum(dim=-1)
    return _result(torch.atan2(y, x), valid)


def compute_bone_lengths(
    pose: Tensor,
    edges: tuple[tuple[int, int], ...] = COCO17_BONE_LENGTH_EDGES,
) -> SkeletonValues:
    """Bone lengths [m], shape ``(..., A)``; zero/short bones are missing."""
    _validate_pose(pose)
    bone = pose[..., [e[0] for e in edges], :] - pose[..., [e[1] for e in edges], :]
    valid = _length_valid(bone)
    bone = _replace_invalid(bone, valid, (1, 0, 0))
    return _result(torch.linalg.vector_norm(bone, dim=-1), valid)


def skeleton_domain_metrics(pose: Tensor) -> dict[str, Tensor]:
    """Raw per-feature metrics for threshold surveys, using the API geometry.

    Zero-length bones produce diagnostic plane sine 0, not a measured plane.
    Axes are ordered hip/shoulder/torso; planes left/right limb pairs and
    hip/shoulder transverse projections. Padding must be removed by the caller.
    """
    with torch.no_grad():
        _validate_pose(pose)
        b0, b1, b2 = _torsion_bones(pose, COCO17_TORSION_QUADRUPLETS)
        hip, shoulder, torso = _torso_axes(pose, COCO17_TORSO_TWIST_JOINTS)
        edges = COCO17_BONE_LENGTH_EDGES
        bones = (
            pose[..., [e[0] for e in edges], :] - pose[..., [e[1] for e in edges], :]
        )
        return {
            "bone_length_m": torch.linalg.vector_norm(bones, dim=-1),
            "torsion_plane_sine": torch.stack(
                [_plane_sine(b0, b1), _plane_sine(b1, b2)], dim=-1
            ),
            "torso_axis_length_m": torch.stack(
                [torch.linalg.vector_norm(v, dim=-1) for v in (hip, shoulder, torso)],
                dim=-1,
            ),
            "torso_plane_sine": torch.stack(
                [_plane_sine(hip, torso), _plane_sine(shoulder, torso)], dim=-1
            ),
        }


__all__ = [
    "MIN_BONE_LENGTH_M",
    "MIN_PLANE_SINE",
    "SkeletonValues",
    "compute_bone_lengths",
    "compute_joint_angles",
    "compute_torsion_angles",
    "compute_torso_twist",
    "skeleton_domain_metrics",
]

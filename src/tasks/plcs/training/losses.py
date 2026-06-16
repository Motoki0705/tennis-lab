"""Loss functions for PLCS training.

Supports frame-level and sequence-level losses.
Temporal consistency is enforced by the GAN discriminator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.utils.pose_geometry import world_pose_to_canonical_pose
from src.utils.schema.player import (
    COCO17_BONE_LENGTH_EDGES as BONE_LENGTH_EDGES,
)
from src.utils.schema.player import (
    COCO17_JOINT_ANGLE_TRIPLETS as JOINT_ANGLE_TRIPLETS,
)
from src.utils.schema.player import (
    COCO17_TORSION_QUADRUPLETS as TORSION_QUADRUPLETS,
)
from src.utils.schema.player import (
    COCO17_TORSO_TWIST_JOINTS as TORSO_TWIST_JOINTS,
)
from src.utils.tensor_utils import masked_mean, normalize_padding_mask

# Loss terms that require pred_canonical_pose and target_human_kp_3d.
CANONICAL_DEPENDENT_TERM_NAMES: tuple[str, ...] = (
    "canonical_pose",
    "joint_angle",
    "torsion_angle",
    "torso_twist",
    "bone_length",
)


@dataclass(frozen=True)
class PLCSLossConfig:
    """Configuration for PLCS loss weights.

    Attributes:
        position_weight: Weight for position loss.
        rotation_weight: Weight for rotation loss.
        canonical_pose_weight: Weight for canonical pose loss.
        joint_angle_weight: Weight for joint-angle loss.
        torsion_angle_weight: Weight for limb torsion / dihedral angle loss.
        torso_twist_weight: Weight for shoulder-hip twist loss.
        bone_length_weight: Weight for bone-length consistency loss.

    """

    position_weight: float = 1.0
    rotation_weight: float = 1.0
    canonical_pose_weight: float = 0.0
    joint_angle_weight: float = 0.0
    torsion_angle_weight: float = 0.0
    torso_twist_weight: float = 0.0
    bone_length_weight: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict) -> PLCSLossConfig:
        """Create config from dictionary, e.g. loaded from YAML."""
        return cls(
            position_weight=float(cfg.get("position_weight", 1.0)),
            rotation_weight=float(cfg.get("rotation_weight", 1.0)),
            canonical_pose_weight=float(cfg.get("canonical_pose_weight", 0.0)),
            joint_angle_weight=float(cfg.get("joint_angle_weight", 0.0)),
            torsion_angle_weight=float(cfg.get("torsion_angle_weight", 0.0)),
            torso_twist_weight=float(cfg.get("torso_twist_weight", 0.0)),
            bone_length_weight=float(cfg.get("bone_length_weight", 0.0)),
        )


# ---------------------------------------------------------------------------
# Basic losses / metrics
# ---------------------------------------------------------------------------


def position_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute smooth-L1 position loss.

    Args:
        pred: Predicted position, shape ``(..., 3)``.
        target: Target position, shape ``(..., 3)``.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.

    Returns:
        Tensor: Position loss.

    """
    return nn.functional.smooth_l1_loss(pred, target, reduction=reduction)


def rotation_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute rotation loss for ``(cos, sin)`` yaw representation.

    Uses ``1 - cosine_similarity``. Both prediction and target are normalized
    for safety.

    Args:
        pred: Predicted ``(cos, sin)``, shape ``(..., 2)``.
        target: Target ``(cos, sin)``, shape ``(..., 2)``.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.

    Returns:
        Tensor: Rotation loss.

    """
    pred_norm = nn.functional.normalize(pred, dim=-1)
    target_norm = nn.functional.normalize(target, dim=-1)
    loss = 1.0 - (pred_norm * target_norm).sum(dim=-1)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def angular_error(pred: Tensor, target: Tensor) -> Tensor:
    """Compute wrapped angular error in radians.

    Args:
        pred: Predicted ``(cos, sin)``, shape ``(..., 2)``.
        target: Target ``(cos, sin)``, shape ``(..., 2)``.

    Returns:
        Tensor: Absolute angular error in radians, shape ``(...)``.

    """
    pred_angle = torch.atan2(pred[..., 1], pred[..., 0])
    target_angle = torch.atan2(target[..., 1], target[..., 0])
    diff = pred_angle - target_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff.abs()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _normalize_vector(v: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Normalize vectors along the last dimension."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _wrapped_angle_diff(pred_angle: Tensor, target_angle: Tensor) -> Tensor:
    """Return signed wrapped angle difference in ``[-pi, pi]``."""
    diff = pred_angle - target_angle
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def compute_joint_angles(
    pose: Tensor,
    triplets: tuple[tuple[int, int, int], ...] = JOINT_ANGLE_TRIPLETS,
) -> Tensor:
    """Compute interior joint angles in radians.

    For each triplet ``(a, vertex, c)``, computes the angle at ``vertex``
    between bones ``vertex -> a`` and ``vertex -> c``.

    The angle is computed with ``atan2(||v1 x v2||, v1 . v2)``, which is
    stable near 0 and pi.

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
    quadruplets: tuple[tuple[int, int, int, int], ...] = TORSION_QUADRUPLETS,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Compute signed torsion / dihedral angles in radians.

    For each quadruplet ``(a, b, c, d)``, computes the signed angle between
    the two planes:

    - plane 1: ``(a, b, c)``
    - plane 2: ``(b, c, d)``

    This captures 3D bending direction of limbs.

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

    n1 = _normalize_vector(torch.cross(b0, b1, dim=-1), eps=eps)
    n2 = _normalize_vector(torch.cross(b1, b2, dim=-1), eps=eps)
    b1n = _normalize_vector(b1, eps=eps)

    # Signed dihedral angle.
    m1 = torch.cross(n1, b1n, dim=-1)
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def signed_angle_around_axis(
    v1: Tensor,
    v2: Tensor,
    axis: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Compute signed angle from ``v1`` to ``v2`` around ``axis``.

    The vectors are first projected onto the plane perpendicular to ``axis``.
    This is useful for measuring body twist around the torso axis.

    Args:
        v1: First vector, shape ``(..., 3)``.
        v2: Second vector, shape ``(..., 3)``.
        axis: Rotation axis, shape ``(..., 3)``.
        eps: Numerical stability epsilon.

    Returns:
        Tensor: Signed angle in radians, shape ``(...)``.

    """
    axis = _normalize_vector(axis, eps=eps)

    v1_proj = v1 - (v1 * axis).sum(dim=-1, keepdim=True) * axis
    v2_proj = v2 - (v2 * axis).sum(dim=-1, keepdim=True) * axis

    v1_proj = _normalize_vector(v1_proj, eps=eps)
    v2_proj = _normalize_vector(v2_proj, eps=eps)

    x = (v1_proj * v2_proj).sum(dim=-1)
    y = (torch.cross(v1_proj, v2_proj, dim=-1) * axis).sum(dim=-1)
    return torch.atan2(y, x)


def compute_torso_twist(
    pose: Tensor,
    joints: tuple[int, int, int, int] = TORSO_TWIST_JOINTS,
) -> Tensor:
    """Compute shoulder-hip twist angle from COCO-17 pose.

    The twist is the signed angle from the hip axis to the shoulder axis
    around the torso axis.

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
    edges: tuple[tuple[int, int], ...] = BONE_LENGTH_EDGES,
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
    return bone_vec.norm(dim=-1).clamp_min(eps)


# ---------------------------------------------------------------------------
# Pose naturalness losses
# ---------------------------------------------------------------------------


def joint_angle_loss(
    pred_pose: Tensor,
    target_pose: Tensor,
    triplets: tuple[tuple[int, int, int], ...] = JOINT_ANGLE_TRIPLETS,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute joint-angle loss between predicted and target poses.

    This compares 12 angles by default:

    - 8 articulated limb angles
    - 4 torso quadrilateral internal angles

    Args:
        pred_pose: Predicted joints, shape ``(..., J, 3)``.
        target_pose: Target joints, shape ``(..., J, 3)``.
        triplets: Joint-angle triplets.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.

    Returns:
        Tensor: Joint-angle loss. If ``reduction='none'``, returns per-frame
        loss with shape ``(...)``.

    """
    pred_angles = compute_joint_angles(pred_pose, triplets)
    target_angles = compute_joint_angles(target_pose, triplets)
    per_angle = nn.functional.smooth_l1_loss(
        pred_angles,
        target_angles,
        reduction="none",
    )

    if reduction == "mean":
        return per_angle.mean()
    if reduction == "sum":
        return per_angle.sum()
    return per_angle.mean(dim=-1)


def torsion_angle_loss(
    pred_pose: Tensor,
    target_pose: Tensor,
    quadruplets: tuple[tuple[int, int, int, int], ...] = TORSION_QUADRUPLETS,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute signed torsion-angle loss between predicted and target poses.

    This captures the 3D bending plane of arms and legs. Angle differences are
    wrapped into ``[-pi, pi]`` before applying smooth-L1.

    Args:
        pred_pose: Predicted joints, shape ``(..., J, 3)``.
        target_pose: Target joints, shape ``(..., J, 3)``.
        quadruplets: Torsion quadruplets.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.

    Returns:
        Tensor: Torsion-angle loss. If ``reduction='none'``, returns per-frame
        loss with shape ``(...)``.

    """
    pred_angles = compute_torsion_angles(pred_pose, quadruplets)
    target_angles = compute_torsion_angles(target_pose, quadruplets)
    diff = _wrapped_angle_diff(pred_angles, target_angles)

    per_angle = nn.functional.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        reduction="none",
    )

    if reduction == "mean":
        return per_angle.mean()
    if reduction == "sum":
        return per_angle.sum()
    return per_angle.mean(dim=-1)


def torso_twist_loss(
    pred_pose: Tensor,
    target_pose: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute shoulder-hip torso twist loss.

    Measures the signed angle from the hip axis to the shoulder axis around
    the torso axis. This captures upper/lower-body twist in 3D.

    Args:
        pred_pose: Predicted joints, shape ``(..., 17, 3)``.
        target_pose: Target joints, shape ``(..., 17, 3)``.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.

    Returns:
        Tensor: Torso twist loss. If ``reduction='none'``, returns per-frame
        loss with shape ``(...)``.

    """
    pred_twist = compute_torso_twist(pred_pose)
    target_twist = compute_torso_twist(target_pose)
    diff = _wrapped_angle_diff(pred_twist, target_twist)

    per_frame = nn.functional.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        reduction="none",
    )

    if reduction == "mean":
        return per_frame.mean()
    if reduction == "sum":
        return per_frame.sum()
    return per_frame


def bone_length_loss(
    pred_pose: Tensor,
    target_pose: Tensor,
    edges: tuple[tuple[int, int], ...] = BONE_LENGTH_EDGES,
    *,
    relative: bool = True,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Compute bone-length consistency loss.

    By default this uses relative bone-length error:

    ``(pred_length - target_length) / target_length``

    so the loss is dimensionless and does not over-emphasize long bones.

    Args:
        pred_pose: Predicted joints, shape ``(..., J, 3)``.
        target_pose: Target joints, shape ``(..., J, 3)``.
        edges: Bone edges.
        relative: Whether to use relative length error.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
        eps: Numerical stability epsilon.

    Returns:
        Tensor: Bone-length loss. If ``reduction='none'``, returns per-frame
        loss with shape ``(...)``.

    """
    pred_lengths = compute_bone_lengths(pred_pose, edges, eps=eps)
    target_lengths = compute_bone_lengths(target_pose, edges, eps=eps)

    if relative:
        diff = (pred_lengths - target_lengths) / target_lengths.clamp_min(eps)
        per_bone = nn.functional.smooth_l1_loss(
            diff,
            torch.zeros_like(diff),
            reduction="none",
        )
    else:
        per_bone = nn.functional.smooth_l1_loss(
            pred_lengths,
            target_lengths,
            reduction="none",
        )

    if reduction == "mean":
        return per_bone.mean()
    if reduction == "sum":
        return per_bone.sum()
    return per_bone.mean(dim=-1)


# ---------------------------------------------------------------------------
# PLCS loss registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLCSLossInputs:
    """Bundle of tensors shared across PLCS loss terms.

    Attributes:
        pred_position: Predicted position, ``(B, 3)`` or ``(B, T, 3)``.
        pred_rotation: Predicted ``(cos, sin)``, ``(B, 2)`` or ``(B, T, 2)``.
        target_position: Target position, same shape as ``pred_position``.
        target_rotation: Target ``(cos, sin)``, same shape as ``pred_rotation``.
        frame_mask: Optional per-frame validity mask for padded sequences.
        pred_canonical_pose: Predicted canonical joints, ``(..., J, 3)``.
        target_canonical_pose: Target canonical joints, ``(..., J, 3)``.

    """

    pred_position: Tensor
    pred_rotation: Tensor
    target_position: Tensor
    target_rotation: Tensor
    frame_mask: Tensor | None = None
    pred_canonical_pose: Tensor | None = None
    target_canonical_pose: Tensor | None = None

    @property
    def zero(self) -> Tensor:
        """Scalar zero on the same device/dtype as the predictions."""
        return self.pred_position.new_zeros(())

    @property
    def has_canonical(self) -> bool:
        """Whether both predicted and target canonical poses are available."""
        return (
            self.pred_canonical_pose is not None
            and self.target_canonical_pose is not None
        )


def _masked_frame_mean(per_frame: Tensor, frame_mask: Tensor | None) -> Tensor:
    """Reduce per-frame losses, honoring the padding mask when shapes match."""
    if frame_mask is not None and per_frame.shape == frame_mask.shape:
        return masked_mean(per_frame, frame_mask, binarize=True, denom_min=1.0)
    return per_frame.mean()


def position_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Position term: masked smooth-L1 between predicted and target positions."""
    per_frame = nn.functional.smooth_l1_loss(
        inputs.pred_position,
        inputs.target_position,
        reduction="none",
    ).mean(dim=-1)
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def rotation_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Rotation term: masked ``1 - cosine similarity`` on ``(cos, sin)``."""
    pred_norm = nn.functional.normalize(inputs.pred_rotation, dim=-1)
    target_norm = nn.functional.normalize(inputs.target_rotation, dim=-1)
    per_frame = 1.0 - (pred_norm * target_norm).sum(dim=-1)
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def canonical_pose_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Canonical-pose term: masked smooth-L1 between canonical joint positions."""
    if not inputs.has_canonical:
        return inputs.zero

    per_frame = nn.functional.smooth_l1_loss(
        inputs.pred_canonical_pose,
        inputs.target_canonical_pose,
        reduction="none",
    ).mean(dim=(-1, -2))
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def joint_angle_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Joint-angle term on canonical joints.

    Includes both articulated limb angles and torso quadrilateral angles.
    """
    if not inputs.has_canonical:
        return inputs.zero

    per_frame = joint_angle_loss(
        inputs.pred_canonical_pose,
        inputs.target_canonical_pose,
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def torsion_angle_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Limb torsion term on canonical joints.

    Captures whether arms/legs bend in the correct 3D plane.
    """
    if not inputs.has_canonical:
        return inputs.zero

    per_frame = torsion_angle_loss(
        inputs.pred_canonical_pose,
        inputs.target_canonical_pose,
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def torso_twist_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Torso twist term on canonical joints.

    Captures shoulder-line vs hip-line twist around the torso axis.
    """
    if not inputs.has_canonical:
        return inputs.zero

    per_frame = torso_twist_loss(
        inputs.pred_canonical_pose,
        inputs.target_canonical_pose,
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def bone_length_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Bone-length consistency term on canonical joints."""
    if not inputs.has_canonical:
        return inputs.zero

    per_frame = bone_length_loss(
        inputs.pred_canonical_pose,
        inputs.target_canonical_pose,
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


# Type of a single loss term: maps shared inputs to a scalar loss tensor.
PLCSLossTerm = Callable[[PLCSLossInputs], Tensor]


# Default registry of loss terms keyed by output name.
# forward() iterates this mapping, so adding a term is a matter of:
#   1. defining a function
#   2. registering it here
#   3. adding ``<name>_weight`` to PLCSLossConfig
DEFAULT_LOSS_TERMS: dict[str, PLCSLossTerm] = {
    "position": position_loss_term,
    "rotation": rotation_loss_term,
    "canonical_pose": canonical_pose_loss_term,
    "joint_angle": joint_angle_loss_term,
    "torsion_angle": torsion_angle_loss_term,
    "torso_twist": torso_twist_loss_term,
    "bone_length": bone_length_loss_term,
}


class PLCSLoss(nn.Module):
    """Combined loss for PLCS training.

    Each loss term is an independent function with the uniform signature
    ``(PLCSLossInputs) -> Tensor``. The module holds a ``{name: function}``
    registry and ``forward`` builds the shared :class:`PLCSLossInputs`, calls
    each term, and accumulates the weight-scaled sum.

    Supports both frame-level and sequence-level inputs:
        - Frame-level: ``(B, 3)``, ``(B, 2)``
        - Sequence-level: ``(B, T, 3)``, ``(B, T, 2)``

    Temporal consistency is enforced by the GAN discriminator.
    """

    def __init__(
        self,
        config: PLCSLossConfig | None = None,
        *,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        loss_terms: dict[str, PLCSLossTerm] | None = None,
    ) -> None:
        """Initialize the loss module.

        Args:
            config: Loss configuration. If provided, overrides legacy weights.
            position_weight: Legacy position-loss weight.
            rotation_weight: Legacy rotation-loss weight.
            loss_terms: Optional custom loss registry.

        """
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PLCSLossConfig(
                position_weight=position_weight,
                rotation_weight=rotation_weight,
            )

        self.loss_terms: dict[str, PLCSLossTerm] = (
            dict(DEFAULT_LOSS_TERMS) if loss_terms is None else dict(loss_terms)
        )

    def weight_for(self, name: str) -> float:
        """Return the configured weight for a loss term, or 0.0 if unset."""
        return float(getattr(self.config, f"{name}_weight", 0.0))

    def _requires_canonical_pose(self) -> bool:
        """Whether any enabled registered term needs canonical pose tensors."""
        return any(
            name in self.loss_terms and self.weight_for(name) > 0.0
            for name in CANONICAL_DEPENDENT_TERM_NAMES
        )

    def _build_inputs(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
        *,
        pred_canonical_pose: Tensor | None,
        target_human_kp_3d: Tensor | None,
        human_mask: Tensor | None,
    ) -> PLCSLossInputs:
        """Build shared loss inputs and target canonical pose."""
        frame_mask = normalize_padding_mask(human_mask, flatten=False)

        canonical_required = self._requires_canonical_pose()
        if canonical_required and pred_canonical_pose is None:
            raise ValueError(
                "pred_canonical_pose is required when any canonical-dependent "
                "loss weight is > 0. Enabled canonical-dependent terms include: "
                f"{CANONICAL_DEPENDENT_TERM_NAMES}."
            )

        target_canonical_pose: Tensor | None = None
        if pred_canonical_pose is not None:
            if target_human_kp_3d is None:
                if canonical_required:
                    raise ValueError(
                        "target_human_kp_3d is required when any "
                        "canonical-dependent loss weight is > 0 and "
                        "pred_canonical_pose is provided."
                    )
            else:
                target_canonical_pose = world_pose_to_canonical_pose(
                    target_human_kp_3d,
                    target_position,
                    target_rotation,
                )

        return PLCSLossInputs(
            pred_position=pred_position,
            pred_rotation=pred_rotation,
            target_position=target_position,
            target_rotation=target_rotation,
            frame_mask=frame_mask,
            pred_canonical_pose=pred_canonical_pose,
            target_canonical_pose=target_canonical_pose,
        )

    def forward(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
        *,
        pred_canonical_pose: Tensor | None = None,
        target_human_kp_3d: Tensor | None = None,
        human_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute combined loss by dispatching to each registered term.

        Args:
            pred_position: Predicted position, ``(B, 3)`` or ``(B, T, 3)``.
            pred_rotation: Predicted ``(cos, sin)``, ``(B, 2)`` or ``(B, T, 2)``.
            target_position: Target position.
            target_rotation: Target ``(cos, sin)``.
            pred_canonical_pose: Predicted canonical joints, ``(..., J, 3)``.
            target_human_kp_3d: Target world/court-space 3D human joints.
            human_mask: Optional validity mask.

        Returns:
            dict: One entry per registered term plus ``"total"``.

        """
        inputs = self._build_inputs(
            pred_position,
            pred_rotation,
            target_position,
            target_rotation,
            pred_canonical_pose=pred_canonical_pose,
            target_human_kp_3d=target_human_kp_3d,
            human_mask=human_mask,
        )

        losses: dict[str, Tensor] = {}
        total = inputs.zero

        for name, term_fn in self.loss_terms.items():
            value = term_fn(inputs)
            losses[name] = value
            total = total + self.weight_for(name) * value

        losses["total"] = total
        return losses
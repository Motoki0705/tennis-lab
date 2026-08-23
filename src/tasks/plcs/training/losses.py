"""Loss functions for PLCS training.

Supports frame-level and sequence-level losses.
Temporal consistency is enforced by the GAN discriminator.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.configuration import (
    ConfigurationTypeError,
    SemanticConfigurationError,
)
from src.utils.geometry.angles import wrapped_angle_diff as _wrapped_angle_diff
from src.utils.geometry.court_pose import world_pose_to_canonical_pose
from src.utils.geometry.skeleton import (
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)
from src.utils.losses.temporal import TemporalSmoothnessPenalty
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)
from src.utils.schema.player import (
    COCO17_BONE_LENGTH_EDGES as BONE_LENGTH_EDGES,
)
from src.utils.schema.player import (
    COCO17_JOINT_ANGLE_TRIPLETS as JOINT_ANGLE_TRIPLETS,
)
from src.utils.schema.player import (
    COCO17_TORSION_QUADRUPLETS as TORSION_QUADRUPLETS,
)
from src.utils.tensor_utils import masked_mean

# Loss terms that require pred_canonical_pose and target_human_kp_3d.
CANONICAL_DEPENDENT_TERM_NAMES: tuple[str, ...] = (
    "canonical_pose",
    "joint_angle",
    "torsion_angle",
    "torso_twist",
    "bone_length",
    "joint_angle_velocity",
    "torsion_angle_velocity",
    "torso_twist_velocity",
)


@dataclass(frozen=True)
class PLCSLossConfig:
    """Configuration for PLCS loss weights.

    Attributes:
        position_weight: Weight for position loss.
        rotation_weight: Weight for rotation loss (1 - cosine on (cos, sin)).
        angle_weight: Weight for wrapped-angle smooth-L1 yaw loss.
        canonical_pose_weight: Weight for canonical pose loss.
        joint_angle_weight: Weight for joint-angle loss.
        torsion_angle_weight: Weight for limb torsion / dihedral angle loss.
        torso_twist_weight: Weight for shoulder-hip twist loss.
        bone_length_weight: Weight for bone-length consistency loss.

    """

    position_weight: float
    rotation_weight: float
    angle_weight: float
    # Temporal jerk prior on predicted player position (removes per-frame
    # jitter / inference velocity spikes). See src/utils/losses/temporal.py.
    position_smoothness_weight: float
    canonical_pose_weight: float
    joint_angle_weight: float
    torsion_angle_weight: float
    torso_twist_weight: float
    bone_length_weight: float
    # Angular-velocity (temporal) loss weights on canonical-pose angles (#521).
    joint_angle_velocity_weight: float
    torsion_angle_velocity_weight: float
    torso_twist_velocity_weight: float
    # Optional per-angle dominance weights (GT-derived) for the velocity terms.
    # None -> uniform. Lengths: 12 joint angles, 4 torsion angles, 1 twist.
    joint_angle_velocity_angle_weights: tuple[float, ...] | None
    torsion_angle_velocity_angle_weights: tuple[float, ...] | None
    # ``None`` preserves the historical normalized beta=1 for v1 and selects
    # the documented 1.0 m transition for v2.
    position_huber_beta_m: float | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, object]) -> PLCSLossConfig:
        """Create config from dictionary, e.g. loaded from YAML."""
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(cfg) - fields)
        missing = sorted(fields - {"position_huber_beta_m"} - set(cfg))
        if unknown or missing:
            raise ValueError(
                f"Invalid PLCS loss keys: missing={missing}, unknown={unknown}."
            )

        scalar_keys = fields - {
            "joint_angle_velocity_angle_weights",
            "torsion_angle_velocity_angle_weights",
            "position_huber_beta_m",
        }

        def _weight(key: str) -> float:
            value = cfg[key]
            if type(value) not in {float, int}:
                raise ConfigurationTypeError(
                    f"loss.{key}: expected float | int, got {type(value).__name__}."
                )
            result = float(cast("float | int", value))
            if not math.isfinite(result) or result < 0.0:
                raise SemanticConfigurationError(
                    f"loss.{key} must be finite and non-negative."
                )
            return result

        weights = {key: _weight(key) for key in scalar_keys}

        def _opt_weights(key: str, *, expected_length: int) -> tuple[float, ...] | None:
            value = cfg[key]
            if value is None:
                return None
            if type(value) not in {list, tuple}:
                raise ConfigurationTypeError(
                    f"loss.{key}: expected list | tuple | None, "
                    f"got {type(value).__name__}."
                )
            raw_values: tuple[object, ...] = tuple(
                cast("Sequence[object]", value)
            )
            if len(raw_values) != expected_length:
                raise SemanticConfigurationError(
                    f"loss.{key} must contain exactly {expected_length} values."
                )
            parsed: list[float] = []
            for index, raw in enumerate(raw_values):
                if type(raw) not in {float, int}:
                    raise ConfigurationTypeError(
                        f"loss.{key}[{index}]: expected float | int, "
                        f"got {type(raw).__name__}."
                    )
                number = float(cast("float | int", raw))
                if not math.isfinite(number) or number < 0.0:
                    raise SemanticConfigurationError(
                        f"loss.{key}[{index}] must be finite and non-negative."
                    )
                parsed.append(number)
            if not any(parsed):
                raise SemanticConfigurationError(
                    f"loss.{key} must contain at least one positive value."
                )
            return tuple(parsed)

        raw_beta = cfg.get("position_huber_beta_m")
        if raw_beta is not None:
            if type(raw_beta) not in {float, int}:
                raise ConfigurationTypeError(
                    "loss.position_huber_beta_m: expected float | int | None, "
                    f"got {type(raw_beta).__name__}."
                )
            position_huber_beta_m = float(cast("float | int", raw_beta))
            if not math.isfinite(position_huber_beta_m) or position_huber_beta_m <= 0:
                raise SemanticConfigurationError(
                    "loss.position_huber_beta_m must be finite and positive."
                )
        else:
            position_huber_beta_m = None

        return cls(
            position_weight=weights["position_weight"],
            rotation_weight=weights["rotation_weight"],
            angle_weight=weights["angle_weight"],
            position_smoothness_weight=weights["position_smoothness_weight"],
            canonical_pose_weight=weights["canonical_pose_weight"],
            joint_angle_weight=weights["joint_angle_weight"],
            torsion_angle_weight=weights["torsion_angle_weight"],
            torso_twist_weight=weights["torso_twist_weight"],
            bone_length_weight=weights["bone_length_weight"],
            joint_angle_velocity_weight=weights["joint_angle_velocity_weight"],
            torsion_angle_velocity_weight=weights[
                "torsion_angle_velocity_weight"
            ],
            torso_twist_velocity_weight=weights["torso_twist_velocity_weight"],
            joint_angle_velocity_angle_weights=_opt_weights(
                "joint_angle_velocity_angle_weights", expected_length=12
            ),
            torsion_angle_velocity_angle_weights=_opt_weights(
                "torsion_angle_velocity_angle_weights", expected_length=4
            ),
            position_huber_beta_m=position_huber_beta_m,
        )


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


@dataclass(frozen=True)
class PLCSPreparedLossTerms:
    """Named scalar terms prepared before entering the loss ``forward``."""

    terms: tuple[tuple[str, Tensor, float], ...]
    zero: Tensor


def _masked_frame_mean(per_frame: Tensor, frame_mask: Tensor | None) -> Tensor:
    """Reduce per-frame losses, honoring the padding mask when shapes match."""
    if frame_mask is not None and per_frame.shape == frame_mask.shape:
        return masked_mean(per_frame, frame_mask, binarize=True, denom_min=1.0)
    return per_frame.mean()


def position_loss_term(inputs: PLCSLossInputs, *, beta: float = 1.0) -> Tensor:
    """Position term: masked smooth-L1 between predicted and target positions."""
    per_frame = nn.functional.smooth_l1_loss(
        inputs.pred_position,
        inputs.target_position,
        reduction="none",
        beta=beta,
    ).mean(dim=-1)
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def rotation_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Rotation term: masked ``1 - cosine similarity`` on ``(cos, sin)``."""
    pred_norm = nn.functional.normalize(inputs.pred_rotation, dim=-1)
    target_norm = nn.functional.normalize(inputs.target_rotation, dim=-1)
    per_frame = 1.0 - (pred_norm * target_norm).sum(dim=-1)
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def angle_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Angle term: masked smooth-L1 on the wrapped yaw error (radians).

    Unlike the ``1 - cos`` rotation term, the gradient magnitude of this
    wrapped-angle loss stays ~constant all the way out to a 180-degree error
    instead of vanishing as the error approaches 180 degrees. The cosine loss
    has a flat saddle at the antipode, which makes front/back flips a sticky
    equilibrium the optimizer cannot escape; the angle term supplies a strong
    restoring gradient there, so the two terms are complementary (cosine is
    smooth near 0, angle keeps pushing near 180).
    """
    pred_angle = torch.atan2(inputs.pred_rotation[..., 1], inputs.pred_rotation[..., 0])
    target_angle = torch.atan2(
        inputs.target_rotation[..., 1], inputs.target_rotation[..., 0]
    )
    diff = _wrapped_angle_diff(pred_angle, target_angle)
    per_frame = nn.functional.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def position_smoothness_loss_term(
    inputs: PLCSLossInputs,
    *,
    penalty: TemporalSmoothnessPenalty,
) -> Tensor:
    """Temporal jerk prior on predicted player position.

    Player pelvis motion is smooth: on the broadcast test split the GT position
    acceleration is ~20-30x smaller than the model's raw prediction, so the
    excess is physically implausible per-frame jitter (the source of the
    inference-time velocity spikes). Penalizing jerk (3rd temporal difference)
    removes it without biasing the real, slowly-varying locomotion acceleration.
    No-op for frame-level (non-sequential) inputs.
    """
    if inputs.pred_position.ndim < 3:
        return inputs.zero
    frame_mask = inputs.frame_mask
    if frame_mask is None:
        frame_mask = torch.ones_like(inputs.pred_position[..., 0], dtype=torch.bool)
    return cast("Tensor", penalty(inputs.pred_position, frame_mask))


def canonical_pose_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Canonical-pose term: masked smooth-L1 between canonical joint positions."""
    if not inputs.has_canonical:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(Tensor, inputs.target_canonical_pose)

    per_frame = nn.functional.smooth_l1_loss(
        pred,
        target,
        reduction="none",
    ).mean(dim=(-1, -2))
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def joint_angle_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Joint-angle term on canonical joints.

    Includes both articulated limb angles and torso quadrilateral angles.
    """
    if not inputs.has_canonical:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(Tensor, inputs.target_canonical_pose)

    per_frame = joint_angle_loss(
        pred,
        target,
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def torsion_angle_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Limb torsion term on canonical joints.

    Captures whether arms/legs bend in the correct 3D plane.
    """
    if not inputs.has_canonical:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(Tensor, inputs.target_canonical_pose)

    per_frame = torsion_angle_loss(
        pred,
        target,
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def torso_twist_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Torso twist term on canonical joints.

    Captures shoulder-line vs hip-line twist around the torso axis.
    """
    if not inputs.has_canonical:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(Tensor, inputs.target_canonical_pose)

    per_frame = torso_twist_loss(
        pred,
        target,
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def bone_length_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Bone-length consistency term on canonical joints."""
    if not inputs.has_canonical:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(Tensor, inputs.target_canonical_pose)

    per_frame = bone_length_loss(
        pred,
        target,
        reduction="none",
    )
    return _masked_frame_mean(per_frame, inputs.frame_mask)


# ---------------------------------------------------------------------------
# Angular-velocity (temporal smoothness) losses (#521)
# ---------------------------------------------------------------------------

# Type of a canonical-pose angle extractor, e.g. compute_joint_angles.
AngleFn = Callable[[Tensor], Tensor]


def _angle_velocity_loss_term(
    inputs: PLCSLossInputs,
    compute_fn: AngleFn,
    *,
    wrap: bool,
    angle_weights: Tensor | None = None,
) -> Tensor:
    """Temporal angular-velocity loss on canonical-pose angles.

    Compares the per-angle frame-to-frame angular velocity (first temporal
    difference) of the predicted vs. target canonical pose. This directly
    supervises *motion* rather than static pose, addressing predictions that
    collapse to a frozen average pose (#521). ``angle_weights`` upweights angles
    whose GT velocity is dominant; ``wrap`` wraps the velocity difference into
    ``[-pi, pi]`` for signed/periodic angles (torsion, twist).
    """
    if not inputs.has_canonical:
        return inputs.zero
    pred = inputs.pred_canonical_pose
    target = inputs.target_canonical_pose
    pred = cast(Tensor, pred)
    target = cast(Tensor, target)
    # Need a temporal axis of >= 2 frames: canonical pose is (B, T, J, 3).
    if pred.ndim < 4 or pred.shape[-3] < 2:
        return inputs.zero

    pred_a = compute_fn(pred)
    target_a = compute_fn(target)
    # twist returns (B, T) with no angle axis -> add a singleton angle axis.
    if pred_a.ndim == pred.ndim - 2:
        pred_a = pred_a.unsqueeze(-1)
        target_a = target_a.unsqueeze(-1)

    # Velocity along the time axis (second to last): shape (B, T-1, A).
    pred_vel = pred_a[..., 1:, :] - pred_a[..., :-1, :]
    target_vel = target_a[..., 1:, :] - target_a[..., :-1, :]
    diff = pred_vel - target_vel
    if wrap:
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))

    per_angle = nn.functional.smooth_l1_loss(
        diff, torch.zeros_like(diff), reduction="none"
    )
    if angle_weights is not None:
        per_angle = per_angle * angle_weights.to(per_angle)
    per_frame = per_angle.mean(dim=-1)  # (B, T-1)

    frame_mask = inputs.frame_mask
    if frame_mask is not None and frame_mask.shape == pred_a.shape[:-1]:
        velocity_mask = (frame_mask[..., 1:] > 0) & (frame_mask[..., :-1] > 0)
        return masked_mean(per_frame, velocity_mask, binarize=True, denom_min=1.0)
    return per_frame.mean()


def joint_angle_velocity_loss_term(
    inputs: PLCSLossInputs, *, angle_weights: Tensor | None = None
) -> Tensor:
    """Joint-angle angular-velocity term (limb + torso interior angles)."""
    return _angle_velocity_loss_term(
        inputs, compute_joint_angles, wrap=False, angle_weights=angle_weights
    )


def torsion_angle_velocity_loss_term(
    inputs: PLCSLossInputs, *, angle_weights: Tensor | None = None
) -> Tensor:
    """Limb torsion/dihedral angular-velocity term (signed, wrapped)."""
    return _angle_velocity_loss_term(
        inputs, compute_torsion_angles, wrap=True, angle_weights=angle_weights
    )


def torso_twist_velocity_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Torso shoulder-hip twist angular-velocity term (signed, wrapped)."""
    return _angle_velocity_loss_term(
        inputs, compute_torso_twist, wrap=True, angle_weights=None
    )


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
    "angle": angle_loss_term,
    "position_smoothness": cast("PLCSLossTerm", position_smoothness_loss_term),
    "canonical_pose": canonical_pose_loss_term,
    "joint_angle": joint_angle_loss_term,
    "torsion_angle": torsion_angle_loss_term,
    "torso_twist": torso_twist_loss_term,
    "bone_length": bone_length_loss_term,
    "joint_angle_velocity": joint_angle_velocity_loss_term,
    "torsion_angle_velocity": torsion_angle_velocity_loss_term,
    "torso_twist_velocity": torso_twist_velocity_loss_term,
}


class PLCSLoss(nn.Module):
    """Combined loss for PLCS training.

    Each loss term is an independent function with the uniform signature
    ``(PLCSLossInputs) -> Tensor``. The module holds a ``{name: function}``
    registry. The boundary-facing :meth:`prepare_inputs` builds the shared
    :class:`PLCSLossInputs`; ``forward`` only evaluates and combines tensor terms.

    Supports both frame-level and sequence-level inputs:
        - Frame-level: ``(B, 3)``, ``(B, 2)``
        - Sequence-level: ``(B, T, 3)``, ``(B, T, 2)``

    Temporal consistency is enforced by the GAN discriminator.
    """

    def __init__(
        self,
        config: PLCSLossConfig,
        *,
        normalization: CourtCoordinateNormalization | str = "v1",
        loss_terms: dict[str, PLCSLossTerm] | None = None,
    ) -> None:
        """Initialize the loss module.

        Args:
            config: Complete validated loss configuration.
            loss_terms: Optional custom loss registry.

        """
        super().__init__()
        self.config = config
        self.court_coordinate_normalization = (
            normalization
            if isinstance(normalization, CourtCoordinateNormalization)
            else resolve_court_coordinate_normalization(normalization)
        )
        self.position_huber_beta = self._resolve_position_huber_beta()
        self.position_smoothness_penalty = TemporalSmoothnessPenalty(
            order=3,
            beta=1e-3,
            axis_weights=(1.0, 1.0, 1.0),
        )
        self.loss_terms: dict[str, PLCSLossTerm]
        if loss_terms is None:
            self.loss_terms = dict(DEFAULT_LOSS_TERMS)
            self.loss_terms["position"] = partial(
                position_loss_term,
                beta=self.position_huber_beta,
            )
            self.loss_terms["position_smoothness"] = partial(
                position_smoothness_loss_term,
                penalty=self.position_smoothness_penalty,
            )
        else:
            self.loss_terms = dict(loss_terms)
        self.loss_weights = {
            name: float(getattr(self.config, f"{name}_weight"))
            for name in self.loss_terms
        }
        self._bind_velocity_angle_weights()

    def _resolve_position_huber_beta(self) -> float:
        """Resolve the normalized beta without changing legacy v1 numerics."""
        configured_m = self.config.position_huber_beta_m
        contract = self.court_coordinate_normalization
        if contract.version == "v1":
            if configured_m is not None:
                raise ValueError(
                    "A single physical position Huber beta is not representable "
                    "under anisotropic v1; leave loss.position_huber_beta_m null "
                    "to preserve the historical normalized beta=1."
                )
            return 1.0
        physical_beta_m = 1.0 if configured_m is None else configured_m
        scale_x, scale_y, scale_z = contract.scale_xyz
        if scale_x != scale_y or scale_y != scale_z:
            raise ValueError("PLCS v2 physical Huber beta requires isotropic scale.")
        return physical_beta_m / scale_x

    def _bind_velocity_angle_weights(self) -> None:
        """Rebind velocity terms with GT-derived per-angle dominance weights.

        Weights are normalized to mean 1.0 so the configured ``*_velocity_weight``
        still controls the overall term scale; relative magnitudes upweight the
        angles whose GT angular velocity is dominant (#521).
        """

        def _norm(values: tuple[float, ...] | None) -> Tensor | None:
            if not values:
                return None
            tensor = torch.tensor(values, dtype=torch.float32)
            mean = tensor.mean().clamp_min(1e-8)
            return tensor / mean

        joint_w = _norm(self.config.joint_angle_velocity_angle_weights)
        if joint_w is not None:
            self.loss_terms["joint_angle_velocity"] = partial(
                joint_angle_velocity_loss_term, angle_weights=joint_w
            )
        torsion_w = _norm(self.config.torsion_angle_velocity_angle_weights)
        if torsion_w is not None:
            self.loss_terms["torsion_angle_velocity"] = partial(
                torsion_angle_velocity_loss_term, angle_weights=torsion_w
            )

    def weight_for(self, name: str) -> float:
        """Return the configured weight for a registered loss term."""
        return self.loss_weights[name]

    def _requires_canonical_pose(self) -> bool:
        """Whether any enabled registered term needs canonical pose tensors."""
        return any(
            name in self.loss_terms and self.loss_weights[name] > 0.0
            for name in CANONICAL_DEPENDENT_TERM_NAMES
        )

    def prepare_inputs(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
        *,
        pred_canonical_pose: Tensor | None,
        target_human_kp_3d: Tensor | None,
        padding_mask: Tensor | None,
    ) -> PLCSPreparedLossTerms:
        """Validate inputs and compute named scalar terms before ``forward``."""
        frame_mask = None
        if padding_mask is not None:
            frame_mask = ~(
                padding_mask.all(dim=1)
                if padding_mask.ndim == 3
                else padding_mask
            )

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
                    normalization=self.court_coordinate_normalization,
                )

        source = PLCSLossInputs(
            pred_position=pred_position,
            pred_rotation=pred_rotation,
            target_position=target_position,
            target_rotation=target_rotation,
            frame_mask=frame_mask,
            pred_canonical_pose=pred_canonical_pose,
            target_canonical_pose=target_canonical_pose,
        )
        return PLCSPreparedLossTerms(
            terms=tuple(
                (name, term_fn(source), self.loss_weights[name])
                for name, term_fn in self.loss_terms.items()
            ),
            zero=source.zero,
        )

    def forward(self, inputs: PLCSPreparedLossTerms) -> dict[str, Tensor]:
        """Combine boundary-prepared scalar terms with configured weights."""
        losses: dict[str, Tensor] = {}
        total = inputs.zero

        for name, value, weight in inputs.terms:
            losses[name] = value
            total = total + weight * value

        losses["total"] = total
        return losses

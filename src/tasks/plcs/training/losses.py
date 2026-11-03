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

from src.tasks.plcs.model_io.contracts import PLCSReprojectionTarget
from src.utils.configuration import (
    ConfigurationTypeError,
    SemanticConfigurationError,
)
from src.utils.geometry.angles import wrapped_angle_diff as _wrapped_angle_diff
from src.utils.geometry.court_pose import (
    canonical_pose_to_world_pose,
    world_pose_to_canonical_pose,
)
from src.utils.geometry.skeleton import (
    SkeletonValues,
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)
from src.utils.losses.temporal import TemporalSmoothnessPenalty
from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
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
        position_smooth_l1_beta: Smooth-L1 transition beta in normalized court
            coordinates for position loss.
        rotation_weight: Weight for rotation loss (1 - cosine on (cos, sin)).
        angle_weight: Weight for wrapped-angle smooth-L1 yaw loss.
        canonical_pose_weight: Weight for canonical pose loss.
        canonical_pose_smooth_l1_beta: Smooth-L1 transition beta in metres for
            canonical pose loss.
        reprojection_weight: Weight for clean 2D pose reprojection loss.
        reprojection_smooth_l1_beta: Smooth-L1 transition beta in normalized UV.
        joint_angle_weight: Weight for joint-angle loss.
        torsion_angle_weight: Weight for limb torsion / dihedral angle loss.
        torso_twist_weight: Weight for shoulder-hip twist loss.
        bone_length_weight: Weight for bone-length consistency loss.

    """

    position_weight: float
    position_smooth_l1_beta: float
    rotation_weight: float
    angle_weight: float
    # Temporal jerk prior on predicted player position (removes per-frame
    # jitter / inference velocity spikes). See src/utils/losses/temporal.py.
    position_smoothness_weight: float
    canonical_pose_weight: float
    canonical_pose_smooth_l1_beta: float
    reprojection_weight: float
    reprojection_smooth_l1_beta: float
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

    @classmethod
    def from_dict(cls, cfg: dict[str, object]) -> PLCSLossConfig:
        """Create config from dictionary, e.g. loaded from YAML."""
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(cfg) - fields)
        missing = sorted(fields - set(cfg))
        if unknown or missing:
            raise ValueError(
                f"Invalid PLCS loss keys: missing={missing}, unknown={unknown}."
            )

        scalar_keys = fields - {
            "joint_angle_velocity_angle_weights",
            "torsion_angle_velocity_angle_weights",
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
        if weights["position_smooth_l1_beta"] <= 0.0:
            raise SemanticConfigurationError(
                "loss.position_smooth_l1_beta must be positive."
            )
        if weights["canonical_pose_smooth_l1_beta"] <= 0.0:
            raise SemanticConfigurationError(
                "loss.canonical_pose_smooth_l1_beta must be positive."
            )
        if weights["reprojection_smooth_l1_beta"] <= 0.0:
            raise SemanticConfigurationError(
                "loss.reprojection_smooth_l1_beta must be positive."
            )

        def _opt_weights(key: str, *, expected_length: int) -> tuple[float, ...] | None:
            value = cfg[key]
            if value is None:
                return None
            if type(value) not in {list, tuple}:
                raise ConfigurationTypeError(
                    f"loss.{key}: expected list | tuple | None, "
                    f"got {type(value).__name__}."
                )
            raw_values: tuple[object, ...] = tuple(cast("Sequence[object]", value))
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

        return cls(
            position_weight=weights["position_weight"],
            position_smooth_l1_beta=weights["position_smooth_l1_beta"],
            rotation_weight=weights["rotation_weight"],
            angle_weight=weights["angle_weight"],
            position_smoothness_weight=weights["position_smoothness_weight"],
            canonical_pose_weight=weights["canonical_pose_weight"],
            canonical_pose_smooth_l1_beta=weights["canonical_pose_smooth_l1_beta"],
            reprojection_weight=weights["reprojection_weight"],
            reprojection_smooth_l1_beta=weights["reprojection_smooth_l1_beta"],
            joint_angle_weight=weights["joint_angle_weight"],
            torsion_angle_weight=weights["torsion_angle_weight"],
            torso_twist_weight=weights["torso_twist_weight"],
            bone_length_weight=weights["bone_length_weight"],
            joint_angle_velocity_weight=weights["joint_angle_velocity_weight"],
            torsion_angle_velocity_weight=weights["torsion_angle_velocity_weight"],
            torso_twist_velocity_weight=weights["torso_twist_velocity_weight"],
            joint_angle_velocity_angle_weights=_opt_weights(
                "joint_angle_velocity_angle_weights", expected_length=12
            ),
            torsion_angle_velocity_angle_weights=_opt_weights(
                "torsion_angle_velocity_angle_weights", expected_length=4
            ),
        )


# ---------------------------------------------------------------------------
# Pose naturalness losses
# ---------------------------------------------------------------------------


def _reduce_skeleton_features(
    per_feature: Tensor, valid: Tensor, reduction: str = "mean"
) -> Tensor:
    """Mean of observed features within a frame, then of observed frames.

    ``none`` returns per-frame means (zero storage for missing frames), while
    ``sum`` preserves the old sum of feature errors. Empty supervision raises.
    """
    if not bool(valid.any()):
        raise ValueError("No valid skeleton features/transitions in this loss term.")
    observed = torch.where(valid, per_feature, torch.zeros_like(per_feature))
    if reduction == "sum":
        return observed.sum()
    counts = valid.sum(dim=-1)
    per_frame = observed.sum(dim=-1) / counts.clamp_min(1)
    if reduction == "none":
        return per_frame
    if reduction != "mean":
        raise ValueError(f"Unknown skeleton loss reduction: {reduction!r}.")
    return per_frame[counts > 0].mean()


def _skeleton_error(
    pred: SkeletonValues, target: SkeletonValues, *, name: str, wrap: bool = False
) -> tuple[Tensor, Tensor]:
    pred.require_valid(f"pred {name}")
    diff = pred.values - target.values
    if wrap:
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff, pred.valid & target.valid


def _smooth_error(diff: Tensor) -> Tensor:
    return nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")


def joint_angle_loss(
    pred_pose: Tensor,
    target_pose: Tensor,
    triplets: tuple[tuple[int, int, int], ...] = JOINT_ANGLE_TRIPLETS,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Interior-angle smooth-L1 over observed target features; pred is strict.

    ``mean`` averages valid features per frame, then frames with observations.
    ``none`` returns frame means and zero storage for missing frames. Padding
    must already be removed. Nonfinite poses and all-missing targets raise.
    """
    diff, valid = _skeleton_error(
        compute_joint_angles(pred_pose, triplets),
        compute_joint_angles(target_pose, triplets),
        name="joint_angle",
    )
    return _reduce_skeleton_features(_smooth_error(diff), valid, reduction)


def torsion_angle_loss(
    pred_pose: Tensor,
    target_pose: Tensor,
    quadruplets: tuple[tuple[int, int, int, int], ...] = TORSION_QUADRUPLETS,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Wrapped dihedral smooth-L1 with the same missingness contract as joint loss."""
    diff, valid = _skeleton_error(
        compute_torsion_angles(pred_pose, quadruplets),
        compute_torsion_angles(target_pose, quadruplets),
        name="torsion_angle",
        wrap=True,
    )
    return _reduce_skeleton_features(_smooth_error(diff), valid, reduction)


def torso_twist_loss(
    pred_pose: Tensor,
    target_pose: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Wrapped hip-to-shoulder twist loss; missing torso frames are excluded."""
    diff, valid = _skeleton_error(
        compute_torso_twist(pred_pose),
        compute_torso_twist(target_pose),
        name="torso_twist",
        wrap=True,
    )
    return _reduce_skeleton_features(
        _smooth_error(diff).unsqueeze(-1), valid.unsqueeze(-1), reduction
    )


def bone_length_loss(
    pred_pose: Tensor,
    target_pose: Tensor,
    edges: tuple[tuple[int, int], ...] = BONE_LENGTH_EDGES,
    *,
    relative: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """Bone smooth-L1, relative by default, with zero/short target bones missing."""
    pred = compute_bone_lengths(pred_pose, edges)
    target = compute_bone_lengths(target_pose, edges)
    diff, valid = _skeleton_error(pred, target, name="bone_length")
    if relative:
        # Replace missing denominators BEFORE division, not after it.
        denominator = torch.where(valid, target.values, torch.ones_like(target.values))
        diff = diff / denominator
    return _reduce_skeleton_features(_smooth_error(diff), valid, reduction)


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
        reprojection_target: Optional clean 2D pose and camera bundle.

    """

    pred_position: Tensor
    pred_rotation: Tensor
    target_position: Tensor
    target_rotation: Tensor
    frame_mask: Tensor | None = None
    pred_canonical_pose: Tensor | None = None
    target_canonical_pose: Tensor | None = None
    reprojection_target: PLCSReprojectionTarget | None = None

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

    @property
    def has_reprojection(self) -> bool:
        """Whether predicted canonical pose and 2D/camera targets are available."""
        return (
            self.pred_canonical_pose is not None
            and self.reprojection_target is not None
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
    """Position term in normalized court coordinates with configurable beta."""
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


def canonical_pose_loss_term(
    inputs: PLCSLossInputs,
    *,
    beta: float = 1.0,
) -> Tensor:
    """Canonical-pose term: masked smooth-L1 between canonical joint positions.

    ``beta`` is expressed in metres because canonical pose coordinates remain
    root-relative physical coordinates and are not court-coordinate normalized.
    """
    if not inputs.has_canonical:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(Tensor, inputs.target_canonical_pose)

    per_frame = nn.functional.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=beta,
    ).mean(dim=(-1, -2))
    return _masked_frame_mean(per_frame, inputs.frame_mask)


def reprojection_loss_term(
    inputs: PLCSLossInputs,
    *,
    projector: DifferentiablePinholeProjection,
    beta: float,
) -> Tensor:
    """Project the composed predicted world pose and compare it to clean 2D pose."""
    if not inputs.has_reprojection:
        return inputs.zero
    canonical_pose = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(PLCSReprojectionTarget, inputs.reprojection_target)
    world_pose = canonical_pose_to_world_pose(
        canonical_pose,
        inputs.pred_position,
        inputs.pred_rotation,
    )
    pred_uv, _in_front = projector(
        world_points=world_pose,
        camera_R=target.camera_R,
        camera_C=target.camera_C,
        camera_f=target.camera_f,
        camera_cx=target.camera_cx,
        camera_cy=target.camera_cy,
        camera_w=target.camera_w,
        camera_h=target.camera_h,
    )
    if pred_uv.shape != target.target_uv.shape:
        raise ValueError(
            "Projected PLCS pose and 2D target must share shape, got "
            f"{tuple(pred_uv.shape)} and {tuple(target.target_uv.shape)}."
        )

    # Visibility is a property of the clean target. A prediction behind a
    # camera remains supervised instead of silently escaping the objective.
    valid = (target.target_vis > 0) & ~target.padding_mask.unsqueeze(-1)
    per_coordinate = nn.functional.smooth_l1_loss(
        pred_uv,
        target.target_uv,
        reduction="none",
        beta=beta,
    )
    return masked_mean(
        per_coordinate,
        valid.unsqueeze(-1),
        binarize=True,
        denom_min=1.0,
    )


def _valid_pose_frames(pose: Tensor, frame_mask: Tensor | None) -> Tensor:
    shape = pose.shape[:-2]
    if frame_mask is None:
        return torch.ones(shape, dtype=torch.bool, device=pose.device)
    if frame_mask.shape != shape:
        raise ValueError(
            f"Skeleton frame mask must have shape {shape}, got {frame_mask.shape}."
        )
    return frame_mask > 0


def _static_skeleton_loss_term(
    inputs: PLCSLossInputs, loss_fn: Callable[[Tensor, Tensor], Tensor]
) -> Tensor:
    if not inputs.has_canonical:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(Tensor, inputs.target_canonical_pose)
    mask = _valid_pose_frames(pred, inputs.frame_mask)
    # Exclude padding before finite validation and all nonlinear geometry.
    return loss_fn(pred[mask], target[mask])


def joint_angle_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Interior-angle loss, including articulated limbs and torso angles."""
    return _static_skeleton_loss_term(inputs, joint_angle_loss)


def torsion_angle_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Limb dihedral loss with missing target planes excluded."""
    return _static_skeleton_loss_term(inputs, torsion_angle_loss)


def torso_twist_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Shoulder-hip twist loss with missing target axes/planes excluded."""
    return _static_skeleton_loss_term(inputs, torso_twist_loss)


def bone_length_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Relative bone-length loss with missing target bones excluded."""
    return _static_skeleton_loss_term(inputs, bone_length_loss)


# ---------------------------------------------------------------------------
# Angular-velocity losses
# ---------------------------------------------------------------------------

AngleFn = Callable[[Tensor], SkeletonValues]


def _restore_frame_features(
    result: SkeletonValues, frame_mask: Tensor
) -> SkeletonValues:
    values, valid = result.values, result.valid
    if values.ndim == 1:  # Twist has no feature axis.
        values, valid = values.unsqueeze(-1), valid.unsqueeze(-1)
    full_values = values.new_zeros((*frame_mask.shape, values.shape[-1]))
    full_valid = torch.zeros_like(full_values, dtype=torch.bool)
    full_values[frame_mask], full_valid[frame_mask] = values, valid
    return SkeletonValues(full_values, full_valid)


def _angle_velocity_loss_term(
    inputs: PLCSLossInputs,
    compute_fn: AngleFn,
    *,
    wrap: bool,
    angle_weights: Tensor | None = None,
) -> Tensor:
    """Compare first differences using the AND of both endpoint feature masks.

    Per-angle dominance weights retain their configured scale; observed feature
    counts, rather than sums of weights, normalize each frame transition.
    """
    if not inputs.has_canonical:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_canonical_pose)
    target = cast(Tensor, inputs.target_canonical_pose)
    if pred.ndim < 4 or pred.shape[-3] < 2:
        raise ValueError(
            "No valid angular-velocity transitions: need at least two frames."
        )
    frame_mask = _valid_pose_frames(pred, inputs.frame_mask)
    pred_selected = compute_fn(pred[frame_mask])
    target_selected = compute_fn(target[frame_mask])
    pred_selected.require_valid("pred angular_velocity")
    pred_a = _restore_frame_features(pred_selected, frame_mask)
    target_a = _restore_frame_features(target_selected, frame_mask)
    observed = pred_a.valid & target_a.valid
    valid = observed[..., 1:, :] & observed[..., :-1, :]
    pred_vel = pred_a.values[..., 1:, :] - pred_a.values[..., :-1, :]
    target_vel = target_a.values[..., 1:, :] - target_a.values[..., :-1, :]
    diff = pred_vel - target_vel
    if wrap:
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    per_angle = _smooth_error(diff)
    if angle_weights is not None:
        per_angle = per_angle * angle_weights.to(per_angle)
    return _reduce_skeleton_features(per_angle, valid)


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
    "reprojection": cast("PLCSLossTerm", reprojection_loss_term),
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
        loss_terms: dict[str, PLCSLossTerm] | None = None,
    ) -> None:
        """Initialize the loss module.

        Args:
            config: Complete validated loss configuration.
            loss_terms: Optional custom loss registry.

        """
        super().__init__()
        self.config = config
        self.position_smoothness_penalty = TemporalSmoothnessPenalty(
            order=3,
            beta=1e-3,
            axis_weights=(1.0, 1.0, 1.0),
        )
        self.reprojection_projector = DifferentiablePinholeProjection()
        self.loss_terms: dict[str, PLCSLossTerm]
        if loss_terms is None:
            self.loss_terms = dict(DEFAULT_LOSS_TERMS)
            self.loss_terms["position"] = cast(
                "PLCSLossTerm",
                partial(
                    position_loss_term,
                    beta=config.position_smooth_l1_beta,
                ),
            )
            self.loss_terms["position_smoothness"] = partial(
                position_smoothness_loss_term,
                penalty=self.position_smoothness_penalty,
            )
            self.loss_terms["canonical_pose"] = cast(
                "PLCSLossTerm",
                partial(
                    canonical_pose_loss_term,
                    beta=config.canonical_pose_smooth_l1_beta,
                ),
            )
            self.loss_terms["reprojection"] = cast(
                "PLCSLossTerm",
                partial(
                    reprojection_loss_term,
                    projector=self.reprojection_projector,
                    beta=config.reprojection_smooth_l1_beta,
                ),
            )
        else:
            self.loss_terms = dict(loss_terms)
        self.loss_weights = {
            name: float(getattr(self.config, f"{name}_weight"))
            for name in self.loss_terms
        }
        self._bind_velocity_angle_weights()

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

    def _requires_reprojection(self) -> bool:
        return (
            "reprojection" in self.loss_terms
            and self.loss_weights["reprojection"] > 0.0
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
        reprojection_target: PLCSReprojectionTarget | None = None,
    ) -> PLCSPreparedLossTerms:
        """Validate inputs and compute named scalar terms before ``forward``."""
        frame_mask = None
        if padding_mask is not None:
            frame_mask = ~(
                padding_mask.all(dim=1) if padding_mask.ndim == 3 else padding_mask
            )

        canonical_required = self._requires_canonical_pose()
        reprojection_required = self._requires_reprojection()
        if (
            canonical_required or reprojection_required
        ) and pred_canonical_pose is None:
            raise ValueError(
                "pred_canonical_pose is required when canonical-dependent or "
                "reprojection loss weight is > 0."
            )
        if reprojection_required and reprojection_target is None:
            raise ValueError(
                "reprojection_target is required when reprojection_weight is > 0."
            )

        target_canonical_pose: Tensor | None = None
        if canonical_required:
            if target_human_kp_3d is None:
                raise ValueError(
                    "target_human_kp_3d is required when any "
                    "canonical-dependent loss weight is > 0 and "
                    "pred_canonical_pose is provided."
                )
            mask = _valid_pose_frames(target_human_kp_3d, frame_mask)
            target_canonical_pose = torch.zeros_like(target_human_kp_3d)
            target_canonical_pose[mask] = world_pose_to_canonical_pose(
                target_human_kp_3d[mask],
                target_position[mask],
                target_rotation[mask],
            )

        source = PLCSLossInputs(
            pred_position=pred_position,
            pred_rotation=pred_rotation,
            target_position=target_position,
            target_rotation=target_rotation,
            frame_mask=frame_mask,
            pred_canonical_pose=pred_canonical_pose,
            target_canonical_pose=target_canonical_pose,
            reprojection_target=reprojection_target,
        )
        return PLCSPreparedLossTerms(
            terms=tuple(
                (name, term_fn(source), self.loss_weights[name])
                for name, term_fn in self.loss_terms.items()
                if self.loss_weights[name] > 0.0
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

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
    # Monocular physical world-pose and HTD-Refine Stage A PVA-Heads weights.
    world_kp_3d_weight: float = 0.0
    pva_velocity_weight: float = 0.0
    pva_acceleration_weight: float = 0.0
    kinematic_velocity_consistency_weight: float = 0.0
    kinematic_acceleration_consistency_weight: float = 0.0
    static_contact_weight: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict[str, object]) -> PLCSLossConfig:
        """Create config from dictionary, e.g. loaded from YAML."""
        optional_monocular_defaults: dict[str, float] = {
            "world_kp_3d_weight": 0.0,
            "pva_velocity_weight": 0.0,
            "pva_acceleration_weight": 0.0,
            "kinematic_velocity_consistency_weight": 0.0,
            "kinematic_acceleration_consistency_weight": 0.0,
            "static_contact_weight": 0.0,
        }
        fields = {field.name for field in dataclasses.fields(cls)}
        required_fields = fields - set(optional_monocular_defaults)
        unknown = sorted(set(cfg) - fields)
        missing = sorted(required_fields - set(cfg))
        if unknown or missing:
            raise ValueError(
                f"Invalid PLCS loss keys: missing={missing}, unknown={unknown}."
            )

        scalar_keys = fields - {
            "joint_angle_velocity_angle_weights",
            "torsion_angle_velocity_angle_weights",
        }

        def _weight(key: str) -> float:
            if key not in cfg and key in optional_monocular_defaults:
                return optional_monocular_defaults[key]
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
            world_kp_3d_weight=weights["world_kp_3d_weight"],
            pva_velocity_weight=weights["pva_velocity_weight"],
            pva_acceleration_weight=weights["pva_acceleration_weight"],
            kinematic_velocity_consistency_weight=weights[
                "kinematic_velocity_consistency_weight"
            ],
            kinematic_acceleration_consistency_weight=weights[
                "kinematic_acceleration_consistency_weight"
            ],
            static_contact_weight=weights["static_contact_weight"],
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
    pred_world_kp_3d: Tensor | None = None
    target_world_kp_3d: Tensor | None = None
    pred_vel_3d: Tensor | None = None
    pred_acc_3d: Tensor | None = None
    pred_static_contact_logits: Tensor | None = None
    frame_rate_hz: Tensor | None = None

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
        """Whether predicted pose and 2D/camera targets are available."""
        return (
            self.pred_world_kp_3d is not None or self.pred_canonical_pose is not None
        ) and self.reprojection_target is not None

    @property
    def has_world_pose(self) -> bool:
        """Whether both predicted and target physical court 3D poses exist."""
        return (
            self.pred_world_kp_3d is not None and self.target_world_kp_3d is not None
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
    target = cast(PLCSReprojectionTarget, inputs.reprojection_target)
    if inputs.pred_world_kp_3d is not None:
        world_pose = inputs.pred_world_kp_3d
    else:
        canonical_pose = cast(Tensor, inputs.pred_canonical_pose)
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


# ---------------------------------------------------------------------------
# Physical world-pose and HTD-Refine Stage A PVA-Heads losses
# ---------------------------------------------------------------------------

STATIC_CONTACT_JOINT_INDICES: tuple[int, int, int, int] = (15, 16, 13, 14)
_STATIC_CONTACT_SPEED_THRESHOLD_MPS = 0.25
_STATIC_CONTACT_HEIGHT_THRESHOLDS_M: tuple[float, float, float, float] = (
    0.15,
    0.15,
    0.70,
    0.70,
)


def _resolve_sequence_fps(
    frame_rate_hz: Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if frame_rate_hz is None:
        return torch.full((batch_size, 1, 1, 1), 30.0, device=device, dtype=dtype)
    fps = frame_rate_hz.to(device=device, dtype=dtype)
    if fps.ndim == 0:
        fps = fps.expand(batch_size)
    elif fps.ndim != 1 or fps.shape[0] != batch_size:
        raise ValueError(
            "frame_rate_hz must be scalar or shape (B,), got "
            f"{tuple(fps.shape)} for batch_size={batch_size}."
        )
    if not bool(torch.isfinite(fps).all().item()) or bool((fps <= 0.0).any().item()):
        raise ValueError("frame_rate_hz must be strictly positive and finite.")
    return fps.view(batch_size, 1, 1, 1)


def _safe_masked_sequence(tensor: Tensor, frame_valid: Tensor) -> Tensor:
    mask = frame_valid
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, tensor, torch.zeros_like(tensor))


def world_kp_3d_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Physical court-space 3D COCO-17 keypoint smooth-L1 loss (metres)."""
    if not inputs.has_world_pose:
        return inputs.zero
    pred = cast(Tensor, inputs.pred_world_kp_3d)
    target = cast(Tensor, inputs.target_world_kp_3d)
    frame_valid = _valid_pose_frames(pred, inputs.frame_mask)
    pred_safe = _safe_masked_sequence(pred, frame_valid)
    target_safe = _safe_masked_sequence(target, frame_valid)
    per_frame = nn.functional.smooth_l1_loss(
        pred_safe,
        target_safe,
        reduction="none",
        beta=0.1,
    ).mean(dim=(-1, -2))
    return _masked_frame_mean(per_frame, frame_valid)


def pva_velocity_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Stage A PVA-Heads 3D joint velocity loss strictly within valid transitions."""
    if inputs.pred_vel_3d is None or inputs.target_world_kp_3d is None:
        return inputs.zero
    pred_vel = inputs.pred_vel_3d
    target_kp = inputs.target_world_kp_3d
    if target_kp.ndim != 4 or target_kp.shape[1] < 2:
        return inputs.zero
    frame_valid = _valid_pose_frames(target_kp, inputs.frame_mask)
    valid_v = frame_valid[:, 1:] & frame_valid[:, :-1]
    if not bool(valid_v.any().item()):
        return inputs.zero
    fps = _resolve_sequence_fps(
        inputs.frame_rate_hz,
        batch_size=target_kp.shape[0],
        device=target_kp.device,
        dtype=target_kp.dtype,
    )
    target_safe = _safe_masked_sequence(target_kp, frame_valid)
    v_gt = (target_safe[:, 1:] - target_safe[:, :-1]) * fps
    pred_v_safe = _safe_masked_sequence(pred_vel[:, :-1], valid_v)
    per_step = nn.functional.smooth_l1_loss(
        pred_v_safe,
        v_gt,
        reduction="none",
        beta=0.5,
    ).mean(dim=(-1, -2))
    return _masked_frame_mean(per_step, valid_v)


def pva_acceleration_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Stage A PVA-Heads 3D joint acceleration loss strictly within 3-frame windows."""
    if inputs.pred_acc_3d is None or inputs.target_world_kp_3d is None:
        return inputs.zero
    pred_acc = inputs.pred_acc_3d
    target_kp = inputs.target_world_kp_3d
    if target_kp.ndim != 4 or target_kp.shape[1] < 3:
        return inputs.zero
    frame_valid = _valid_pose_frames(target_kp, inputs.frame_mask)
    valid_a = frame_valid[:, 2:] & frame_valid[:, 1:-1] & frame_valid[:, :-2]
    if not bool(valid_a.any().item()):
        return inputs.zero
    fps = _resolve_sequence_fps(
        inputs.frame_rate_hz,
        batch_size=target_kp.shape[0],
        device=target_kp.device,
        dtype=target_kp.dtype,
    )
    target_safe = _safe_masked_sequence(target_kp, frame_valid)
    a_gt = (
        target_safe[:, 2:] - 2.0 * target_safe[:, 1:-1] + target_safe[:, :-2]
    ) * (fps * fps)
    pred_a_safe = _safe_masked_sequence(pred_acc[:, 1:-1], valid_a)
    per_step = nn.functional.smooth_l1_loss(
        pred_a_safe,
        a_gt,
        reduction="none",
        beta=2.0,
    ).mean(dim=(-1, -2))
    return _masked_frame_mean(per_step, valid_a)


def kinematic_velocity_consistency_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """First-order kinematic consistency between predicted world pose and GT/PVA."""
    if not inputs.has_world_pose:
        return inputs.zero
    pred_kp = cast(Tensor, inputs.pred_world_kp_3d)
    target_kp = cast(Tensor, inputs.target_world_kp_3d)
    if pred_kp.ndim != 4 or pred_kp.shape[1] < 2:
        return inputs.zero
    frame_valid = _valid_pose_frames(pred_kp, inputs.frame_mask)
    valid_v = frame_valid[:, 1:] & frame_valid[:, :-1]
    if not bool(valid_v.any().item()):
        return inputs.zero
    fps = _resolve_sequence_fps(
        inputs.frame_rate_hz,
        batch_size=pred_kp.shape[0],
        device=pred_kp.device,
        dtype=pred_kp.dtype,
    )
    pred_safe = _safe_masked_sequence(pred_kp, frame_valid)
    target_safe = _safe_masked_sequence(target_kp, frame_valid)
    v_pred = (pred_safe[:, 1:] - pred_safe[:, :-1]) * fps
    v_gt = (target_safe[:, 1:] - target_safe[:, :-1]) * fps
    per_step = nn.functional.smooth_l1_loss(
        v_pred,
        v_gt,
        reduction="none",
        beta=0.5,
    ).mean(dim=(-1, -2))
    if inputs.pred_vel_3d is not None:
        pred_v_safe = _safe_masked_sequence(inputs.pred_vel_3d[:, :-1].detach(), valid_v)
        per_step = per_step + 0.5 * nn.functional.smooth_l1_loss(
            v_pred,
            pred_v_safe,
            reduction="none",
            beta=0.5,
        ).mean(dim=(-1, -2))
    return _masked_frame_mean(per_step, valid_v)


def kinematic_acceleration_consistency_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """Second-order kinematic consistency between predicted world pose and GT/PVA."""
    if not inputs.has_world_pose:
        return inputs.zero
    pred_kp = cast(Tensor, inputs.pred_world_kp_3d)
    target_kp = cast(Tensor, inputs.target_world_kp_3d)
    if pred_kp.ndim != 4 or pred_kp.shape[1] < 3:
        return inputs.zero
    frame_valid = _valid_pose_frames(pred_kp, inputs.frame_mask)
    valid_a = frame_valid[:, 2:] & frame_valid[:, 1:-1] & frame_valid[:, :-2]
    if not bool(valid_a.any().item()):
        return inputs.zero
    fps = _resolve_sequence_fps(
        inputs.frame_rate_hz,
        batch_size=pred_kp.shape[0],
        device=pred_kp.device,
        dtype=pred_kp.dtype,
    )
    pred_safe = _safe_masked_sequence(pred_kp, frame_valid)
    target_safe = _safe_masked_sequence(target_kp, frame_valid)
    a_pred = (
        pred_safe[:, 2:] - 2.0 * pred_safe[:, 1:-1] + pred_safe[:, :-2]
    ) * (fps * fps)
    a_gt = (
        target_safe[:, 2:] - 2.0 * target_safe[:, 1:-1] + target_safe[:, :-2]
    ) * (fps * fps)
    per_step = nn.functional.smooth_l1_loss(
        a_pred,
        a_gt,
        reduction="none",
        beta=2.0,
    ).mean(dim=(-1, -2))
    if inputs.pred_acc_3d is not None:
        pred_a_safe = _safe_masked_sequence(
            inputs.pred_acc_3d[:, 1:-1].detach(), valid_a
        )
        per_step = per_step + 0.5 * nn.functional.smooth_l1_loss(
            a_pred,
            pred_a_safe,
            reduction="none",
            beta=2.0,
        ).mean(dim=(-1, -2))
    return _masked_frame_mean(per_step, valid_a)


def static_contact_loss_term(inputs: PLCSLossInputs) -> Tensor:
    """BCE-with-logits for lower-body static contact on valid non-boundary frames."""
    if inputs.pred_static_contact_logits is None or inputs.target_world_kp_3d is None:
        return inputs.zero
    logits = inputs.pred_static_contact_logits
    target_kp = inputs.target_world_kp_3d
    if target_kp.ndim != 4 or target_kp.shape[1] < 2:
        return inputs.zero
    frame_valid = _valid_pose_frames(target_kp, inputs.frame_mask)
    valid_step = frame_valid[:, 1:] & frame_valid[:, :-1]
    if not bool(valid_step.any().item()):
        return inputs.zero
    fps = _resolve_sequence_fps(
        inputs.frame_rate_hz,
        batch_size=target_kp.shape[0],
        device=target_kp.device,
        dtype=target_kp.dtype,
    ).squeeze(-1)
    contact_kp = _safe_masked_sequence(
        target_kp[:, :, STATIC_CONTACT_JOINT_INDICES, :], frame_valid
    )
    step_speed = (
        torch.linalg.vector_norm(contact_kp[:, 1:] - contact_kp[:, :-1], dim=-1) * fps
    )
    batch_size, seq_len = frame_valid.shape
    num_c = len(STATIC_CONTACT_JOINT_INDICES)
    speed_sum = torch.zeros(
        (batch_size, seq_len, num_c), device=target_kp.device, dtype=target_kp.dtype
    )
    speed_cnt = torch.zeros(
        (batch_size, seq_len, 1), device=target_kp.device, dtype=target_kp.dtype
    )
    step_w = valid_step.to(dtype=target_kp.dtype).unsqueeze(-1)
    speed_sum[:, :-1] = speed_sum[:, :-1] + step_speed * step_w
    speed_cnt[:, :-1] = speed_cnt[:, :-1] + step_w
    speed_sum[:, 1:] = speed_sum[:, 1:] + step_speed * step_w
    speed_cnt[:, 1:] = speed_cnt[:, 1:] + step_w

    has_speed = frame_valid & (speed_cnt.squeeze(-1) > 0.0)
    if not bool(has_speed.any().item()):
        return inputs.zero
    mean_speed = speed_sum / speed_cnt.clamp_min(1.0)
    height_thresh = torch.tensor(
        _STATIC_CONTACT_HEIGHT_THRESHOLDS_M,
        device=target_kp.device,
        dtype=target_kp.dtype,
    ).view(1, 1, num_c)
    contact_target = (
        (mean_speed < _STATIC_CONTACT_SPEED_THRESHOLD_MPS)
        & (contact_kp[..., 2] < height_thresh)
    ).to(dtype=logits.dtype)
    logits_safe = _safe_masked_sequence(logits, has_speed)
    per_frame = nn.functional.binary_cross_entropy_with_logits(
        logits_safe,
        contact_target,
        reduction="none",
    ).mean(dim=-1)
    return _masked_frame_mean(per_frame, has_speed)


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
    "world_kp_3d": world_kp_3d_loss_term,
    "pva_velocity": pva_velocity_loss_term,
    "pva_acceleration": pva_acceleration_loss_term,
    "kinematic_velocity_consistency": kinematic_velocity_consistency_loss_term,
    "kinematic_acceleration_consistency": kinematic_acceleration_consistency_loss_term,
    "static_contact": static_contact_loss_term,
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

    def _requires_world_pose(self) -> bool:
        return any(
            name in self.loss_terms and self.loss_weights[name] > 0.0
            for name in (
                "world_kp_3d",
                "kinematic_velocity_consistency",
                "kinematic_acceleration_consistency",
            )
        )

    def _requires_pva_velocity(self) -> bool:
        return (
            "pva_velocity" in self.loss_terms
            and self.loss_weights["pva_velocity"] > 0.0
        )

    def _requires_pva_acceleration(self) -> bool:
        return (
            "pva_acceleration" in self.loss_terms
            and self.loss_weights["pva_acceleration"] > 0.0
        )

    def _requires_static_contact(self) -> bool:
        return (
            "static_contact" in self.loss_terms
            and self.loss_weights["static_contact"] > 0.0
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
        pred_world_kp_3d: Tensor | None = None,
        pred_vel_3d: Tensor | None = None,
        pred_acc_3d: Tensor | None = None,
        pred_static_contact_logits: Tensor | None = None,
        frame_rate_hz: Tensor | None = None,
        frame_observed: Tensor | None = None,
    ) -> PLCSPreparedLossTerms:
        """Validate inputs and compute named scalar terms before ``forward``."""
        frame_mask = None
        if padding_mask is not None:
            frame_mask = ~(
                padding_mask.all(dim=1) if padding_mask.ndim == 3 else padding_mask
            )
        if frame_observed is not None:
            if frame_observed.dtype != torch.bool:
                raise ValueError(
                    f"frame_observed must be bool, got {frame_observed.dtype}."
                )
            obs_mask = frame_observed.to(device=pred_position.device)
            if obs_mask.shape != pred_position.shape[:-1]:
                raise ValueError(
                    "frame_observed must match position leading axes "
                    f"{tuple(pred_position.shape[:-1])}, got {tuple(obs_mask.shape)}."
                )
            frame_mask = obs_mask if frame_mask is None else (frame_mask & obs_mask)

        canonical_required = self._requires_canonical_pose()
        reprojection_required = self._requires_reprojection()
        world_required = self._requires_world_pose()
        pva_v_required = self._requires_pva_velocity()
        pva_a_required = self._requires_pva_acceleration()
        contact_required = self._requires_static_contact()

        if canonical_required and pred_canonical_pose is None:
            raise ValueError(
                "pred_canonical_pose is required when canonical-dependent "
                "loss weight is > 0."
            )
        if reprojection_required and (
            pred_canonical_pose is None and pred_world_kp_3d is None
        ):
            raise ValueError(
                "pred_canonical_pose or pred_world_kp_3d is required when "
                "reprojection loss weight is > 0."
            )
        if reprojection_required and reprojection_target is None:
            raise ValueError(
                "reprojection_target is required when reprojection_weight is > 0."
            )
        if world_required and (
            pred_world_kp_3d is None or target_human_kp_3d is None
        ):
            raise ValueError(
                "pred_world_kp_3d and target_human_kp_3d are required when "
                "world_kp_3d or kinematic consistency loss weight is > 0."
            )
        if pva_v_required and (pred_vel_3d is None or target_human_kp_3d is None):
            raise ValueError(
                "pred_vel_3d and target_human_kp_3d are required when "
                "pva_velocity_weight is > 0."
            )
        if pva_a_required and (pred_acc_3d is None or target_human_kp_3d is None):
            raise ValueError(
                "pred_acc_3d and target_human_kp_3d are required when "
                "pva_acceleration_weight is > 0."
            )
        if contact_required and (
            pred_static_contact_logits is None or target_human_kp_3d is None
        ):
            raise ValueError(
                "pred_static_contact_logits and target_human_kp_3d are required "
                "when static_contact_weight is > 0."
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
            pred_world_kp_3d=pred_world_kp_3d,
            target_world_kp_3d=target_human_kp_3d,
            pred_vel_3d=pred_vel_3d,
            pred_acc_3d=pred_acc_3d,
            pred_static_contact_logits=pred_static_contact_logits,
            frame_rate_hz=frame_rate_hz,
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

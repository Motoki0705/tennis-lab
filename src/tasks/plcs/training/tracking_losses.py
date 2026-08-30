"""Post-matching losses for multi-person tracking and canonical pose."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.base.training.tracking_lifecycle import (
    lifecycle_transition_mask,
    weighted_presence_bce_with_logits,
)
from src.tasks.plcs.training.tracking_matching import match_player_tracks
from src.utils.geometry.angles import wrapped_angle_diff
from src.utils.geometry.court_pose import (
    canonical_pose_to_world_pose,
    world_pose_to_canonical_pose,
)
from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
)
from src.utils.tensor_utils import masked_mean

Assignment = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class PLCSTrackingLossInputs:
    """Precomputed tensor terms entering the loss module hot path."""

    position: torch.Tensor
    rotation: torch.Tensor
    angle: torch.Tensor
    canonical_pose: torch.Tensor
    reprojection: torch.Tensor
    presence: torch.Tensor
    cardinality: torch.Tensor
    cardinality_nll: torch.Tensor
    presence_hard_negative: torch.Tensor
    presence_pairwise: torch.Tensor
    track_smoothness: torch.Tensor


_SUPPORTED_WEIGHT_FIELDS = frozenset(
    {
        "position_weight",
        "rotation_weight",
        "angle_weight",
        "canonical_pose_weight",
        "reprojection_weight",
        "presence_weight",
        "cardinality_weight",
        "cardinality_nll_weight",
        "presence_hard_negative_weight",
        "presence_pairwise_weight",
        "presence_inactive_weight",
        "presence_active_weight",
        "presence_transition_weight",
        "track_smoothness_weight",
        "match_position_weight",
        "match_rotation_weight",
        "match_presence_weight",
        "match_presence_inactive_weight",
    }
)
_MISSING_CONFIG_VALUE = object()


def _config_value(config: Any, name: str, default: object) -> object:
    """Read a loss setting from mappings and Hydra objects uniformly."""
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_items(config: Any) -> tuple[tuple[str, object], ...]:
    """Return top-level loss settings when the configuration exposes them."""
    if isinstance(config, Mapping):
        return tuple((str(key), value) for key, value in config.items())
    attributes = getattr(config, "__dict__", None)
    if isinstance(attributes, Mapping):
        return tuple((str(key), value) for key, value in attributes.items())
    return ()


def _require_positive_beta(config: Any, name: str, default: float) -> float:
    value = float(cast("Any", _config_value(config, name, default)))
    if value <= 0.0:
        raise ValueError(f"loss.{name} must be positive, got {value}.")
    return value


def _require_nonnegative_weight(
    config: Any,
    name: str,
    default: object = _MISSING_CONFIG_VALUE,
) -> float:
    """Read one finite non-negative weight without masking malformed values."""
    raw_value = _config_value(config, name, default)
    if raw_value is _MISSING_CONFIG_VALUE:
        raise ValueError(f"loss.{name} is required.")
    try:
        value = float(cast("Any", raw_value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"loss.{name} must be a finite non-negative number."
        ) from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"loss.{name} must be finite and non-negative, got {value}."
        )
    return value


def _validate_canonical_pose(name: str, value: torch.Tensor) -> None:
    if value.ndim != 5 or value.shape[-2:] != (17, 3):
        raise ValueError(
            f"{name} must have shape (B,T,Q,17,3), got {tuple(value.shape)}."
        )


def _poisson_binomial_log_count_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return exact log probabilities for every reachable Bernoulli count."""
    log_present = F.logsigmoid(logits)
    log_absent = F.logsigmoid(-logits)
    log_probabilities = logits.new_zeros((logits.shape[0], 1))
    for query_index in range(logits.shape[1]):
        absent = log_probabilities + log_absent[:, query_index, None]
        present = log_probabilities + log_present[:, query_index, None]
        reachable = [absent[:, 0]]
        reachable.extend(
            torch.logaddexp(absent[:, count], present[:, count - 1])
            for count in range(1, log_probabilities.shape[1])
        )
        reachable.append(present[:, -1])
        log_probabilities = torch.stack(reachable, dim=-1)
    return log_probabilities


def _poisson_binomial_count_nll(
    presence_logits: torch.Tensor,
    target_presence: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute exact count NLL on all-view-valid frames only."""
    if presence_logits.ndim != 3:
        raise ValueError(
            "prediction['presence_logits'] must have shape (B,T,Q) for "
            f"cardinality_nll, got {tuple(presence_logits.shape)}."
        )
    batch_size, frames, queries = presence_logits.shape
    if target_presence.ndim != 3 or target_presence.shape[:2] != (
        batch_size,
        frames,
    ):
        raise ValueError(
            "batch['target_presence'] must have shape (B,T,P) aligned with "
            "prediction['presence_logits'] for cardinality_nll, got "
            f"{tuple(target_presence.shape)} and {tuple(presence_logits.shape)}."
        )
    if target_presence.dtype != torch.bool:
        raise ValueError(
            "batch['target_presence'] must be boolean for cardinality_nll."
        )
    if (
        padding_mask.ndim != 3
        or padding_mask.shape[0] != batch_size
        or padding_mask.shape[2] != frames
    ):
        raise ValueError(
            "batch['padding_mask'] must have shape (B,V,T) aligned with "
            "prediction['presence_logits'] for cardinality_nll, got "
            f"{tuple(padding_mask.shape)} and {tuple(presence_logits.shape)}."
        )
    if padding_mask.dtype != torch.bool:
        raise ValueError("batch['padding_mask'] must be boolean for cardinality_nll.")

    valid_frames = (~padding_mask).any(dim=1)
    valid_logits = presence_logits[valid_frames].to(torch.float64)
    if valid_logits.shape[0] == 0:
        return valid_logits.sum()
    valid_target = target_presence[valid_frames]
    target_count = valid_target.sum(dim=-1, dtype=torch.long)
    if valid_target.shape[-1] > queries and bool((target_count > queries).any().item()):
        raise ValueError(
            "batch['target_presence'] contains a valid-frame player count greater "
            f"than the {queries} predicted queries."
        )
    log_count_probabilities = _poisson_binomial_log_count_probabilities(valid_logits)
    target_log_probability = log_count_probabilities.gather(
        dim=1,
        index=target_count.unsqueeze(-1),
    ).squeeze(-1)
    return -target_log_probability.mean()


def _presence_hard_negative_loss(
    presence_logits: torch.Tensor,
    presence_target: torch.Tensor,
    valid_frames: torch.Tensor,
    *,
    gamma: float,
    transition_radius: int,
) -> torch.Tensor:
    """Apply focal supervision only to fixed-assignment non-transition negatives."""
    transition = lifecycle_transition_mask(
        presence_target,
        valid_frames,
        radius=transition_radius,
    )
    hard_negative = valid_frames & (~presence_target) & (~transition)
    selected_logits = presence_logits[hard_negative].to(torch.float64)
    if selected_logits.shape[0] == 0:
        return selected_logits.sum()
    negative_bce = -F.logsigmoid(-selected_logits)
    focal_modulation = (
        torch.ones_like(selected_logits)
        if gamma == 0.0
        else torch.exp(gamma * F.logsigmoid(selected_logits))
    )
    return (focal_modulation * negative_bce).mean()


def _presence_pairwise_loss(
    presence_logits: torch.Tensor,
    presence_target: torch.Tensor,
    valid_frames: torch.Tensor,
    *,
    margin: float,
    transition_radius: int,
) -> torch.Tensor:
    """Rank every stable active query above every stable inactive query."""
    transition = lifecycle_transition_mask(
        presence_target,
        valid_frames,
        radius=transition_radius,
    )
    stable = valid_frames & (~transition)
    positive = stable & presence_target
    negative = stable & (~presence_target)
    pair_mask = positive.unsqueeze(-1) & negative.unsqueeze(-2)
    positive_logits = presence_logits.unsqueeze(-1).expand_as(pair_mask)[
        pair_mask
    ].to(torch.float64)
    if positive_logits.shape[0] == 0:
        return positive_logits.sum()
    negative_logits = presence_logits.unsqueeze(-2).expand_as(pair_mask)[
        pair_mask
    ].to(torch.float64)
    pair_losses = F.relu((negative_logits - positive_logits) + margin)
    pair_counts = pair_mask.sum(dim=(-2, -1))
    paired_frame_counts = pair_counts[pair_counts > 0]
    pair_weights = paired_frame_counts.to(torch.float64).reciprocal().repeat_interleave(
        paired_frame_counts
    )
    return (pair_losses * pair_weights).sum() / paired_frame_counts.shape[0]


def validate_tracking_projection_shapes(
    pred_uv: torch.Tensor,
    in_front: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> None:
    """Validate query projections and slot-aligned clean 2D supervision shapes."""
    batch_size, frames, queries = prediction["position"].shape[:3]
    views = batch["camera_R"].shape[1]
    target_slots = batch["target_presence"].shape[2]
    expected = {
        "pred_uv": (batch_size, views, frames, queries, 17, 2),
        "in_front": (batch_size, views, frames, queries, 17),
        "human_kp_target": (batch_size, views, frames, target_slots, 17, 2),
        "human_vis_target": (batch_size, views, frames, target_slots, 17),
        "padding_mask": (batch_size, views, frames),
    }
    actual = {
        "pred_uv": tuple(pred_uv.shape),
        "in_front": tuple(in_front.shape),
        "human_kp_target": tuple(batch["human_kp_target"].shape),
        "human_vis_target": tuple(batch["human_vis_target"].shape),
        "padding_mask": tuple(batch["padding_mask"].shape),
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {shape}"
        for name, shape in actual.items()
        if shape != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "Invalid tracking projection tensor shape(s): " + "; ".join(mismatches)
        )


class PLCSTrackingLoss(nn.Module):
    """Supervise fixed queries after clip-level assignment."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.position_weight = float(config.position_weight)
        self.position_smooth_l1_beta = _require_positive_beta(
            config, "position_smooth_l1_beta", 1.0
        )
        self.rotation_weight = float(config.rotation_weight)
        self.angle_weight = float(
            cast("Any", _config_value(config, "angle_weight", 0.0))
        )
        self.canonical_pose_weight = float(
            cast("Any", _config_value(config, "canonical_pose_weight", 0.0))
        )
        self.canonical_pose_smooth_l1_beta = _require_positive_beta(
            config, "canonical_pose_smooth_l1_beta", 1.0
        )
        self.reprojection_weight = float(
            cast("Any", _config_value(config, "reprojection_weight", 0.0))
        )
        self.reprojection_smooth_l1_beta = _require_positive_beta(
            config, "reprojection_smooth_l1_beta", 0.01
        )
        self.presence_weight = float(config.presence_weight)
        self.cardinality_weight = _require_nonnegative_weight(
            config,
            "cardinality_weight",
            0.0,
        )
        self.cardinality_nll_weight = _require_nonnegative_weight(
            config,
            "cardinality_nll_weight",
            0.0,
        )
        self.presence_hard_negative_weight = _require_nonnegative_weight(
            config,
            "presence_hard_negative_weight",
            0.0,
        )
        self.presence_hard_negative_gamma = _require_nonnegative_weight(
            config,
            "presence_hard_negative_gamma",
            2.0,
        )
        self.presence_pairwise_weight = _require_nonnegative_weight(
            config,
            "presence_pairwise_weight",
            0.0,
        )
        self.presence_pairwise_margin = _require_nonnegative_weight(
            config,
            "presence_pairwise_margin",
            0.5,
        )
        self.presence_inactive_weight = _require_nonnegative_weight(
            config,
            "presence_inactive_weight",
        )
        # Checkpoints predating the split have no matching-only value. Preserve
        # their coupled behavior exactly; a present but malformed value still
        # fails instead of falling back silently.
        self.match_presence_inactive_weight = _require_nonnegative_weight(
            config,
            "match_presence_inactive_weight",
            self.presence_inactive_weight,
        )
        self.presence_active_weight = float(config.presence_active_weight)
        self.presence_transition_weight = float(config.presence_transition_weight)
        self.transition_radius = int(config.transition_radius)
        self.track_smoothness_weight = float(config.track_smoothness_weight)
        self.match_position_weight = float(config.match_position_weight)
        self.match_rotation_weight = float(config.match_rotation_weight)
        self.match_presence_weight = float(config.match_presence_weight)
        self._include_angle = self.angle_weight > 0.0 or any(
            name == "angle_weight" for name, _ in _config_items(config)
        )
        self._include_canonical_pose = self.canonical_pose_weight > 0.0 or any(
            name == "canonical_pose_weight" for name, _ in _config_items(config)
        )
        self._include_reprojection = self.reprojection_weight > 0.0 or any(
            name == "reprojection_weight" for name, _ in _config_items(config)
        )
        self._include_cardinality = self.cardinality_weight > 0.0
        self._include_cardinality_nll = self.cardinality_nll_weight > 0.0
        self._include_presence_hard_negative = (
            self.presence_hard_negative_weight > 0.0
        )
        self._include_presence_pairwise = self.presence_pairwise_weight > 0.0
        self.reprojection_projector = DifferentiablePinholeProjection()

        for name, raw_value in _config_items(config):
            if not name.endswith("_weight") or name in _SUPPORTED_WEIGHT_FIELDS:
                continue
            try:
                value = float(cast("Any", raw_value))
            except (TypeError, ValueError):
                continue
            if value != 0.0:
                raise ValueError(
                    "PLCSTrackingLoss does not support nonzero "
                    f"loss.{name}={value}."
                )

    def prepare_inputs(
        self,
        prediction: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[PLCSTrackingLossInputs, list[Assignment]]:
        """Match tracks and prepare tensor loss terms outside ``forward``."""
        pred_position = prediction["position"]
        pred_presence = prediction["presence_logits"]
        zero = pred_position.sum() * 0.0
        cardinality_nll = (
            _poisson_binomial_count_nll(
                pred_presence,
                batch["target_presence"],
                batch["padding_mask"],
            )
            if self._include_cardinality_nll
            else zero
        )
        assignments = match_player_tracks(
            prediction,
            batch,
            position_cost_weight=self.match_position_weight,
            rotation_cost_weight=self.match_rotation_weight,
            presence_cost_weight=self.match_presence_weight,
            match_presence_inactive_weight=self.match_presence_inactive_weight,
            presence_active_weight=self.presence_active_weight,
            presence_transition_weight=self.presence_transition_weight,
            transition_radius=self.transition_radius,
        )
        pred_rotation = F.normalize(prediction["rotation"], dim=-1)
        pred_canonical_pose = prediction.get("canonical_pose")
        canonical_required = self.canonical_pose_weight > 0.0
        reprojection_required = self.reprojection_weight > 0.0
        if canonical_required or reprojection_required:
            if pred_canonical_pose is None:
                raise ValueError(
                    "prediction['canonical_pose'] is required when canonical-pose "
                    "or reprojection supervision is enabled."
                )
            _validate_canonical_pose("prediction['canonical_pose']", pred_canonical_pose)
        if canonical_required and "target_human_kp_3d" not in batch:
            raise ValueError(
                "batch['target_human_kp_3d'] is required when "
                "canonical_pose_weight is > 0."
            )
        if reprojection_required:
            required_reprojection_fields = (
                "human_kp_target",
                "human_vis_target",
                "camera_R",
                "camera_C",
                "camera_f",
                "camera_cx",
                "camera_cy",
                "camera_w",
                "camera_h",
            )
            missing = [
                name for name in required_reprojection_fields if name not in batch
            ]
            if missing:
                raise ValueError(
                    "Tracking reprojection supervision requires batch fields "
                    f"{missing}."
                )
        presence_target = torch.zeros_like(pred_presence)
        position_terms: list[torch.Tensor] = []
        rotation_terms: list[torch.Tensor] = []
        angle_terms: list[torch.Tensor] = []
        canonical_pose_terms: list[torch.Tensor] = []
        smoothness_terms: list[torch.Tensor] = []
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            for query_index, target_index in zip(
                query_indices.tolist(), target_indices.tolist(), strict=True
            ):
                active = (
                    batch["target_presence"][batch_index, :, target_index]
                    & (~batch["padding_mask"][batch_index]).any(dim=0)
                )
                presence_target[batch_index, :, query_index] = batch["target_presence"][
                    batch_index, :, target_index
                ].float()
                if active.any():
                    position_terms.append(
                        F.smooth_l1_loss(
                            pred_position[batch_index, active, query_index],
                            batch["target_position"][batch_index, active, target_index],
                            beta=self.position_smooth_l1_beta,
                        )
                    )
                    target_rotation = F.normalize(
                        batch["target_rotation"][batch_index, active, target_index],
                        dim=-1,
                    )
                    rotation_terms.append(
                        (
                            1.0
                            - (
                                pred_rotation[batch_index, active, query_index]
                                * target_rotation
                            ).sum(-1)
                        ).mean()
                    )
                    if self.angle_weight > 0.0:
                        pred_angle = torch.atan2(
                            prediction["rotation"][
                                batch_index, active, query_index, 1
                            ],
                            prediction["rotation"][
                                batch_index, active, query_index, 0
                            ],
                        )
                        target_angle = torch.atan2(
                            batch["target_rotation"][
                                batch_index, active, target_index, 1
                            ],
                            batch["target_rotation"][
                                batch_index, active, target_index, 0
                            ],
                        )
                        angle_error = wrapped_angle_diff(pred_angle, target_angle)
                        angle_terms.append(
                            F.smooth_l1_loss(
                                angle_error,
                                torch.zeros_like(angle_error),
                            )
                        )
                    if canonical_required:
                        assert pred_canonical_pose is not None
                        target_world_pose = batch["target_human_kp_3d"][
                            batch_index, active, target_index
                        ]
                        if target_world_pose.shape[-2:] != (17, 3):
                            raise ValueError(
                                "batch['target_human_kp_3d'] must contain exactly "
                                "17 XYZ joints."
                            )
                        target_canonical_pose = world_pose_to_canonical_pose(
                            target_world_pose,
                            batch["target_position"][
                                batch_index, active, target_index
                            ],
                            batch["target_rotation"][
                                batch_index, active, target_index
                            ],
                        )
                        canonical_pose_terms.append(
                            F.smooth_l1_loss(
                                pred_canonical_pose[
                                    batch_index, active, query_index
                                ],
                                target_canonical_pose,
                                beta=self.canonical_pose_smooth_l1_beta,
                            )
                        )
                if active.sum() >= 3 and self.track_smoothness_weight > 0.0:
                    consecutive = active[:-2] & active[1:-1] & active[2:]
                    if consecutive.any():
                        track = pred_position[batch_index, :, query_index]
                        acceleration = track[2:] - 2.0 * track[1:-1] + track[:-2]
                        smoothness_terms.append(
                            F.smooth_l1_loss(
                                acceleration[consecutive],
                                torch.zeros_like(acceleration[consecutive]),
                            )
                        )
        valid_frames = (~batch["padding_mask"]).any(dim=1).unsqueeze(-1).expand_as(
            pred_presence
        )
        presence_target_bool = presence_target.bool()
        presence = weighted_presence_bce_with_logits(
            pred_presence,
            presence_target_bool,
            valid_frames,
            inactive_weight=self.presence_inactive_weight,
            active_weight=self.presence_active_weight,
            transition_weight=self.presence_transition_weight,
            transition_radius=self.transition_radius,
        )
        presence_hard_negative = (
            _presence_hard_negative_loss(
                pred_presence,
                presence_target_bool,
                valid_frames,
                gamma=self.presence_hard_negative_gamma,
                transition_radius=self.transition_radius,
            )
            if self._include_presence_hard_negative
            else zero
        )
        presence_pairwise = (
            _presence_pairwise_loss(
                pred_presence,
                presence_target_bool,
                valid_frames,
                margin=self.presence_pairwise_margin,
                transition_radius=self.transition_radius,
            )
            if self._include_presence_pairwise
            else zero
        )
        if self._include_cardinality:
            cardinality_valid = (~batch["padding_mask"]).any(dim=1)
            valid_presence_logits = pred_presence[cardinality_valid]
            if valid_presence_logits.numel() == 0:
                cardinality = valid_presence_logits.sum()
            else:
                pred_count = valid_presence_logits.sigmoid().sum(dim=-1)
                target_count = batch["target_presence"][cardinality_valid].to(
                    pred_count.dtype
                ).sum(dim=-1)
                cardinality = F.smooth_l1_loss(
                    pred_count,
                    target_count,
                    beta=1.0,
                )
        else:
            cardinality = zero
        position = torch.stack(position_terms).mean() if position_terms else zero
        rotation = torch.stack(rotation_terms).mean() if rotation_terms else zero
        angle = torch.stack(angle_terms).mean() if angle_terms else zero
        canonical_zero = (
            pred_canonical_pose.sum() * 0.0
            if pred_canonical_pose is not None
            else zero
        )
        canonical_pose = (
            torch.stack(canonical_pose_terms).mean()
            if canonical_pose_terms
            else canonical_zero
        )
        reprojection = self._reprojection_loss(
            prediction,
            batch,
            assignments,
            zero=canonical_zero,
        )
        smoothness = torch.stack(smoothness_terms).mean() if smoothness_terms else zero
        return PLCSTrackingLossInputs(
            position=position,
            rotation=rotation,
            angle=angle,
            canonical_pose=canonical_pose,
            reprojection=reprojection,
            presence=presence,
            cardinality=cardinality,
            cardinality_nll=cardinality_nll,
            presence_hard_negative=presence_hard_negative,
            presence_pairwise=presence_pairwise,
            track_smoothness=smoothness,
        ), assignments

    def _reprojection_loss(
        self,
        prediction: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        assignments: list[Assignment],
        *,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        """Project matched query poses and compare them with clean slot-aligned UV."""
        if self.reprojection_weight <= 0.0:
            return zero
        canonical_pose = prediction.get("canonical_pose")
        if canonical_pose is None:
            raise ValueError(
                "prediction['canonical_pose'] is required for reprojection."
            )
        world_pose = canonical_pose_to_world_pose(
            canonical_pose,
            prediction["position"],
            prediction["rotation"],
        )
        pred_uv, in_front = self.reprojection_projector(
            world_points=world_pose,
            camera_R=batch["camera_R"],
            camera_C=batch["camera_C"],
            camera_f=batch["camera_f"],
            camera_cx=batch["camera_cx"],
            camera_cy=batch["camera_cy"],
            camera_w=batch["camera_w"],
            camera_h=batch["camera_h"],
        )
        validate_tracking_projection_shapes(pred_uv, in_front, prediction, batch)
        terms: list[torch.Tensor] = []
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            for query_index, target_index in zip(
                query_indices.tolist(), target_indices.tolist(), strict=True
            ):
                target_uv = batch["human_kp_target"][
                    batch_index, :, :, target_index
                ]
                target_vis = batch["human_vis_target"][
                    batch_index, :, :, target_index
                ]
                if target_uv.shape != pred_uv[batch_index, :, :, query_index].shape:
                    raise ValueError(
                        "Matched tracking reprojection tensors must share "
                        "(V,T,17,2), got "
                        f"{tuple(pred_uv[batch_index, :, :, query_index].shape)} "
                        f"and {tuple(target_uv.shape)}."
                    )
                active = batch["target_presence"][
                    batch_index, :, target_index
                ]
                valid = (
                    (target_vis > 0)
                    & (~batch["padding_mask"][batch_index]).unsqueeze(-1)
                    & active.unsqueeze(0).unsqueeze(-1)
                )
                per_coordinate = F.smooth_l1_loss(
                    pred_uv[batch_index, :, :, query_index],
                    target_uv,
                    reduction="none",
                    beta=self.reprojection_smooth_l1_beta,
                )
                terms.append(
                    masked_mean(
                        per_coordinate,
                        valid.unsqueeze(-1),
                        binarize=True,
                        denom_min=1.0,
                    )
                )
        return torch.stack(terms).mean() if terms else zero

    def forward(self, inputs: PLCSTrackingLossInputs) -> dict[str, torch.Tensor]:
        """Combine boundary-prepared tensor terms with configured weights."""
        total = (
            self.position_weight * inputs.position
            + self.rotation_weight * inputs.rotation
            + self.presence_weight * inputs.presence
            + self.track_smoothness_weight * inputs.track_smoothness
        )
        losses = {
            "total": total,
            "position": inputs.position,
            "rotation": inputs.rotation,
            "presence": inputs.presence,
            "track_smoothness": inputs.track_smoothness,
        }
        if self._include_angle:
            losses["angle"] = inputs.angle
            total = total + self.angle_weight * inputs.angle
        if self._include_canonical_pose:
            losses["canonical_pose"] = inputs.canonical_pose
            total = total + self.canonical_pose_weight * inputs.canonical_pose
        if self._include_reprojection:
            losses["reprojection"] = inputs.reprojection
            total = total + self.reprojection_weight * inputs.reprojection
        if self._include_cardinality:
            losses["cardinality"] = inputs.cardinality
            total = total + self.cardinality_weight * inputs.cardinality
        if self._include_cardinality_nll:
            losses["cardinality_nll"] = inputs.cardinality_nll
            total = total + self.cardinality_nll_weight * inputs.cardinality_nll
        if self._include_presence_hard_negative:
            losses["presence_hard_negative"] = inputs.presence_hard_negative
            total = (
                total
                + self.presence_hard_negative_weight
                * inputs.presence_hard_negative
            )
        if self._include_presence_pairwise:
            losses["presence_pairwise"] = inputs.presence_pairwise
            total = total + self.presence_pairwise_weight * inputs.presence_pairwise
        losses["total"] = total
        return losses

"""Matched multi-ball position, presence, and optional physics losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.base.training.tracking_lifecycle import (
    weighted_presence_bce_with_logits,
)
from src.tasks.blcs.configuration import (
    resolve_position_huber_beta,
    resolve_tracking_gravity_target,
)
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
)
from src.tasks.blcs.training.tracking_matching import match_ball_tracks
from src.tasks.blcs.training.tracking_position import (
    position_axis_weight_tensor,
    weighted_position_axis_mean,
)
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)

Assignment = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class BLCSTrackingLossInputs:
    """Boundary-prepared scalar tensors entering the loss hot path."""

    position: torch.Tensor
    position_per_axis: torch.Tensor
    presence: torch.Tensor
    smoothness: torch.Tensor
    gravity: torch.Tensor


class BLCSTrackingLoss(nn.Module):
    """Apply supervision after clip-level Hungarian matching."""

    position_axis_weights: torch.Tensor

    def __init__(
        self,
        config: Any,
        *,
        normalization: CourtCoordinateNormalization | str = "v1",
        gravity: float = 9.81,
        frame_dt: float = 1.0 / 30.0,
    ) -> None:
        super().__init__()
        contract = (
            normalization
            if isinstance(normalization, CourtCoordinateNormalization)
            else resolve_court_coordinate_normalization(normalization)
        )
        self.position_weight = float(config.position_weight)
        self.presence_weight = float(config.presence_weight)
        self.presence_inactive_weight = float(config.presence_inactive_weight)
        self.presence_active_weight = float(config.presence_active_weight)
        self.presence_transition_weight = float(config.presence_transition_weight)
        self.transition_radius = int(config.transition_radius)
        self.smoothness_weight = float(config.smoothness_weight)
        self.gravity_weight = float(config.gravity_weight)
        self.gravity_target = resolve_tracking_gravity_target(
            contract,
            legacy_v1_target=float(config.gravity_target),
            gravity=gravity,
            frame_dt=frame_dt,
        )
        self.match_position_weight = float(config.match_position_weight)
        self.match_presence_weight = float(config.match_presence_weight)
        configured_axis_weights = (
            config.position_axis_weights
            if contract.version == "v1"
            else config.position_axis_weights_v2
        )
        self.position_beta = resolve_position_huber_beta(
            contract,
            legacy_v1_beta=float(getattr(config, "position_huber_beta_v1", 1.0)),
            v2_transition_m=float(
                getattr(config, "position_huber_transition_m_v2", 1.0)
            ),
        )
        self.register_buffer(
            "position_axis_weights",
            position_axis_weight_tensor(configured_axis_weights),
            persistent=False,
        )

    @staticmethod
    def _zero(prediction: BLCSTrackQueryPrediction) -> torch.Tensor:
        return prediction.position.sum() * 0.0

    def prepare_inputs(
        self,
        prediction: BLCSTrackQueryPrediction,
        batch: BLCSTrackQueryTrainingBatch,
    ) -> tuple[BLCSTrackingLossInputs, list[Assignment]]:
        """Match clips and prepare all task tensors outside ``forward``."""
        assignments = match_ball_tracks(
            prediction.position,
            prediction.presence_logits,
            batch.target_position,
            batch.target_presence,
            batch.target_slot_mask,
            batch.frame_valid,
            position_cost_weight=self.match_position_weight,
            presence_cost_weight=self.match_presence_weight,
            presence_inactive_weight=self.presence_inactive_weight,
            presence_active_weight=self.presence_active_weight,
            presence_transition_weight=self.presence_transition_weight,
            transition_radius=self.transition_radius,
            position_axis_weights=self.position_axis_weights,
            position_beta=self.position_beta,
        )
        pred_position = prediction.position
        pred_presence = prediction.presence_logits
        presence_target = torch.zeros_like(pred_presence)
        position_terms: list[torch.Tensor] = []
        position_axis_terms: list[torch.Tensor] = []
        smoothness_terms: list[torch.Tensor] = []
        gravity_terms: list[torch.Tensor] = []
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            for query_index, target_index in zip(
                query_indices.tolist(), target_indices.tolist(), strict=True
            ):
                active = (
                    batch.target_presence[batch_index, :, target_index]
                    & batch.frame_valid[batch_index]
                )
                presence_target[batch_index, :, query_index] = batch.target_presence[
                    batch_index, :, target_index
                ].float()
                if active.any():
                    position_error_xyz = F.smooth_l1_loss(
                        pred_position[batch_index, active, query_index],
                        batch.target_position[batch_index, active, target_index],
                        reduction="none",
                        beta=self.position_beta,
                    )
                    position_per_axis = position_error_xyz.mean(0)
                    position_axis_terms.append(position_per_axis)
                    position_terms.append(
                        weighted_position_axis_mean(
                            position_per_axis,
                            self.position_axis_weights,
                        )
                    )
                if active.sum() >= 3 and (
                    self.smoothness_weight > 0.0 or self.gravity_weight > 0.0
                ):
                    track = pred_position[batch_index, :, query_index]
                    consecutive = active[:-2] & active[1:-1] & active[2:]
                    second_difference = track[2:] - 2.0 * track[1:-1] + track[:-2]
                    if consecutive.any():
                        smoothness_terms.append(
                            F.smooth_l1_loss(
                                second_difference[consecutive],
                                torch.zeros_like(second_difference[consecutive]),
                            )
                        )
                        gravity_terms.append(
                            F.smooth_l1_loss(
                                second_difference[consecutive, 2],
                                torch.full_like(
                                    second_difference[consecutive, 2],
                                    self.gravity_target,
                                ),
                            )
                        )

        valid_frames = batch.frame_valid.unsqueeze(-1).expand_as(pred_presence)
        presence = weighted_presence_bce_with_logits(
            pred_presence,
            presence_target.bool(),
            valid_frames,
            inactive_weight=self.presence_inactive_weight,
            active_weight=self.presence_active_weight,
            transition_weight=self.presence_transition_weight,
            transition_radius=self.transition_radius,
        )
        zero = self._zero(prediction)
        position = torch.stack(position_terms).mean() if position_terms else zero
        position_per_axis = (
            torch.stack(position_axis_terms).mean(0)
            if position_axis_terms
            else torch.stack([zero, zero, zero])
        )
        smoothness = torch.stack(smoothness_terms).mean() if smoothness_terms else zero
        gravity = torch.stack(gravity_terms).mean() if gravity_terms else zero
        return BLCSTrackingLossInputs(
            position=position,
            position_per_axis=position_per_axis,
            presence=presence,
            smoothness=smoothness,
            gravity=gravity,
        ), assignments

    def forward(self, inputs: BLCSTrackingLossInputs) -> dict[str, torch.Tensor]:
        """Combine boundary-prepared tensor terms with configured weights."""
        total = (
            self.position_weight * inputs.position
            + self.presence_weight * inputs.presence
            + self.smoothness_weight * inputs.smoothness
            + self.gravity_weight * inputs.gravity
        )
        return {
            "total": total,
            "position": inputs.position,
            "position_x": inputs.position_per_axis[0],
            "position_y": inputs.position_per_axis[1],
            "position_z": inputs.position_per_axis[2],
            "presence": inputs.presence,
            "smoothness": inputs.smoothness,
            "gravity": inputs.gravity,
        }


__all__ = ["Assignment", "BLCSTrackingLoss", "BLCSTrackingLossInputs"]

"""Matched multi-ball position, presence, and optional physics losses."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.base.training.tracking_lifecycle import (
    weighted_presence_bce_with_logits,
)
from src.tasks.blcs.training.tracking_matching import match_ball_tracks
from src.tasks.blcs.training.tracking_position import (
    position_axis_weight_tensor,
    weighted_position_axis_mean,
)

Assignment = tuple[torch.Tensor, torch.Tensor]


class BLCSTrackingLoss(nn.Module):
    """Apply supervision after clip-level Hungarian matching."""

    position_axis_weights: torch.Tensor

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.position_weight = float(config.position_weight)
        self.presence_weight = float(config.presence_weight)
        self.presence_inactive_weight = float(config.presence_inactive_weight)
        self.presence_active_weight = float(config.presence_active_weight)
        self.presence_transition_weight = float(config.presence_transition_weight)
        self.transition_radius = int(config.transition_radius)
        self.smoothness_weight = float(config.smoothness_weight)
        self.gravity_weight = float(config.gravity_weight)
        self.gravity_target = float(config.gravity_target)
        self.match_position_weight = float(config.match_position_weight)
        self.match_presence_weight = float(config.match_presence_weight)
        configured_axis_weights = config.position_axis_weights
        self.register_buffer(
            "position_axis_weights",
            position_axis_weight_tensor(configured_axis_weights),
            persistent=False,
        )

    @staticmethod
    def _zero(prediction: dict[str, torch.Tensor]) -> torch.Tensor:
        return prediction["position"].sum() * 0.0

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], list[Assignment]]:
        assignments = match_ball_tracks(
            prediction,
            batch["target_position"],
            batch["target_presence"],
            batch["target_slot_mask"],
            batch["frame_mask"],
            position_cost_weight=self.match_position_weight,
            presence_cost_weight=self.match_presence_weight,
            presence_inactive_weight=self.presence_inactive_weight,
            presence_active_weight=self.presence_active_weight,
            presence_transition_weight=self.presence_transition_weight,
            transition_radius=self.transition_radius,
            position_axis_weights=self.position_axis_weights,
        )
        pred_position = prediction["position"]
        pred_presence = prediction["presence_logits"]
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
                    batch["target_presence"][batch_index, :, target_index]
                    & batch["frame_mask"][batch_index]
                )
                presence_target[batch_index, :, query_index] = batch["target_presence"][
                    batch_index, :, target_index
                ].float()
                if active.any():
                    position_error_xyz = F.smooth_l1_loss(
                        pred_position[batch_index, active, query_index],
                        batch["target_position"][batch_index, active, target_index],
                        reduction="none",
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

        valid_frames = batch["frame_mask"].unsqueeze(-1).expand_as(pred_presence)
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
        total = (
            self.position_weight * position
            + self.presence_weight * presence
            + self.smoothness_weight * smoothness
            + self.gravity_weight * gravity
        )
        return {
            "total": total,
            "position": position,
            "position_x": position_per_axis[0],
            "position_y": position_per_axis[1],
            "position_z": position_per_axis[2],
            "presence": presence,
            "smoothness": smoothness,
            "gravity": gravity,
        }, assignments

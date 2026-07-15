"""Matched multi-ball position, presence, and optional physics losses."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.ball_tracking.matching import match_ball_tracks

Assignment = tuple[torch.Tensor, torch.Tensor]


class BallTrackingLoss(nn.Module):
    """Apply supervision after clip-level Hungarian matching."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.position_weight = float(config.position_weight)
        self.presence_weight = float(config.presence_weight)
        self.smoothness_weight = float(config.smoothness_weight)
        self.gravity_weight = float(config.gravity_weight)
        self.gravity_target = float(config.gravity_target)
        self.match_position_weight = float(config.match_position_weight)
        self.match_presence_weight = float(config.match_presence_weight)

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
            batch["position_3d"],
            batch["ball_present"],
            batch["target_ball_mask"],
            batch["frame_mask"],
            position_cost_weight=self.match_position_weight,
            presence_cost_weight=self.match_presence_weight,
        )
        pred_position = prediction["position"]
        pred_presence = prediction["presence_logits"]
        presence_target = torch.zeros_like(pred_presence)
        position_terms: list[torch.Tensor] = []
        smoothness_terms: list[torch.Tensor] = []
        gravity_terms: list[torch.Tensor] = []
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            for query_index, target_index in zip(
                query_indices.tolist(), target_indices.tolist(), strict=True
            ):
                active = (
                    batch["ball_present"][batch_index, :, target_index]
                    & batch["frame_mask"][batch_index]
                )
                presence_target[batch_index, :, query_index] = batch["ball_present"][
                    batch_index, :, target_index
                ].float()
                if active.any():
                    position_terms.append(
                        F.smooth_l1_loss(
                            pred_position[batch_index, active, query_index],
                            batch["position_3d"][batch_index, active, target_index],
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
        presence_raw = F.binary_cross_entropy_with_logits(
            pred_presence, presence_target, reduction="none"
        )
        presence = (presence_raw * valid_frames).sum() / valid_frames.sum().clamp_min(1)
        zero = self._zero(prediction)
        position = torch.stack(position_terms).mean() if position_terms else zero
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
            "presence": presence,
            "smoothness": smoothness,
            "gravity": gravity,
        }, assignments

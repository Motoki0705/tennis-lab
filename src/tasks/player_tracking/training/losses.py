"""Post-matching losses for multi-person position, rotation, and presence."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.player_tracking.matching import match_player_tracks

Assignment = tuple[torch.Tensor, torch.Tensor]


class PlayerTrackingLoss(nn.Module):
    """Supervise fixed queries after clip-level assignment."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.position_weight = float(config.position_weight)
        self.rotation_weight = float(config.rotation_weight)
        self.presence_weight = float(config.presence_weight)
        self.track_smoothness_weight = float(config.track_smoothness_weight)
        self.match_position_weight = float(config.match_position_weight)
        self.match_rotation_weight = float(config.match_rotation_weight)
        self.match_presence_weight = float(config.match_presence_weight)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], list[Assignment]]:
        assignments = match_player_tracks(
            prediction,
            batch,
            position_cost_weight=self.match_position_weight,
            rotation_cost_weight=self.match_rotation_weight,
            presence_cost_weight=self.match_presence_weight,
        )
        pred_position = prediction["position"]
        pred_rotation = F.normalize(prediction["rotation"], dim=-1)
        pred_presence = prediction["presence_logits"]
        presence_target = torch.zeros_like(pred_presence)
        position_terms: list[torch.Tensor] = []
        rotation_terms: list[torch.Tensor] = []
        smoothness_terms: list[torch.Tensor] = []
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            for query_index, target_index in zip(
                query_indices.tolist(), target_indices.tolist(), strict=True
            ):
                active = (
                    batch["person_present"][batch_index, :, target_index]
                    & batch["frame_mask"][batch_index]
                )
                presence_target[batch_index, :, query_index] = batch["person_present"][
                    batch_index, :, target_index
                ].float()
                if active.any():
                    position_terms.append(
                        F.smooth_l1_loss(
                            pred_position[batch_index, active, query_index],
                            batch["position"][batch_index, active, target_index],
                        )
                    )
                    target_rotation = F.normalize(
                        batch["rotation"][batch_index, active, target_index], dim=-1
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
        valid_frames = batch["frame_mask"].unsqueeze(-1).expand_as(pred_presence)
        presence_raw = F.binary_cross_entropy_with_logits(
            pred_presence, presence_target, reduction="none"
        )
        presence = (presence_raw * valid_frames).sum() / valid_frames.sum().clamp_min(1)
        zero = pred_position.sum() * 0.0
        position = torch.stack(position_terms).mean() if position_terms else zero
        rotation = torch.stack(rotation_terms).mean() if rotation_terms else zero
        smoothness = torch.stack(smoothness_terms).mean() if smoothness_terms else zero
        total = (
            self.position_weight * position
            + self.rotation_weight * rotation
            + self.presence_weight * presence
            + self.track_smoothness_weight * smoothness
        )
        return {
            "total": total,
            "position": position,
            "rotation": rotation,
            "presence": presence,
            "track_smoothness": smoothness,
        }, assignments

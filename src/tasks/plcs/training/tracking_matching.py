"""Hungarian matching over complete predicted and GT player clips."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.tasks.base.training.tracking_lifecycle import (
    weighted_presence_bce_with_logits,
)


def match_player_tracks(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    position_cost_weight: float,
    rotation_cost_weight: float,
    presence_cost_weight: float,
    presence_inactive_weight: float,
    presence_active_weight: float,
    presence_transition_weight: float,
    transition_radius: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match query slots to valid GT persons with padded frames excluded."""
    pred_position = prediction["position"]
    pred_rotation = prediction["rotation"]
    pred_presence = prediction["presence_logits"]
    batch_size, _, num_queries, _ = pred_position.shape
    assignments: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch_index in range(batch_size):
        target_indices = torch.nonzero(
            batch["target_slot_mask"][batch_index], as_tuple=False
        ).flatten()
        if target_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=pred_position.device)
            assignments.append((empty, empty))
            continue
        costs = pred_position.new_zeros((num_queries, target_indices.numel()))
        valid_frames = batch["frame_mask"][batch_index]
        for target_column, target_index in enumerate(target_indices.tolist()):
            target_active = (
                batch["target_presence"][batch_index, :, target_index] & valid_frames
            )
            target_presence = batch["target_presence"][
                batch_index, :, target_index
            ].float()
            presence = torch.stack(
                [
                    weighted_presence_bce_with_logits(
                        pred_presence[batch_index, :, query_index],
                        target_presence.bool(),
                        valid_frames,
                        inactive_weight=presence_inactive_weight,
                        active_weight=presence_active_weight,
                        transition_weight=presence_transition_weight,
                        transition_radius=transition_radius,
                    )
                    for query_index in range(num_queries)
                ]
            )
            if target_active.any():
                position = F.smooth_l1_loss(
                    pred_position[batch_index],
                    batch["target_position"][
                        batch_index, :, target_index, None
                    ].expand_as(pred_position[batch_index]),
                    reduction="none",
                ).mean(-1)
                position = (position * target_active[:, None]).sum(
                    0
                ) / target_active.sum()
                target_rotation = F.normalize(
                    batch["target_rotation"][batch_index, :, target_index], dim=-1
                )
                rotation = 1.0 - (
                    F.normalize(pred_rotation[batch_index], dim=-1)
                    * target_rotation[:, None]
                ).sum(-1)
                rotation = (rotation * target_active[:, None]).sum(
                    0
                ) / target_active.sum()
            else:
                position = torch.zeros_like(presence)
                rotation = torch.zeros_like(presence)
            costs[:, target_column] = (
                position_cost_weight * position
                + rotation_cost_weight * rotation
                + presence_cost_weight * presence
            )
        query_np, column_np = linear_sum_assignment(
            costs.detach().float().cpu().numpy()
        )
        assignments.append(
            (
                torch.as_tensor(
                    query_np, device=pred_position.device, dtype=torch.long
                ),
                target_indices[
                    torch.as_tensor(
                        column_np, device=pred_position.device, dtype=torch.long
                    )
                ],
            )
        )
    return assignments

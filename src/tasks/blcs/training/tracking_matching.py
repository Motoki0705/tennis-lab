"""Hungarian assignment over whole predicted and target ball clips."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.tasks.base.training.tracking_lifecycle import (
    weighted_presence_bce_with_logits,
)
from src.tasks.blcs.training.tracking_position import (
    position_axis_weight_tensor,
    weighted_position_axis_mean,
)


def match_ball_tracks(
    prediction: dict[str, torch.Tensor],
    target_position: torch.Tensor,
    target_presence: torch.Tensor,
    target_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    position_cost_weight: float,
    presence_cost_weight: float,
    presence_inactive_weight: float,
    presence_active_weight: float,
    presence_transition_weight: float,
    transition_radius: int,
    position_axis_weights: tuple[float, float, float] | torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match `Q` predictions to valid `P` targets using clip-level costs."""
    pred_position = prediction["position"]
    pred_presence = prediction["presence_logits"]
    axis_weights = position_axis_weight_tensor(position_axis_weights).to(
        pred_position.device
    )
    batch_size, _, num_queries, _ = pred_position.shape
    assignments: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in range(batch_size):
        target_indices = torch.nonzero(target_mask[batch], as_tuple=False).flatten()
        if target_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=pred_position.device)
            assignments.append((empty, empty))
            continue
        costs = torch.zeros(
            num_queries,
            target_indices.numel(),
            device=pred_position.device,
            dtype=pred_position.dtype,
        )
        valid_frames = frame_mask[batch]
        for target_column, target_index in enumerate(target_indices.tolist()):
            target_active = target_presence[batch, :, target_index] & valid_frames
            presence_target = target_presence[batch, :, target_index].float()
            presence_cost = torch.stack(
                [
                    weighted_presence_bce_with_logits(
                        pred_presence[batch, :, query_index],
                        presence_target.bool(),
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
                position_error_xyz = F.smooth_l1_loss(
                    pred_position[batch],
                    target_position[batch, :, target_index, None, :].expand_as(
                        pred_position[batch]
                    ),
                    reduction="none",
                )
                position_error = weighted_position_axis_mean(
                    position_error_xyz, axis_weights
                )
                position_cost = (position_error * target_active[:, None]).sum(
                    0
                ) / target_active.sum()
            else:
                position_cost = torch.zeros_like(presence_cost)
            costs[:, target_column] = (
                position_cost_weight * position_cost
                + presence_cost_weight * presence_cost
            )
        query_np, target_column_np = linear_sum_assignment(
            costs.detach().float().cpu().numpy()
        )
        assignments.append(
            (
                torch.as_tensor(
                    query_np, device=pred_position.device, dtype=torch.long
                ),
                target_indices[
                    torch.as_tensor(
                        target_column_np, device=pred_position.device, dtype=torch.long
                    )
                ],
            )
        )
    return assignments

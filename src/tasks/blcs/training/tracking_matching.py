"""Hungarian assignment over whole predicted and target ball clips."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def match_ball_tracks(
    prediction: dict[str, torch.Tensor],
    target_position: torch.Tensor,
    target_presence: torch.Tensor,
    target_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    position_cost_weight: float = 1.0,
    presence_cost_weight: float = 1.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match `Q` predictions to valid `P` targets using clip-level costs."""
    pred_position = prediction["position"]
    pred_presence = prediction["presence_logits"]
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
            presence_loss = F.binary_cross_entropy_with_logits(
                pred_presence[batch],
                presence_target[:, None].expand(-1, num_queries),
                reduction="none",
            )
            presence_denominator = valid_frames.sum().clamp_min(1)
            presence_cost = (presence_loss * valid_frames[:, None]).sum(
                0
            ) / presence_denominator
            if target_active.any():
                position_error = F.smooth_l1_loss(
                    pred_position[batch],
                    target_position[batch, :, target_index, None, :].expand_as(
                        pred_position[batch]
                    ),
                    reduction="none",
                ).mean(-1)
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

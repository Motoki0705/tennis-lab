"""Hungarian matching over complete predicted and GT player clips."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def match_player_tracks(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    position_cost_weight: float = 1.0,
    rotation_cost_weight: float = 1.0,
    presence_cost_weight: float = 1.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match query slots to valid GT persons with padded frames excluded."""
    pred_position = prediction["position"]
    pred_rotation = prediction["rotation"]
    pred_presence = prediction["presence_logits"]
    batch_size, _, num_queries, _ = pred_position.shape
    assignments: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch_index in range(batch_size):
        target_indices = torch.nonzero(
            batch["target_person_mask"][batch_index], as_tuple=False
        ).flatten()
        if target_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=pred_position.device)
            assignments.append((empty, empty))
            continue
        costs = pred_position.new_zeros((num_queries, target_indices.numel()))
        valid_frames = batch["frame_mask"][batch_index]
        for target_column, target_index in enumerate(target_indices.tolist()):
            target_active = (
                batch["person_present"][batch_index, :, target_index] & valid_frames
            )
            target_presence = batch["person_present"][
                batch_index, :, target_index
            ].float()
            presence = F.binary_cross_entropy_with_logits(
                pred_presence[batch_index],
                target_presence[:, None].expand(-1, num_queries),
                reduction="none",
            )
            presence = (presence * valid_frames[:, None]).sum(
                0
            ) / valid_frames.sum().clamp_min(1)
            if target_active.any():
                position = F.smooth_l1_loss(
                    pred_position[batch_index],
                    batch["position"][batch_index, :, target_index, None].expand_as(
                        pred_position[batch_index]
                    ),
                    reduction="none",
                ).mean(-1)
                position = (position * target_active[:, None]).sum(
                    0
                ) / target_active.sum()
                target_rotation = F.normalize(
                    batch["rotation"][batch_index, :, target_index], dim=-1
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

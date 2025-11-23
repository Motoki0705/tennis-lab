"""Helpers for matching DETR queries to player targets via Hungarian assignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def match_queries_to_targets(
    pose_pred: Tensor,
    pose_gt: Tensor,
    exist_gt: Tensor,
    exist_logit: Tensor,
    lambda_pose_match: float,
    lambda_exist_match: float,
) -> list[tuple[Tensor, Tensor]]:
    """Run Hungarian matching between predicted queries and GT players.

    This utility mirrors the behavior used across all TennisDETR lightning
    modules and is shared between v1/v2/v2.5/v3 variants.
    """
    B, Q, _, _, _ = pose_pred.shape
    _, _, M, J, _ = pose_gt.shape
    exist_any = exist_gt.any(dim=1)  # [B, M]
    matches: list[tuple[Tensor, Tensor]] = []

    for b in range(B):
        valid_mask = exist_any[b]
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
        if valid_indices.numel() == 0 or Q == 0:
            empty = (
                pose_pred.new_zeros((0,), dtype=torch.long),
                pose_pred.new_zeros((0,), dtype=torch.long),
            )
            matches.append(empty)
            continue

        pose_gt_b = pose_gt[b][:, valid_indices, :, :].permute(1, 0, 2, 3)
        exist_mask = exist_gt[b][:, valid_indices].permute(1, 0)  # [M_valid, T]
        pose_pred_b = pose_pred[b]  # [Q, T, J, 3]

        diff = torch.abs(pose_pred_b.unsqueeze(1) - pose_gt_b.unsqueeze(0))
        mask = exist_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(diff.dtype)
        diff = diff * mask
        counts = exist_mask.sum(dim=1).clamp_min(1).to(diff.dtype) * float(J * 3)
        pose_cost = diff.sum(dim=(2, 3, 4)) / counts.unsqueeze(0)

        exist_cost_q = F.binary_cross_entropy_with_logits(
            exist_logit[b, :, 0],
            torch.ones_like(exist_logit[b, :, 0]),
            reduction="none",
        )
        exist_cost = exist_cost_q[:, None].expand(-1, pose_cost.shape[1])

        total_cost = lambda_pose_match * pose_cost + lambda_exist_match * exist_cost
        cost_np = total_cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        matched_queries = torch.as_tensor(
            row_ind, dtype=torch.long, device=pose_pred.device
        )
        matched_targets = valid_indices[
            torch.as_tensor(col_ind, dtype=torch.long, device=pose_pred.device)
        ]
        matches.append((matched_queries, matched_targets))

    return matches

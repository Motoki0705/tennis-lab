"""Matched localization and track-query diagnostics for players."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.tasks.player_tracking.training.losses import Assignment


def player_tracking_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
    *,
    presence_threshold: float = 0.5,
    duplicate_distance: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute matched position/angular error and tracking failure counts."""
    pred_active = prediction["presence_logits"].sigmoid() >= presence_threshold
    target_presence = torch.zeros_like(pred_active)
    position_errors: list[torch.Tensor] = []
    angular_errors: list[torch.Tensor] = []
    id_switches = prediction["position"].new_zeros(())
    missed = prediction["position"].new_zeros(())
    matched_queries: list[set[int]] = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        matched_queries.append(set(query_indices.tolist()))
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch["person_present"][batch_index, :, target_index]
                & batch["frame_mask"][batch_index]
            )
            target_presence[batch_index, :, query_index] = batch["person_present"][
                batch_index, :, target_index
            ]
            if not active.any():
                continue
            position_errors.append(
                torch.linalg.vector_norm(
                    prediction["position"][batch_index, active, query_index]
                    - batch["position"][batch_index, active, target_index],
                    dim=-1,
                ).mean()
            )
            cosine = (
                (
                    F.normalize(
                        prediction["rotation"][batch_index, active, query_index], dim=-1
                    )
                    * F.normalize(
                        batch["rotation"][batch_index, active, target_index], dim=-1
                    )
                )
                .sum(-1)
                .clamp(-1.0, 1.0)
            )
            angular_errors.append(torch.acos(cosine).mean() * (180.0 / math.pi))
            missed = missed + (~pred_active[batch_index, active, query_index]).sum()
            distances = torch.linalg.vector_norm(
                prediction["position"][batch_index]
                - batch["position"][batch_index, :, target_index, None],
                dim=-1,
            ).masked_fill(~pred_active[batch_index], float("inf"))
            nearest = distances.argmin(-1)[active]
            if nearest.numel() > 1:
                id_switches = id_switches + (nearest[1:] != nearest[:-1]).sum()

    valid = batch["frame_mask"].unsqueeze(-1)
    true_positive = (pred_active & target_presence & valid).sum()
    false_positive = (pred_active & ~target_presence & valid).sum()
    false_negative = (~pred_active & target_presence & valid).sum()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    duplicate = prediction["position"].new_zeros(())
    inactive_false_positive = prediction["position"].new_zeros(())
    for batch_index in range(pred_active.size(0)):
        unmatched = torch.ones(
            pred_active.size(-1), dtype=torch.bool, device=pred_active.device
        )
        if matched_queries[batch_index]:
            unmatched[list(matched_queries[batch_index])] = False
        inactive_false_positive = (
            inactive_false_positive
            + (
                pred_active[batch_index]
                & unmatched[None]
                & batch["frame_mask"][batch_index, :, None]
            ).sum()
        )
        for frame in torch.nonzero(batch["frame_mask"][batch_index]).flatten().tolist():
            indices = torch.nonzero(pred_active[batch_index, frame]).flatten()
            if indices.numel() < 2:
                continue
            distances = torch.cdist(
                prediction["position"][batch_index, frame, indices],
                prediction["position"][batch_index, frame, indices],
            )
            duplicate = (
                duplicate + torch.triu(distances < duplicate_distance, diagonal=1).sum()
            )
    zero = prediction["position"].new_zeros(())
    return {
        "position_error": torch.stack(position_errors).mean()
        if position_errors
        else zero,
        "angular_error_deg": torch.stack(angular_errors).mean()
        if angular_errors
        else zero,
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "id_switches": id_switches,
        "duplicate_active_tracks": duplicate,
        "missed_gt_frames": missed,
        "inactive_query_false_positives": inactive_false_positive,
    }

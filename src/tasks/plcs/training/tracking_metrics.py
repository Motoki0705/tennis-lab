"""Lifecycle-aware localization and identity diagnostics for player tracks."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F

from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    common_lifecycle_tracking_metrics,
)
from src.tasks.plcs.training.tracking_losses import Assignment


def plcs_tracking_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
    *,
    config: TrackingMetricConfig,
) -> dict[str, torch.Tensor]:
    """Compute shared lifecycle metrics plus matched angular error."""
    metrics = cast(
        dict[str, torch.Tensor],
        common_lifecycle_tracking_metrics(
            prediction,
            batch,
            assignments,
            config=config,
        ),
    )
    angular_errors: list[torch.Tensor] = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch["target_presence"][batch_index, :, target_index]
                & batch["frame_mask"][batch_index]
            )
            if not active.any():
                continue
            cosine = (
                (
                    F.normalize(
                        prediction["rotation"][batch_index, active, query_index],
                        dim=-1,
                    )
                    * F.normalize(
                        batch["target_rotation"][batch_index, active, target_index],
                        dim=-1,
                    )
                )
                .sum(-1)
                .clamp(-1.0, 1.0)
            )
            angular_errors.append(torch.acos(cosine).mean() * (180.0 / math.pi))
    zero = prediction["position"].new_zeros(())
    metrics["angular_error_deg"] = (
        torch.stack(angular_errors).mean() if angular_errors else zero
    )
    return metrics

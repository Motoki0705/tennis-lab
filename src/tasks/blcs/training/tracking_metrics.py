"""Lifecycle-aware localization and identity diagnostics for ball tracks."""

from __future__ import annotations

import torch

from src.tasks.base.training.tracking_metrics import (
    common_lifecycle_tracking_metrics,
)
from src.tasks.blcs.training.tracking_losses import Assignment


def blcs_tracking_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
    *,
    presence_threshold: float = 0.5,
    duplicate_distance: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute shared lifecycle metrics for BLCS predictions."""
    return common_lifecycle_tracking_metrics(
        prediction,
        batch,
        assignments,
        presence_threshold=presence_threshold,
        duplicate_distance=duplicate_distance,
    )

"""Pose-conditioned compatibility API and shared observed-coordinate loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .coordinate_dataset import CoordinateWindowDataset, collate_coordinate_windows


class PoseWindowDataset(CoordinateWindowDataset):
    """Read either the original native-FPS manifest or the mixed-FPS manifest."""

    def __init__(self, manifest: Path, *, split: str, mdd_a: float, mdd_b: float) -> None:
        super().__init__(manifest, split=split, requires_pose=True, mdd_a=mdd_a, mdd_b=mdd_b)


def collate_pose_windows(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples or any(sample["input_kind"] != "mdd_pose" for sample in samples):
        raise ValueError("Expected nonempty pose-conditioned samples")
    return collate_coordinate_windows(samples)


def coordinate_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if prediction.shape != target.shape or valid.shape != target.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Coordinate loss shape/mask mismatch")
    if not bool(valid.any()):
        raise ValueError("No observed coordinate supervision")
    return torch.nn.functional.smooth_l1_loss(prediction[valid], target[valid], beta=.01)

"""Task-local Dice losses for court detection training."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.tasks.court_detection.geometry.pose import CourtDecodedPose


class CourtPoseLossTarget(Protocol):
    @property
    def translation_m(self) -> Tensor: ...

    @property
    def rotation(self) -> Tensor: ...

    @property
    def log_focal(self) -> Tensor: ...


class DiceLoss(nn.Module):
    """Per-class Dice loss averaged over classes (for segmentation)."""

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Parameters
        ----------
        logits:
            ``[B, C, H, W]`` raw logits.
        targets:
            ``[B, H, W]`` int64 labels.
        """
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).to(dtype=probs.dtype)

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BinaryDiceLoss(nn.Module):
    """Dice loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def rotation_geodesic_radians(prediction: Tensor, target: Tensor) -> Tensor:
    """Return stable per-sample SO(3) geodesic angles in radians."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-2:] != (3, 3):
        raise ValueError("Rotation geodesic inputs must share shape (B,3,3).")
    relative = prediction @ target.transpose(-1, -2)
    skew = torch.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = 0.5 * (
        relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2] - 1.0
    )
    return torch.atan2(sine, cosine)


def query_pose_losses(
    prediction: CourtDecodedPose,
    target: CourtPoseLossTarget,
) -> dict[str, Tensor]:
    """Compute explicit metric translation, SO(3), and log-focal losses."""
    target_translation = target.translation_m.to(
        device=prediction.translation_m.device,
        dtype=prediction.translation_m.dtype,
    )
    target_rotation = target.rotation.to(
        device=prediction.rotation.device,
        dtype=prediction.rotation.dtype,
    )
    target_log_focal = target.log_focal.to(
        device=prediction.log_focal.device,
        dtype=prediction.log_focal.dtype,
    )
    return {
        "pose_translation": F.mse_loss(
            prediction.translation_m,
            target_translation,
        ),
        "pose_rotation": rotation_geodesic_radians(
            prediction.rotation,
            target_rotation,
        ).mean(),
        "pose_focal": F.mse_loss(
            prediction.log_focal,
            target_log_focal,
        ),
    }


__all__ = [
    "BinaryDiceLoss",
    "DiceLoss",
    "query_pose_losses",
    "rotation_geodesic_radians",
]

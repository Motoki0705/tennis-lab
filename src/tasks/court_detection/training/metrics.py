"""Metrics for court detection training.

* **seg** — Mean IoU across 7 classes.
* **kp**  — Mean keypoint distance error (in pixels).
* **line** — Binary Dice score.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.utils.data.heatmaps import heatmaps_to_pixel_coords


class CourtDetectionMetrics:
    """Unified metrics accumulator for court detection tasks.

    Designed to be used per-stage (train/val) with ``update``/``compute``/``reset``.

    Parameters
    ----------
    task:
        One of ``"seg"``, ``"kp"``, ``"line"``.
    data_cfg:
        Data config dict (used to resolve num_classes / num_keypoints).
    """

    def __init__(self, task: str, data_cfg: dict | Any = None) -> None:
        self.task = task
        data_cfg = data_cfg or {}

        if task == "seg":
            self.num_classes = int(data_cfg.get("num_classes", 7))
            self._intersection: list[Tensor] = []
            self._union: list[Tensor] = []
        elif task == "kp":
            self.num_keypoints = int(data_cfg.get("num_keypoints", 14))
            self._distances: list[Tensor] = []
        elif task == "line":
            self._dice_sum: float = 0.0
            self._dice_count: int = 0

    def update(self, logits: Tensor, batch: dict[str, Tensor]) -> None:
        """Accumulate a batch of predictions."""
        if self.task == "seg":
            self._update_seg(logits, batch)
        elif self.task == "kp":
            self._update_kp(logits, batch)
        else:
            self._update_line(logits, batch)

    def _update_seg(self, logits: Tensor, batch: dict[str, Tensor]) -> None:
        """Compute per-class intersection and union for IoU."""
        preds = logits.argmax(dim=1)  # (B, H, W)
        targets = batch["mask"]  # (B, H, W)
        for c in range(self.num_classes):
            pred_c = (preds == c)
            target_c = (targets == c)
            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            self._intersection.append(intersection.detach().cpu())
            self._union.append(union.detach().cpu())

    def _update_kp(self, logits: Tensor, batch: dict[str, Tensor]) -> None:
        """Compute argmax distance to ground truth keypoints.

        Only keypoints marked visible in ``batch["kp_visible"]`` enter the
        metric: invisible keypoints have blanked heatmap targets, so the
        argmax of their predicted map carries no meaningful location.
        """
        coords_pred = heatmaps_to_pixel_coords(logits)  # (B, K, 2)
        coords_gt = batch["keypoints"]  # (B, K, 2)
        dist = torch.norm(coords_pred.cpu() - coords_gt.cpu(), dim=-1)  # (B, K)
        visible = batch["kp_visible"].cpu().bool()  # (B, K)
        self._distances.append(dist[visible].detach())

    def _update_line(self, logits: Tensor, batch: dict[str, Tensor]) -> None:
        """Compute per-batch binary Dice score."""
        preds = (torch.sigmoid(logits) > 0.5).float()
        targets = batch["mask"]
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        self._dice_sum += dice.item()
        self._dice_count += 1

    def compute(self) -> dict[str, float]:
        """Compute aggregated metrics."""
        if self.task == "seg":
            return self._compute_seg()
        if self.task == "kp":
            return self._compute_kp()
        return self._compute_line()

    def _compute_seg(self) -> dict[str, float]:
        if not self._intersection:
            return {"miou": 0.0}
        intersection = torch.stack(self._intersection).view(-1, self.num_classes).sum(0)
        union = torch.stack(self._union).view(-1, self.num_classes).sum(0)
        iou = intersection / (union + 1e-8)
        return {"miou": iou.mean().item()}

    def _compute_kp(self) -> dict[str, float]:
        if not self._distances:
            return {"mean_dist": 0.0}
        all_dist = torch.cat(self._distances, dim=0)  # flat over visible KPs
        if all_dist.numel() == 0:
            return {"mean_dist": 0.0}
        return {"mean_dist": all_dist.mean().item()}

    def _compute_line(self) -> dict[str, float]:
        if self._dice_count == 0:
            return {"dice": 0.0}
        return {"dice": self._dice_sum / self._dice_count}

    def reset(self) -> None:
        """Reset accumulated state."""
        if self.task == "seg":
            self._intersection = []
            self._union = []
        elif self.task == "kp":
            self._distances = []
        else:
            self._dice_sum = 0.0
            self._dice_count = 0

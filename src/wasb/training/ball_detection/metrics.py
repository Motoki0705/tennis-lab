"""Metrics for WASB ball detection training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class _MetricState:
    sum_sq_px: Tensor
    sum_abs_px: Tensor
    sum_sq_norm: Tensor
    count: Tensor
    correct: Tensor
    pred_min: Tensor
    pred_max: Tensor


def _initial_state(device: torch.device) -> _MetricState:
    zeros = torch.tensor(0.0, device=device)
    return _MetricState(
        sum_sq_px=zeros.clone(),
        sum_abs_px=zeros.clone(),
        sum_sq_norm=zeros.clone(),
        count=zeros.clone(),
        correct=zeros.clone(),
        pred_min=torch.tensor(float("inf"), device=device),
        pred_max=torch.tensor(float("-inf"), device=device),
    )


class WASBMetrics:
    """Running metrics for ball localization."""

    def __init__(self, accuracy_thresh_px: float = 5.0) -> None:
        self.state: _MetricState | None = None
        self.accuracy_thresh_px = accuracy_thresh_px

    def reset(self, device: torch.device | None = None) -> None:
        """Reset accumulated metrics."""
        if device is None and self.state is not None:
            device = self.state.count.device
        device = device or torch.device("cpu")
        self.state = _initial_state(device)

    def update(
        self,
        pred_heatmaps: Tensor,
        target_heatmaps: Tensor,
        visibility: Tensor | None,
        image_hw: tuple[int, int],
    ) -> None:
        """Update metrics with a new batch."""
        if self.state is None:
            self.reset(pred_heatmaps.device)
        assert self.state is not None  # for type checker

        device = pred_heatmaps.device
        mask = None
        if visibility is not None:
            mask = (visibility > 0).to(device=device, dtype=torch.float32)
        else:
            mask = torch.ones(
                pred_heatmaps.shape[:2], device=device, dtype=torch.float32
            )

        target_coords_norm = heatmap_argmax_coords(target_heatmaps)
        pred_coords_norm = heatmap_argmax_coords(pred_heatmaps)
        self.state.pred_min = torch.minimum(
            self.state.pred_min, pred_heatmaps.min().to(device)
        )
        self.state.pred_max = torch.maximum(
            self.state.pred_max, pred_heatmaps.max().to(device)
        )
        width, height = image_hw[1], image_hw[0]
        scale = torch.tensor([width, height], device=device, dtype=torch.float32)

        diff_norm = pred_coords_norm - target_coords_norm
        diff_px = diff_norm * scale

        abs_px = torch.linalg.norm(diff_px, dim=-1)
        sq_px = abs_px**2
        sq_norm = (diff_norm**2).sum(dim=-1)

        masked_count = mask.sum()
        self.state.sum_sq_px += (sq_px * mask).sum()
        self.state.sum_abs_px += (abs_px * mask).sum()
        self.state.sum_sq_norm += (sq_norm * mask).sum()
        self.state.count += masked_count
        self.state.correct += ((abs_px <= self.accuracy_thresh_px).float() * mask).sum()
        return None

    def compute(self) -> dict[str, float]:
        """Compute current metrics."""
        if self.state is None or self.state.count == 0:
            return {
                "rmse_px": 0.0,
                "mae_px": 0.0,
                "rmse_norm": 0.0,
                "accuracy": 0.0,
                "pred_min": 0.0,
                "pred_max": 0.0,
            }

        rmse_px = torch.sqrt(self.state.sum_sq_px / self.state.count)
        mae_px = self.state.sum_abs_px / self.state.count
        rmse_norm = torch.sqrt(self.state.sum_sq_norm / self.state.count)
        accuracy = self.state.correct / self.state.count.clamp(min=1.0)

        return {
            "rmse_px": rmse_px.item(),
            "mae_px": mae_px.item(),
            "rmse_norm": rmse_norm.item(),
            "accuracy": accuracy.item(),
            "pred_min": self.state.pred_min.item(),
            "pred_max": self.state.pred_max.item(),
        }


def heatmap_argmax_coords(pred_heatmaps: Tensor) -> Tensor:
    """Convert heatmaps to normalized (x, y) coordinates via argmax."""
    heatmaps = pred_heatmaps.detach()
    b, t, h, w = heatmaps.shape
    flat = heatmaps.view(b, t, -1)
    idx = torch.argmax(flat, dim=-1)  # [B, T]
    y = idx // w
    x = idx % w

    denom_w = max(w - 1, 1)
    denom_h = max(h - 1, 1)
    coords = torch.stack(
        (
            x.to(torch.float32) / denom_w,
            y.to(torch.float32) / denom_h,
        ),
        dim=-1,
    )  # [B, T, 2]
    return coords

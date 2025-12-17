"""Trajectory event-detection inference utilities.

This module loads a trained event detector and predicts per-frame event labels
from a ball (x, y) trajectory sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.wasb.training import EventDetectionLightningModule

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class EventDetectionResult:
    """Per-frame event prediction result."""

    pred_status: NDArray[np.int64]
    probs: NDArray[np.float32]


class TrajectoryEventDetector:
    """Wrapper around `EventDetectionLightningModule` for inference."""

    def __init__(self, module: EventDetectionLightningModule, *, device: str = "cpu") -> None:
        self.module = module.eval()
        self.device = torch.device(device)
        self.module.to(self.device)

    def predict(
        self,
        *,
        xy: NDArray[np.float32],
        visibility: NDArray[np.int32] | NDArray[np.bool_],
        xy_scale: tuple[float, float] = (1920.0, 1080.0),
    ) -> EventDetectionResult:
        """Predict event labels for a single trajectory.

        Args:
            xy: Array of shape (T, 2) in pixel coordinates.
            visibility: Array of shape (T,) where >0 indicates valid positions.
            xy_scale: (width, height) scaling used for normalization.

        """
        if xy.ndim != 2 or xy.shape[-1] != 2:
            raise ValueError(f"xy must have shape (T, 2), got {xy.shape}")
        t = xy.shape[0]
        if visibility.shape[0] != t:
            raise ValueError("visibility length must match xy length")

        with torch.no_grad():
            xy_t = torch.from_numpy(xy.astype(np.float32)).to(self.device)
            scale_t = torch.tensor(xy_scale, dtype=torch.float32, device=self.device)
            xy_norm = (xy_t / scale_t).unsqueeze(0)  # (1, T, 2)

            vis_np = (
                visibility.astype(np.int32) if visibility.dtype != np.bool_ else visibility.astype(np.int32)
            )
            vis_t = torch.from_numpy(vis_np).to(self.device).unsqueeze(0)  # (1, T)
            key_padding_mask = vis_t <= 0

            logits = self.module(xy_norm, key_padding_mask=key_padding_mask)  # (1, T, 3)
            probs = torch.softmax(logits, dim=-1)[0].to(dtype=torch.float32).cpu().numpy()
        pred = probs.argmax(axis=-1).astype(np.int64)
        return EventDetectionResult(pred_status=pred, probs=probs.astype(np.float32))


def load_event_detector_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> TrajectoryEventDetector:
    """Load an event detector from a Lightning checkpoint."""
    module = EventDetectionLightningModule.load_from_checkpoint(str(checkpoint_path))
    return TrajectoryEventDetector(module, device=device)

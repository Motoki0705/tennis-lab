"""Inference predictor for ball detection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.ball_detection.data.utils.input_adapter import to_model_input
from src.tasks.ball_detection.training.lightning_module import BallDetectionLightningModule
from src.tasks.base.inference.predictor import BasePredictor


class BallDetectionPredictor(BasePredictor):
    """Predictor for ball detection.

    Provides inference over frame sequences to produce ball heatmaps and
    peak coordinates.

    Attributes:
        model: Ball detection model instance.
        device: Device for inference.
        model_config: Model configuration dict used for input adaptation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.model_config = model_config or {}

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Load predictor from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file(s).
            device: Device for inference.
            **kwargs: Additional arguments (unused).

        Returns:
            BallDetectionPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        resolved_device = cls._resolve_device(device)

        lightning_module = BallDetectionLightningModule.load_from_checkpoint(
            checkpoints[0],
            map_location=resolved_device,
        )

        model = lightning_module.model
        model_config = dict(lightning_module.config.get("model", {}))

        return cls(model=model, device=resolved_device, model_config=model_config)

    @torch.no_grad()
    def predict(
        self,
        images: Tensor,
        return_heatmaps: bool = False,
    ) -> dict[str, Tensor]:
        """Run inference on a batch of frame sequences.

        Args:
            images: Input frames of shape ``(B, T, 3, H, W)`` as float32 in
                ``[0, 1]``. Already resized and normalized.
            return_heatmaps: Whether to include full probability heatmaps.

        Returns:
            Dictionary with CPU tensors:
                - ``coords``: Predicted peak coordinates in heatmap space,
                  shape ``(B, T, 2)`` as ``(x, y)``.
                - ``visibility``: Peak confidence per frame, shape ``(B, T)``.
                - ``heatmaps``: Probability heatmaps ``(B, T, H, W)`` if
                  *return_heatmaps* is True.
        """
        model_input = to_model_input(images.to(self.device), self.model_config)
        logits = self.model(model_input)

        # (B, 1, T, H, W) -> (B, T, H, W)
        logits = logits.squeeze(1)
        heatmaps = torch.sigmoid(logits)

        b, t, h, w = heatmaps.shape
        flat = heatmaps.reshape(b, t, -1)
        peak_values, peak_indices = flat.max(dim=-1)
        pred_x = (peak_indices % w).float()
        pred_y = torch.div(peak_indices, w, rounding_mode="floor").float()
        coords = torch.stack([pred_x, pred_y], dim=-1)

        result: dict[str, Tensor] = {
            "coords": coords.cpu(),
            "visibility": peak_values.cpu(),
        }
        if return_heatmaps:
            result["heatmaps"] = heatmaps.cpu()
        return result

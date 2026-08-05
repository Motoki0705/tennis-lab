"""Inference predictor for ball detection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.tasks.ball_detection.models.input_adapter import to_model_input
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.utils.configuration import PathResolver
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic


class BallDetectionPredictor(BasePredictor):
    """Predictor for ball detection.

    Provides inference over frame sequences to produce ball heatmaps and
    peak coordinates.

    Attributes:
        model: Ball detection model instance.
        device: Device for inference.
        model_config: Model configuration dict used for input adaptation.
        subpixel_refine: Whether peak coordinates are refined to sub-cell
            precision (log-parabolic fit) instead of raw lattice argmax.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        model_config: dict[str, object],
        *,
        subpixel_refine: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.model_config = model_config
        self.subpixel_refine = subpixel_refine

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        allow_device_fallback: bool,
        subpixel_refine: bool,
        strict: bool,
        weights_only: bool,
        **kwargs: Any,
    ) -> Self:
        """Load predictor from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file(s).
            device: Device for inference.
            **kwargs: ``subpixel_refine`` forwards to the constructor;
                remaining arguments are unused.

        Returns:
            BallDetectionPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            BallDetectionLightningModule,
            resolver=resolver,
            device=device,
            allow_device_fallback=allow_device_fallback,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

        model = lightning_module.model
        model_config = dict(lightning_module.config.model)

        return cls(
            model=model,
            device=resolved_device,
            model_config=model_config,
            subpixel_refine=subpixel_refine,
        )

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
                - ``coords``: Predicted normalized peak coordinates,
                  shape ``(B, T, 2)`` as ``(x, y)``.
                - ``visibility``: Peak confidence per frame, shape ``(B, T)``.
                - ``heatmaps``: Probability heatmaps ``(B, T, H, W)`` if
                  *return_heatmaps* is True.
        """
        with torch.no_grad():
            moved_images = self._to_device(self.device, images)[0]
            if moved_images is None:
                raise TypeError("Ball detector input must be a tensor.")
            images = moved_images
            model_input = to_model_input(images, self.model_config)
            logits = self.model(model_input)
            if not isinstance(logits, Tensor):
                raise TypeError("Ball detector output must be a tensor.")

            # (B, 1, T, H, W) -> (B, T, H, W)
            logits = logits.squeeze(1)
            heatmaps = torch.sigmoid(logits)
            coords, peak_values = heatmaps_to_argmax(heatmaps)
            if self.subpixel_refine:
                coords = refine_peaks_log_parabolic(heatmaps, coords)

            result: dict[str, Tensor] = {
                "coords": coords.cpu(),
                "visibility": peak_values.cpu(),
            }
            if return_heatmaps:
                result["heatmaps"] = heatmaps.cpu()
            return result

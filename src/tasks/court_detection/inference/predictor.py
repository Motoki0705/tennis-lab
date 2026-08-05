"""Inference predictor for court keypoint detection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.inference.preprocess import preprocess_court_image
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.configuration import PathResolver
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic


class CourtKeypointPredictor(BasePredictor):
    """Predictor for court keypoint detection using a court detection model.

    Loads a Lightning checkpoint and runs keypoint heatmap inference.

    Attributes:
        model: Court detection model instance.
        device: Device to run inference on.
        short_side: Short side resize for preprocessing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        short_side: int | None = None,
        *,
        subpixel_refine: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.short_side = short_side
        self.subpixel_refine = bool(subpixel_refine)

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
        **kwargs: Any,
    ) -> Self:
        """Load predictor from a Lightning checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to ``.ckpt`` checkpoint file.
        device:
            Device to run inference on.

        Returns
        -------
        CourtKeypointPredictor
        """
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            CourtDetectionLightningModule,
            resolver=resolver,
            device=device,
            allow_device_fallback=allow_device_fallback,
            weights_only=False,
            **kwargs,
        )

        model = lightning_module.model
        runtime = CourtTrainingConfig.from_config(lightning_module.config)
        short_side = runtime.data.augmentation.val_short_side

        return cls(
            model=model,
            device=resolved_device,
            short_side=short_side,
            subpixel_refine=subpixel_refine,
        )

    def preprocess(self, image: np.ndarray | Image.Image) -> tuple[Tensor, int, int]:
        """Preprocess image for inference.

        Returns
        -------
        tuple
            (tensor ``[1, 3, H', W']``, original_height, original_width)
        """
        if self.short_side is None:
            raise RuntimeError("short_side is required for raw-image preprocessing.")
        prepared: tuple[Tensor, int, int] = preprocess_court_image(
            image,
            short_side=self.short_side,
            device=self.device,
        )
        return prepared

    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
        return_heatmaps: bool = False,
        *,
        subpixel_refine: bool | None = None,
    ) -> dict[str, Tensor]:
        """Run inference on a single image.

        Returns
        -------
        dict
            - ``keypoints``: ``(K, 2)`` pixel coordinates on CPU.
            - ``scores``: ``(K,)`` sigmoid peak scores on CPU.
            - ``heatmaps`` (optional): ``(K, H, W)`` raw logits on CPU.
        """
        if isinstance(image, (np.ndarray, Image.Image)):
            image_tensor, orig_h, orig_w = self.preprocess(image)
        else:
            image_tensor = image.to(self.device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            orig_h, orig_w = image_tensor.shape[-2], image_tensor.shape[-1]

        with torch.no_grad():
            logits = self.model(image_tensor)  # [1, K, H, W]

        heatmaps = torch.sigmoid(logits)
        use_subpixel = (
            self.subpixel_refine if subpixel_refine is None else subpixel_refine
        )
        coords_normalized, scores = self._heatmaps_to_coords(
            heatmaps,
            subpixel_refine=bool(use_subpixel),
        )
        coords = coords_normalized[0].cpu()  # (K, 2)
        if orig_w > 1:
            coords[:, 0] *= float(orig_w - 1)
        else:
            coords[:, 0] = 0.0
        if orig_h > 1:
            coords[:, 1] *= float(orig_h - 1)
        else:
            coords[:, 1] = 0.0

        result: dict[str, Tensor] = {
            "keypoints": coords,
            "scores": scores[0].cpu(),
        }
        if return_heatmaps:
            result["heatmaps"] = logits[0].cpu()
        return result

    @staticmethod
    def _heatmaps_to_coords(
        heatmaps: Tensor,
        *,
        subpixel_refine: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Convert probability heatmaps to normalized coordinates and scores.

        Parameters
        ----------
        heatmaps:
            ``[B, K, H, W]`` sigmoid probability heatmaps.
        subpixel_refine:
            Whether to refine argmax peaks with log-parabolic sub-cell fitting.

        Returns
        -------
        tuple[Tensor, Tensor]
            Coordinates ``[B, K, 2]`` in normalised ``[0, 1]`` range and
            sigmoid peak scores ``[B, K]``.
        """
        coords, scores = heatmaps_to_argmax(heatmaps)
        if subpixel_refine:
            coords = refine_peaks_log_parabolic(heatmaps, coords)
        return coords, scores

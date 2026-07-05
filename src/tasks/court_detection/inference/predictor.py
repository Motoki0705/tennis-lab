"""Inference predictor for court keypoint detection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.court_detection.inference.preprocess import preprocess_court_image
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.data.heatmaps import heatmaps_to_argmax


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
        short_side: int = 640,
    ) -> None:
        self.model = model
        self.device = device
        self.short_side = short_side

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
            device,
            weights_only=bool(kwargs.pop("weights_only", False)),
            **kwargs,
        )

        model = lightning_module.model
        data_cfg = dict(lightning_module.config.get("data", {}))
        aug_cfg = data_cfg.get("augmentation", {})
        short_side = int(aug_cfg.get("val_short_side", 640))

        return cls(model=model, device=resolved_device, short_side=short_side)

    def preprocess(self, image: np.ndarray | Image.Image) -> tuple[Tensor, int, int]:
        """Preprocess image for inference.

        Returns
        -------
        tuple
            (tensor ``[1, 3, H', W']``, original_height, original_width)
        """
        return cast(
            "tuple[Tensor, int, int]",
            preprocess_court_image(
                image,
                short_side=self.short_side,
                device=self.device,
            ),
        )

    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
        return_heatmaps: bool = False,
    ) -> dict[str, Tensor]:
        """Run inference on a single image.

        Returns
        -------
        dict
            - ``keypoints``: ``(K, 2)`` pixel coordinates on CPU.
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

        coords = self._heatmaps_to_coords(logits)[0].cpu()  # (K, 2)
        if orig_w > 1:
            coords[:, 0] *= float(orig_w - 1)
        else:
            coords[:, 0] = 0.0
        if orig_h > 1:
            coords[:, 1] *= float(orig_h - 1)
        else:
            coords[:, 1] = 0.0

        result: dict[str, Tensor] = {"keypoints": coords}
        if return_heatmaps:
            result["heatmaps"] = logits[0].cpu()
        return result

    @staticmethod
    def _heatmaps_to_coords(heatmaps: Tensor) -> Tensor:
        """Convert heatmaps to normalized keypoint coordinates via argmax.

        Parameters
        ----------
        heatmaps:
            ``[B, K, H, W]`` raw logits.

        Returns
        -------
        Tensor
            ``[B, K, 2]`` in normalised ``[0, 1]`` range.
        """
        coords, _ = heatmaps_to_argmax(heatmaps)
        return coords

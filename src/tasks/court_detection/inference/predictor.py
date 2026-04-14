"""Inference predictor for court keypoint detection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.court_detection.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.data.heatmaps import heatmaps_to_argmax


class CourtKeypointPredictor(BasePredictor):
    """Predictor for court keypoint detection using CourtUNet.

    Loads a Lightning checkpoint and runs keypoint heatmap inference.

    Attributes:
        model: CourtUNet instance.
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
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        resolved_device = cls._resolve_device(device)

        lightning_module = CourtDetectionLightningModule.load_from_checkpoint(
            checkpoints[0],
            map_location=resolved_device,
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
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        orig_w, orig_h = image.size

        if orig_h <= orig_w:
            new_h = self.short_side
            new_w = int(round(orig_w * new_h / orig_h))
        else:
            new_w = self.short_side
            new_h = int(round(orig_h * new_w / orig_w))
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        image = image.resize((new_w, new_h), Image.BILINEAR)

        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)
        return img_tensor.unsqueeze(0).to(self.device), orig_h, orig_w

    @torch.no_grad()
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

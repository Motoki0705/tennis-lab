"""Inference predictors for court detection tasks."""

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


class CourtDetectionPredictor(BasePredictor):
    """Unified predictor for court detection tasks."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        task: str,
        short_side: int = 640,
    ) -> None:
        self.model = model
        self.device = device
        self.task = task
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
        """Load predictor from a Lightning checkpoint."""
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        resolved_device = cls._resolve_device(device)

        lightning_module = CourtDetectionLightningModule.load_from_checkpoint(
            checkpoints[0],
            map_location=resolved_device,
            weights_only=False,
        )

        model = lightning_module.model
        data_cfg = dict(lightning_module.config.get("data", {}))
        aug_cfg = data_cfg.get("augmentation", {})
        task = str(data_cfg.get("task", "seg"))
        short_side = int(aug_cfg.get("val_short_side", 640))

        return cls(
            model=model,
            device=resolved_device,
            task=task,
            short_side=short_side,
        )

    def preprocess(
        self,
        image: np.ndarray | Image.Image,
    ) -> tuple[Tensor, int, int, int, int]:
        """Preprocess image for inference."""
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
        return img_tensor.unsqueeze(0).to(self.device), orig_h, orig_w, new_h, new_w

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
        return_logits: bool = False,
    ) -> dict[str, Tensor]:
        """Run inference on a single image."""
        if isinstance(image, (np.ndarray, Image.Image)):
            image_tensor, orig_h, orig_w, resized_h, resized_w = self.preprocess(image)
        else:
            image_tensor = image.to(self.device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            orig_h, orig_w = image_tensor.shape[-2], image_tensor.shape[-1]
            resized_h, resized_w = orig_h, orig_w

        logits = self.model(image_tensor)  # [1, C, H, W]
        result: dict[str, Tensor]

        if self.task == "kp":
            coords = self._heatmaps_to_coords(logits)[0].cpu()
            coords[:, 0] *= orig_w / resized_w
            coords[:, 1] *= orig_h / resized_h
            result = {"keypoints": coords}
        elif self.task == "seg":
            result = {"mask": logits.argmax(dim=1)[0].cpu()}
        elif self.task == "line":
            result = {"mask": torch.sigmoid(logits)[0, 0].cpu()}
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        if return_logits:
            result["logits"] = logits[0].cpu()
        return result

    @staticmethod
    def _heatmaps_to_coords(heatmaps: Tensor) -> Tensor:
        """Convert heatmaps to keypoint coordinates using per-channel argmax."""
        bsz, num_kp, _, width = heatmaps.shape
        heatmaps_flat = heatmaps.view(bsz, num_kp, -1)
        max_idx = heatmaps_flat.argmax(dim=-1)
        x = (max_idx % width).float()
        y = torch.div(max_idx, width, rounding_mode="floor").float()
        return torch.stack([x, y], dim=-1)


class CourtKeypointPredictor(CourtDetectionPredictor):
    """Backward-compatible alias for keypoint predictor usage."""


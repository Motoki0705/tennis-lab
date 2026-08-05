"""Inference predictors for court segmentation and line detection."""

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


class _CourtMaskPredictor(BasePredictor):
    """Shared base for dense (per-pixel) court predictors.

    Subclasses only need to implement :meth:`_postprocess`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        short_side: int,
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
        *,
        resolver: PathResolver,
        device: str | torch.device,
        allow_device_fallback: bool,
        **kwargs: Any,
    ) -> Self:
        """Load predictor from a court-detection Lightning checkpoint."""
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

        return cls(model=model, device=resolved_device, short_side=short_side)

    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
        return_logits: bool = False,
    ) -> dict[str, Tensor]:
        """Run inference on a single image and return dense predictions."""
        if isinstance(image, (np.ndarray, Image.Image)):
            image_tensor, _, _ = preprocess_court_image(
                image,
                short_side=self.short_side,
                device=self.device,
            )
        else:
            image_tensor = image.to(self.device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(image_tensor)  # [1, C, H, W]
        return self._postprocess(logits[0].cpu(), return_logits=return_logits)

    def _postprocess(
        self,
        logits: Tensor,
        *,
        return_logits: bool,
    ) -> dict[str, Tensor]:
        raise NotImplementedError


class CourtSegPredictor(_CourtMaskPredictor):
    """Predictor for multi-class court cell segmentation."""

    def _postprocess(
        self,
        logits: Tensor,
        *,
        return_logits: bool,
    ) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {"seg_mask": logits.argmax(0).to(torch.long)}
        if return_logits:
            result["seg_logits"] = logits
        return result


class CourtLinePredictor(_CourtMaskPredictor):
    """Predictor for binary court white-line segmentation."""

    def _postprocess(
        self,
        logits: Tensor,
        *,
        return_logits: bool,
    ) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {"line_prob": torch.sigmoid(logits.squeeze(0))}
        if return_logits:
            result["line_logits"] = logits
        return result

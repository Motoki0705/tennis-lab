"""Typed inference predictor for court keypoint detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.model_io.adapters import CourtKeypointModelIO
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtModelIOError,
    CourtTask,
)
from src.tasks.court_detection.model_io.images import prepare_court_image
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.configuration import PathResolver


class CourtKeypointPredictor(BasePredictor[CourtKeypointPrediction]):
    """Predict court keypoints through one checkpoint-selected adapter."""

    def __init__(
        self,
        model_io: BoundModelIO[Mapping[str, object], Tensor, Tensor],
        device: torch.device,
        *,
        subpixel_refine: bool,
        max_peaks: int = 4,
    ) -> None:
        if not isinstance(model_io.adapter, CourtKeypointModelIO):
            raise CourtModelIOError(
                "CourtKeypointPredictor requires a keypoint model-I/O adapter."
            )
        if max_peaks <= 0:
            raise ValueError("Court keypoint max_peaks must be positive.")
        self.model_io = model_io
        self.model = model_io.model
        self.adapter = model_io.adapter
        self.device = device
        self.subpixel_refine = subpixel_refine
        self.max_peaks = max_peaks

        self.adapter.validate_model_pair(self.model)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        subpixel_refine: bool,
        max_peaks: int = 4,
        **kwargs: Any,
    ) -> Self:
        """Load one keypoint checkpoint and preserve its bound adapter."""
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            CourtDetectionLightningModule,
            resolver=resolver,
            device=device,
            weights_only=False,
            **kwargs,
        )
        adapter = lightning_module.model_io
        adapter.validate_model_pair(lightning_module.model)
        return cls(
            bind_model_io(lightning_module.model, adapter),
            resolved_device,
            subpixel_refine=subpixel_refine,
            max_peaks=max_peaks,
        )

    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
    ) -> CourtKeypointPrediction:
        """Return multi-peak channels, scores, validity, covariance, and heatmaps."""
        if isinstance(image, Tensor):
            if image.ndim not in {3, 4}:
                raise CourtModelIOError(
                    "Court predictor tensors must have shape (C,H,W) or (1,C,H,W)."
                )
            original_size_hw = (image.shape[-2], image.shape[-1])
            images = image.unsqueeze(0) if image.ndim == 3 else image
            if images.shape[0] != 1:
                raise CourtModelIOError("Court predictors accept exactly one image.")
            images = images.to(self.device)
        else:
            images, original_height, original_width = prepare_court_image(
                image,
                short_side=self.adapter.spec.short_side,
                device=self.device,
            )
            original_size_hw = (original_height, original_width)

        with torch.no_grad():
            call = self.adapter.prepare_images(images)
            logits = self.model(*call.model_args)
        return self.adapter.decode_prediction(
            logits,
            original_size_hw=original_size_hw,
            subpixel_refine=self.subpixel_refine,
            max_peaks=self.max_peaks,
        )

    @property
    def task(self) -> CourtTask:
        """Return the checkpoint-selected court task."""
        return self.adapter.spec.task

    @property
    def short_side(self) -> int:
        """Return the checkpoint-selected preprocessing size."""
        return self.adapter.spec.short_side


__all__ = ["CourtKeypointPredictor"]

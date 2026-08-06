"""Typed inference predictors for court segmentation and line detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, TypeAlias

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.model_io.adapters import (
    CourtLineModelIO,
    CourtSegmentationModelIO,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtLinePrediction,
    CourtModelIOError,
    CourtSegmentationPrediction,
)
from src.tasks.court_detection.model_io.images import prepare_court_image
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.configuration import PathResolver

CourtImage: TypeAlias = np.ndarray | Image.Image | Tensor
CourtBoundModelIO: TypeAlias = BoundModelIO[Mapping[str, object], Tensor, Tensor]


class CourtSegPredictor(BasePredictor[CourtSegmentationPrediction]):
    """Predict multi-class court segmentation through one selected adapter."""

    def __init__(self, model_io: CourtBoundModelIO, device: torch.device) -> None:
        if not isinstance(model_io.adapter, CourtSegmentationModelIO):
            raise CourtModelIOError(
                "CourtSegPredictor requires a segmentation model-I/O adapter."
            )
        self.model_io = model_io
        self.model = model_io.model
        self.adapter = model_io.adapter
        self.device = device
        self.adapter.validate_model_pair(self.model)
        self.model.to(device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        **kwargs: Any,
    ) -> Self:
        """Load one segmentation checkpoint and preserve its bound adapter."""
        module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            CourtDetectionLightningModule,
            resolver=resolver,
            device=device,
            weights_only=False,
            **kwargs,
        )
        adapter = module.model_io
        adapter.validate_model_pair(module.model)
        return cls(bind_model_io(module.model, adapter), resolved_device)

    def predict(self, image: CourtImage) -> CourtSegmentationPrediction:
        """Return a label mask and raw segmentation logits on CPU."""
        images, original_size_hw = _prepare_images(
            image,
            adapter=self.adapter,
            device=self.device,
        )
        with torch.no_grad():
            call = self.adapter.prepare_images(images)
            logits = self.model(*call.model_args)
        return self.adapter.decode_prediction(
            logits,
            original_size_hw=original_size_hw,
            subpixel_refine=False,
        )


class CourtLinePredictor(BasePredictor[CourtLinePrediction]):
    """Predict court-line probabilities through one selected adapter."""

    def __init__(self, model_io: CourtBoundModelIO, device: torch.device) -> None:
        if not isinstance(model_io.adapter, CourtLineModelIO):
            raise CourtModelIOError(
                "CourtLinePredictor requires a line model-I/O adapter."
            )
        self.model_io = model_io
        self.model = model_io.model
        self.adapter = model_io.adapter
        self.device = device
        self.adapter.validate_model_pair(self.model)
        self.model.to(device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        **kwargs: Any,
    ) -> Self:
        """Load one line checkpoint and preserve its bound adapter."""
        module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            CourtDetectionLightningModule,
            resolver=resolver,
            device=device,
            weights_only=False,
            **kwargs,
        )
        adapter = module.model_io
        adapter.validate_model_pair(module.model)
        return cls(bind_model_io(module.model, adapter), resolved_device)

    def predict(self, image: CourtImage) -> CourtLinePrediction:
        """Return a probability mask and raw line logits on CPU."""
        images, original_size_hw = _prepare_images(
            image,
            adapter=self.adapter,
            device=self.device,
        )
        with torch.no_grad():
            call = self.adapter.prepare_images(images)
            logits = self.model(*call.model_args)
        return self.adapter.decode_prediction(
            logits,
            original_size_hw=original_size_hw,
            subpixel_refine=False,
        )


def _prepare_images(
    image: CourtImage,
    *,
    adapter: CourtSegmentationModelIO | CourtLineModelIO,
    device: torch.device,
) -> tuple[Tensor, tuple[int, int]]:
    if isinstance(image, Tensor):
        if image.ndim not in {3, 4}:
            raise CourtModelIOError(
                "Court predictor tensors must have shape (C,H,W) or (1,C,H,W)."
            )
        original_size_hw = (image.shape[-2], image.shape[-1])
        images = image.unsqueeze(0) if image.ndim == 3 else image
        if images.shape[0] != 1:
            raise CourtModelIOError("Court predictors accept exactly one image.")
        return images.to(device), original_size_hw
    images, original_height, original_width = prepare_court_image(
        image,
        short_side=adapter.spec.short_side,
        device=device,
    )
    return images, (original_height, original_width)


__all__ = ["CourtLinePredictor", "CourtSegPredictor"]

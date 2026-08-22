"""Typed inference predictors for Court segmentation and line heads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, TypeAlias, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtLinePrediction,
    CourtLogits,
    CourtModelIOError,
    CourtSegmentationPrediction,
)
from src.tasks.court_detection.model_io.images import prepare_court_image
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.configuration import PathResolver

CourtImage: TypeAlias = np.ndarray | Image.Image | Tensor
CourtBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object],
    CourtLogits,
    CourtLogits,
]


class CourtSegPredictor(BasePredictor[CourtSegmentationPrediction]):
    """Predict the segmentation head from a Court bundle checkpoint."""

    def __init__(self, model_io: CourtBoundModelIO, device: torch.device) -> None:
        adapter = _require_adapter(model_io, kind="seg")
        self.model_io = model_io
        self.model = model_io.model
        self.adapter = adapter
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
        return cls(
            cast(
                CourtBoundModelIO,
                bind_model_io(module.model, adapter),
            ),
            resolved_device,
        )

    def predict(self, image: CourtImage) -> CourtSegmentationPrediction:
        images, original_size_hw = _prepare_images(
            image,
            adapter=self.adapter,
            device=self.device,
        )
        with torch.no_grad():
            call = self.adapter.prepare_images(images)
            logits = cast(CourtLogits, self.model(*call.model_args))
        return cast(
            CourtSegmentationPrediction,
            self.adapter.decode_prediction(
                "seg",
                logits["seg"],
                original_size_hw=original_size_hw,
                subpixel_refine=False,
            ),
        )


class CourtLinePredictor(BasePredictor[CourtLinePrediction]):
    """Predict the binary line head from a Court bundle checkpoint."""

    def __init__(self, model_io: CourtBoundModelIO, device: torch.device) -> None:
        adapter = _require_adapter(model_io, kind="line")
        self.model_io = model_io
        self.model = model_io.model
        self.adapter = adapter
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
        return cls(
            cast(
                CourtBoundModelIO,
                bind_model_io(module.model, adapter),
            ),
            resolved_device,
        )

    def predict(self, image: CourtImage) -> CourtLinePrediction:
        images, original_size_hw = _prepare_images(
            image,
            adapter=self.adapter,
            device=self.device,
        )
        with torch.no_grad():
            call = self.adapter.prepare_images(images)
            logits = cast(CourtLogits, self.model(*call.model_args))
        return cast(
            CourtLinePrediction,
            self.adapter.decode_prediction(
                "line",
                logits["line"],
                original_size_hw=original_size_hw,
                subpixel_refine=False,
            ),
        )


def _require_adapter(
    model_io: CourtBoundModelIO,
    *,
    kind: CourtTargetKind,
) -> CourtModelIOAdapter:
    adapter = model_io.adapter
    if not isinstance(adapter, CourtModelIOAdapter):
        raise CourtModelIOError("Court predictor requires CourtModelIOAdapter.")
    if kind not in adapter.spec.target_bundle.targets:
        raise CourtModelIOError(
            f"Court predictor requires a checkpoint with a {kind!r} head."
        )
    return adapter


def _prepare_images(
    image: CourtImage,
    *,
    adapter: CourtModelIOAdapter,
    device: torch.device,
) -> tuple[Tensor, tuple[int, int]]:
    if isinstance(image, Tensor):
        if image.ndim not in {3, 4}:
            raise CourtModelIOError(
                "Court predictor tensors must have shape "
                "(C,H,W) or (1,C,H,W)."
            )
        original_size_hw = (image.shape[-2], image.shape[-1])
        images = image.unsqueeze(0) if image.ndim == 3 else image
        if images.shape[0] != 1:
            raise CourtModelIOError(
                "Court predictors accept exactly one image."
            )
        return images.to(device), original_size_hw
    images, original_height, original_width = prepare_court_image(
        image,
        short_side=adapter.spec.short_side,
        device=device,
    )
    return images, (original_height, original_width)


__all__ = ["CourtLinePredictor", "CourtSegPredictor"]

"""Inference predictor for ball detection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.contracts import BallModelIOError, BallPrediction
from src.tasks.ball_detection.model_io.normalization import (
    IDENTITY_NORMALIZATION,
    BallImageNormalization,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import BoundModelIO
from src.utils.configuration import PathResolver
from src.utils.device import resolve_device


class BallDetectionPredictor(BasePredictor[BallPrediction]):
    """Predictor for ball detection.

    Provides inference over frame sequences to produce ball heatmaps and
    peak coordinates.

    Attributes:
        model: Ball detection model instance.
        device: Device for inference.
        subpixel_refine: Whether peak coordinates are refined to sub-cell
            precision (log-parabolic fit) instead of raw lattice argmax.
    """

    def __init__(
        self,
        model_io: BoundModelIO[Tensor, Tensor, Tensor],
        device: torch.device,
        *,
        subpixel_refine: bool,
        image_normalization: BallImageNormalization = IDENTITY_NORMALIZATION,
    ) -> None:
        if not isinstance(model_io.adapter, BallModelIOAdapter):
            raise BallModelIOError(
                "BallDetectionPredictor requires a ball model-I/O adapter."
            )
        self.model_io = model_io
        self.model = model_io.model
        self.adapter = model_io.adapter
        self.device = device
        self.subpixel_refine = subpixel_refine
        self.image_normalization = image_normalization

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
        strict: bool,
        weights_only: bool,
        **kwargs: Any,
    ) -> Self:
        """Restore one checkpoint's model and adapter without training state."""
        if kwargs:
            raise TypeError(f"Unsupported Ball checkpoint options: {sorted(kwargs)}")
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects a single checkpoint, got {len(checkpoints)} checkpoints."
            )
        resolved_device = resolve_device(device)
        loaded = load_ball_checkpoint(checkpoints[0], strict=strict, weights_only=weights_only)
        return cls(
            model_io=loaded.model_io,
            device=resolved_device,
            subpixel_refine=subpixel_refine,
            image_normalization=loaded.image_normalization,
        )

    def predict(
        self,
        images: Tensor,
    ) -> BallPrediction:
        """Run inference on a batch of frame sequences.

        Args:
            images: Input frames of shape ``(B, T, 3, H, W)`` as float32 in
                ``[0, 1]``. Already resized and scaled from raw RGB values.
                The checkpoint's saved normalization is applied internally.
        Returns:
            Typed coordinates, confidence, and probability heatmaps on CPU.
        """
        if not isinstance(images, Tensor):
            raise BallModelIOError("Ball detector input must be a Tensor.")
        with torch.no_grad():
            call = self.adapter.prepare_model_call(
                images.to(self.device), image_normalization=self.image_normalization,
            )
            logits = self.model(*call.model_args)
            return self.adapter.prediction(
                logits,
                call,
                subpixel_refine=self.subpixel_refine,
            )

    @property
    def configured_frames(self) -> int:
        """Return the checkpoint's declared sequence length contract."""
        return int(self.adapter.spec.configured_frames)

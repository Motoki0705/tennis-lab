"""Inference predictor for ball detection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.contracts import BallModelIOError, BallPrediction
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.utils.configuration import PathResolver


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
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

        adapter = lightning_module.model_io
        adapter.validate_model_pair(lightning_module.model)

        return cls(
            model_io=bind_model_io(lightning_module.model, adapter),
            device=resolved_device,
            subpixel_refine=subpixel_refine,
        )

    def predict(
        self,
        images: Tensor,
    ) -> BallPrediction:
        """Run inference on a batch of frame sequences.

        Args:
            images: Input frames of shape ``(B, T, 3, H, W)`` as float32 in
                ``[0, 1]``. Already resized and scaled from raw RGB values.
        Returns:
            Typed coordinates, confidence, and probability heatmaps on CPU.
        """
        if not isinstance(images, Tensor):
            raise BallModelIOError("Ball detector input must be a Tensor.")
        with torch.no_grad():
            call = self.adapter.prepare_model_call(images.to(self.device))
            logits = self.model(*call.model_args)
            return self.adapter.prediction(
                logits,
                call,
                subpixel_refine=self.subpixel_refine,
            )

    @property
    def configured_frames(self) -> int:
        """Return the checkpoint's declared sequence length contract."""
        return self.adapter.spec.configured_frames

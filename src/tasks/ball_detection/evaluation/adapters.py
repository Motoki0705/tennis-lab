"""Common prediction adapters for heterogeneous ball-detector checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.ball_detection.models.input_adapter import to_model_input
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)


class BallPredictionAdapter(Protocol):
    """Minimal prediction contract consumed by the evaluation runner."""

    device: torch.device

    def predict_heatmaps(
        self,
        images: Tensor,
        *,
        target_size_hw: tuple[int, int],
    ) -> Tensor:
        """Return probability heatmaps with shape ``(B, T, H, W)``."""
        ...


class LightningBallPredictionAdapter:
    """Adapt any registered Lightning ball detector to one heatmap contract."""

    def __init__(
        self,
        module: BallDetectionLightningModule,
        *,
        device: torch.device,
    ) -> None:
        self.module = module.to(device).eval()
        self.device = device
        self.model_config = module.config.get("model", {}) or {}

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        strict: bool,
        weights_only: bool,
    ) -> LightningBallPredictionAdapter:
        """Load a Lightning checkpoint and expose the common adapter."""
        module = BallDetectionLightningModule.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
            strict=strict,
            weights_only=weights_only,
        )
        return cls(module, device=device)

    def predict_heatmaps(
        self,
        images: Tensor,
        *,
        target_size_hw: tuple[int, int],
    ) -> Tensor:
        """Run model-specific input adaptation and validate output shape."""
        if images.ndim != 5:
            raise ValueError(
                "Evaluation images must have shape (B, T, C, H, W), "
                f"got {tuple(images.shape)}."
            )
        images = images.to(self.device, non_blocking=True)
        model_input = to_model_input(images, self.model_config)
        logits = self.module.model(model_input)
        expected_prefix = (images.shape[0], 1, images.shape[1])
        if logits.ndim != 5 or tuple(logits.shape[:3]) != expected_prefix:
            raise ValueError(
                "Ball detector output shape mismatch: expected prefix "
                f"{expected_prefix}, got {tuple(logits.shape)}."
            )
        logits = logits.squeeze(1)
        if logits.shape[-2:] != target_size_hw:
            batch_size, num_frames = logits.shape[:2]
            logits = F.interpolate(
                logits.reshape(batch_size * num_frames, 1, *logits.shape[-2:]),
                size=target_size_hw,
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_frames, *target_size_hw)
        return torch.sigmoid(logits)


def resolve_evaluation_device(device: str) -> torch.device:
    """Resolve ``auto`` and fail clearly for unavailable explicit CUDA."""
    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA evaluation requested but unavailable: {device}")
    return resolved


__all__ = [
    "BallPredictionAdapter",
    "LightningBallPredictionAdapter",
    "resolve_evaluation_device",
]

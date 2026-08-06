"""Checkpoint composition for ball evaluation heatmap prediction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)


class LightningBallHeatmapPredictor:
    """Expose a loaded, verified Lightning pair to the evaluation loop."""

    def __init__(
        self,
        module: BallDetectionLightningModule,
        *,
        device: torch.device,
    ) -> None:
        self.module = module.to(device).eval()
        self.device = device
        self.adapter: BallModelIOAdapter = module.model_io
        self.adapter.validate_model_pair(module.model)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        strict: bool,
        weights_only: bool,
    ) -> LightningBallHeatmapPredictor:
        """Load one checkpoint and verify its model-I/O pair."""
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
        """Predict probability heatmaps through the resolved adapter."""
        call = self.adapter.prepare_model_call(
            images.to(self.device, non_blocking=True)
        )
        logits = self.module.model(*call.model_args)
        return self.adapter.probability_heatmaps(
            logits,
            call,
            target_size_hw=target_size_hw,
        )


def resolve_evaluation_device(device: str) -> torch.device:
    """Resolve ``auto`` and reject unavailable explicitly requested CUDA."""
    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA evaluation requested but unavailable: {device}")
    return resolved


__all__ = ["LightningBallHeatmapPredictor", "resolve_evaluation_device"]

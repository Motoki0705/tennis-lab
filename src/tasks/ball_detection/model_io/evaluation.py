"""Checkpoint composition for ball evaluation heatmap prediction."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from src.tasks.ball_detection.inference.checkpoint import (
    LoadedBallCheckpoint,
    load_ball_checkpoint,
)
from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.utils.configuration import PathResolver


class CheckpointBallHeatmapPredictor:
    """Expose the strict inference checkpoint pair to the evaluation loop."""

    def __init__(
        self,
        loaded: LoadedBallCheckpoint,
        *,
        device: torch.device,
    ) -> None:
        self.model = loaded.model_io.model.to(device).eval()
        self.device = device
        self.adapter = cast(BallModelIOAdapter, loaded.model_io.adapter)
        self.image_normalization = loaded.image_normalization
        self.adapter.validate_model_pair(self.model)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        strict: bool,
        weights_only: bool,
        resolver: PathResolver | None = None,
    ) -> CheckpointBallHeatmapPredictor:
        """Load one checkpoint and verify its model-I/O pair."""
        loaded = load_ball_checkpoint(
            checkpoint_path,
            strict=strict,
            weights_only=weights_only,
            resolver=resolver,
        )
        return cls(loaded, device=device)

    def predict_heatmaps(
        self,
        images: Tensor,
        *,
        target_size_hw: tuple[int, int],
    ) -> Tensor:
        """Predict probability heatmaps through the resolved adapter."""
        call = self.adapter.prepare_model_call(
            images.to(self.device, non_blocking=True),
            image_normalization=self.image_normalization,
            preprocessed=True,
        )
        logits = self.model(*call.model_args)
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


__all__ = ["CheckpointBallHeatmapPredictor", "resolve_evaluation_device"]

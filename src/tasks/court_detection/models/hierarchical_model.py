"""Configurable hierarchical court detection model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.court_detection.models.decoder import build_court_decoder
from src.tasks.court_detection.models.encoders import build_court_encoder

if TYPE_CHECKING:
    from omegaconf import DictConfig


class CourtHierarchicalModel(nn.Module):
    """Court model assembled from configurable encoder and decoder modules."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 7,
        encoder_config: Mapping[str, Any] | None = None,
        decoder_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._validate_init_args(num_classes=num_classes)

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.encoder_config = dict(encoder_config or {})
        self.decoder_config = dict(decoder_config or {})

        self.encoder = build_court_encoder(
            encoder_name=str(self.encoder_config.get("name", "default")),
            in_channels=self.in_channels,
        )
        self.decoder = build_court_decoder(
            decoder_name=str(self.decoder_config.get("name", "fpn")),
            encoder_channels=self.encoder.feature_channels,
            decoder_channels=tuple(
                int(channel)
                for channel in self.decoder_config.get("channels", [64, 128, 256, 512])
            ),
        )
        self.final_conv = nn.Conv2d(
            self.decoder.output_channels, self.num_classes, kernel_size=1
        )

    @staticmethod
    def _validate_init_args(*, num_classes: int) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

    @classmethod
    def from_config(cls, config: DictConfig) -> CourtHierarchicalModel:
        """Create the model from a composed Hydra config."""

        model_cfg = config.get("model", {}) or {}
        return cls(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 7)),
            encoder_config=model_cfg.get("encoder", {}) or {},
            decoder_config=model_cfg.get("decoder", {}) or {},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_forward_input(x)

        input_hw = x.shape[-2:]
        decoded = self.decoder(self.encoder(x))
        if decoded.shape[-2:] != input_hw:
            decoded = F.interpolate(
                decoded, size=input_hw, mode="bilinear", align_corners=False
            )
        return self.final_conv(decoded)

    def _validate_forward_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "CourtHierarchicalModel expects input with shape (B, C, H, W), "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )


__all__ = ["CourtHierarchicalModel"]

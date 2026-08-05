"""Configurable hierarchical court detection model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.court_detection.configuration import CourtModelConfig
from src.tasks.court_detection.models.decoder import build_court_decoder
from src.tasks.court_detection.models.encoders import build_court_encoder


class CourtHierarchicalModel(nn.Module):
    """Court model assembled from configurable encoder and decoder modules."""

    def __init__(
        self,
        config: CourtModelConfig,
    ) -> None:
        super().__init__()
        self._validate_init_args(num_classes=config.num_classes)

        self.in_channels = config.in_channels
        self.num_classes = config.num_classes

        self.encoder = build_court_encoder(
            config=config.encoder,
            in_channels=self.in_channels,
        )
        self.decoder = build_court_decoder(
            decoder_name=config.decoder.name,
            encoder_channels=self.encoder.feature_channels,
            decoder_channels=config.decoder.channels,
            reassemble_factors=config.decoder.reassemble_factors,
        )
        self.final_conv = nn.Conv2d(
            self.decoder.output_channels, self.num_classes, kernel_size=1
        )

    @staticmethod
    def _validate_init_args(*, num_classes: int) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

    @classmethod
    def from_config(cls, config: CourtModelConfig) -> CourtHierarchicalModel:
        """Construct from the already validated typed model contract."""
        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_forward_input(x)

        input_hw = x.shape[-2:]
        decoded = self.decoder(self.encoder(x))
        if decoded.shape[-2:] != input_hw:
            decoded = F.interpolate(
                decoded, size=input_hw, mode="bilinear", align_corners=False
            )
        return self.final_conv.forward(decoded)

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

"""Configurable hierarchical court detection model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.tasks.court_detection.configuration import CourtModelConfig
from src.tasks.court_detection.models.decoder import build_court_decoder
from src.tasks.court_detection.models.encoders import build_court_encoder

CourtFeatures = tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]


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
        self._feature_forward = (
            self._forward_prepared_features
            if self.encoder.requires_prepared_features
            else self._forward_encoder
        )

    @staticmethod
    def _validate_init_args(*, num_classes: int) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

    @classmethod
    def from_config(cls, config: CourtModelConfig) -> CourtHierarchicalModel:
        """Construct from the already validated typed model contract."""
        return cls(config)

    def forward(
        self,
        x: torch.Tensor,
        feature_1: Tensor | None = None,
        feature_2: Tensor | None = None,
        feature_3: Tensor | None = None,
        feature_4: Tensor | None = None,
    ) -> torch.Tensor:
        """Decode validated images and any boundary-prepared encoder features."""
        features: CourtFeatures = (feature_1, feature_2, feature_3, feature_4)
        return self._feature_forward(x, features)

    def _forward_encoder(self, x: Tensor, features: CourtFeatures) -> Tensor:
        _ = features
        return self._decode(x, self.encoder(x))

    def _forward_prepared_features(
        self,
        x: Tensor,
        features: CourtFeatures,
    ) -> Tensor:
        prepared = (
            features[0],
            features[1],
            features[2],
            features[3],
        )
        return self._decode(
            x,
            prepared,
        )

    def _decode(
        self,
        x: Tensor,
        features: tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None],
    ) -> Tensor:
        input_hw = x.shape[-2:]
        decoded = self.decoder(features)
        decoded = F.interpolate(
            decoded, size=input_hw, mode="bilinear", align_corners=False
        )
        return self.final_conv.forward(decoded)


__all__ = ["CourtHierarchicalModel"]

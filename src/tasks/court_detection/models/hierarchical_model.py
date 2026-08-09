"""Configurable hierarchical Court model with bundle-derived heads."""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.tasks.court_detection.configuration import CourtModelConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.models.decoder import build_court_decoder
from src.tasks.court_detection.models.encoders import build_court_encoder

CourtFeatures = tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]


class CourtHierarchicalModel(nn.Module):
    """Run one encoder/decoder trunk and one head per selected target."""

    def __init__(
        self,
        config: CourtModelConfig,
        target_bundle: CourtTargetBundleSpec,
    ) -> None:
        super().__init__()
        if not target_bundle.targets:
            raise ValueError("Court model requires a non-empty target bundle.")
        self.in_channels = config.in_channels
        self.target_bundle_spec = target_bundle

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
        self.heads = nn.ModuleDict(
            {
                kind: nn.Conv2d(
                    self.decoder.output_channels,
                    spec.output_channels,
                    kernel_size=1,
                )
                for kind, spec in target_bundle.targets.items()
            }
        )
        self._feature_forward = (
            self._forward_prepared_features
            if self.encoder.requires_prepared_features
            else self._forward_encoder
        )

    @property
    def output_channels(self) -> Mapping[CourtTargetKind, int]:
        return self.target_bundle_spec.head_channels

    @classmethod
    def from_config(
        cls,
        config: CourtModelConfig,
        target_bundle: CourtTargetBundleSpec,
    ) -> CourtHierarchicalModel:
        return cls(config, target_bundle)

    def forward(
        self,
        x: Tensor,
        feature_1: Tensor | None = None,
        feature_2: Tensor | None = None,
        feature_3: Tensor | None = None,
        feature_4: Tensor | None = None,
    ) -> dict[CourtTargetKind, Tensor]:
        """Decode validated images and return exactly the selected head mapping."""
        features: CourtFeatures = (feature_1, feature_2, feature_3, feature_4)
        return self._feature_forward(x, features)

    def _forward_encoder(
        self,
        x: Tensor,
        features: CourtFeatures,
    ) -> dict[CourtTargetKind, Tensor]:
        _ = features
        return self._decode(x, self.encoder(x))

    def _forward_prepared_features(
        self,
        x: Tensor,
        features: CourtFeatures,
    ) -> dict[CourtTargetKind, Tensor]:
        return self._decode(x, features)

    def _decode(
        self,
        x: Tensor,
        features: CourtFeatures,
    ) -> dict[CourtTargetKind, Tensor]:
        decoded = self.decoder(features)
        decoded = F.interpolate(
            decoded,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            kind: self.heads[kind](decoded)
            for kind in self.target_bundle_spec.kinds
        }


__all__ = ["CourtHierarchicalModel"]

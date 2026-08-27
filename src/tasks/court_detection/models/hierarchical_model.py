"""Configurable hierarchical Court model with bundle-derived heads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

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
from src.tasks.court_detection.models.pose_head import (
    CourtModelOutput,
    CourtPose10DHead,
)
from src.tasks.court_detection.models.transformer_encoder import (
    CourtTransformerEncoder,
    TransformerEncoderOutput,
)

CourtFeatures = tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]


@dataclass(frozen=True, slots=True)
class CourtHierarchicalOutput:
    """Auxiliary output exposed by ``forward_with_pose`` when enabled."""

    dense_outputs: Mapping[CourtTargetKind, Tensor]
    spatial_feature_map: Tensor
    pose_query: Tensor
    pose_raw: Tensor

    @property
    def dense_logits(self) -> Mapping[CourtTargetKind, Tensor]:
        """Alias matching the typed model-I/O raw-output vocabulary."""

        return self.dense_outputs


class CourtHierarchicalModel(nn.Module):
    """Run one encoder/decoder trunk and one head per selected target."""

    # Type declarations only: these attributes are deliberately absent on the
    # legacy/disabled path so its module tree and parameter set remain exact.
    transformer_encoder: CourtTransformerEncoder
    pose_head: CourtPose10DHead

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
            config=config.decoder,
            encoder_channels=self.encoder.feature_channels,
        )

        transformer_config = config.transformer_encoder
        self._transformer_enabled = transformer_config.enabled
        if self._transformer_enabled:
            deepest_dim = int(self.encoder.feature_channels[-1])
            if transformer_config.dim != deepest_dim:
                raise ValueError(
                    "Transformer dimension must match the deepest encoder feature: "
                    f"{transformer_config.dim} != {deepest_dim}."
                )
            assert transformer_config.depth is not None
            assert transformer_config.num_heads is not None
            assert transformer_config.rope_dim is not None
            assert transformer_config.ffn_dim is not None
            assert transformer_config.rope_theta is not None
            assert transformer_config.dropout is not None
            self.transformer_encoder = CourtTransformerEncoder(
                dim=deepest_dim,
                depth=transformer_config.depth,
                num_heads=transformer_config.num_heads,
                rope_dim=transformer_config.rope_dim,
                ffn_dim=transformer_config.ffn_dim,
                rope_theta=transformer_config.rope_theta,
                dropout=transformer_config.dropout,
            )
            self.pose_head = CourtPose10DHead(
                input_dim=deepest_dim,
                hidden_dim=deepest_dim,
                depth=2,
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
        output_channels: Mapping[CourtTargetKind, int] = (
            self.target_bundle_spec.head_channels
        )
        return output_channels

    @property
    def transformer_enabled(self) -> bool:
        """Whether the optional Transformer/query/pose branch exists."""

        return bool(self._transformer_enabled)

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
        patch_valid_mask: Tensor | None = None,
    ) -> dict[CourtTargetKind, Tensor] | CourtModelOutput:
        """Decode images, returning a typed pose output only when enabled."""
        if patch_valid_mask is not None and not self._transformer_enabled:
            raise ValueError(
                "patch_valid_mask requires an enabled intermediate Transformer."
            )
        features: CourtFeatures = (feature_1, feature_2, feature_3, feature_4)
        return self._feature_forward(x, features, patch_valid_mask)

    def forward_with_pose(
        self,
        x: Tensor,
        feature_1: Tensor | None = None,
        feature_2: Tensor | None = None,
        feature_3: Tensor | None = None,
        feature_4: Tensor | None = None,
        patch_valid_mask: Tensor | None = None,
    ) -> CourtHierarchicalOutput:
        """Return dense heads plus spatial/query/raw-pose outputs.

        This API is intentionally separate from :meth:`forward`: the latter
        keeps the legacy mapping contract for all existing model-I/O adapters,
        including when the optional trunk is disabled.  Calling this method on
        a legacy model is an explicit error rather than a silent zero pose.
        """

        if not self._transformer_enabled:
            raise RuntimeError(
                "forward_with_pose requires an enabled intermediate Transformer."
            )
        features: CourtFeatures = (feature_1, feature_2, feature_3, feature_4)
        resolved_features = self._feature_forward_values(x, features)
        output, transformed_features = self._decode_with_transformer(
            x,
            resolved_features,
            patch_valid_mask,
        )
        assert transformed_features.pose_query is not None
        assert hasattr(self, "pose_head")
        pose_raw = self.pose_head(transformed_features.pose_query)
        spatial = transformed_features.spatial
        return CourtHierarchicalOutput(
            dense_outputs=MappingProxyType(output),
            spatial_feature_map=spatial,
            pose_query=transformed_features.pose_query,
            pose_raw=pose_raw.values,
        )

    def _forward_encoder(
        self,
        x: Tensor,
        features: CourtFeatures,
        patch_valid_mask: Tensor | None,
    ) -> dict[CourtTargetKind, Tensor] | CourtModelOutput:
        _ = features
        encoded = self.encoder(x)
        if self._transformer_enabled:
            output, transformed = self._decode_with_transformer(
                x, encoded, patch_valid_mask
            )
            assert transformed.pose_query is not None
            return CourtModelOutput(
                dense_logits=output,
                pose=self.pose_head(transformed.pose_query),
            )
        return self._decode(x, encoded)

    def _forward_prepared_features(
        self,
        x: Tensor,
        features: CourtFeatures,
        patch_valid_mask: Tensor | None,
    ) -> dict[CourtTargetKind, Tensor] | CourtModelOutput:
        if self._transformer_enabled:
            output, transformed = self._decode_with_transformer(
                x, features, patch_valid_mask
            )
            assert transformed.pose_query is not None
            return CourtModelOutput(
                dense_logits=output,
                pose=self.pose_head(transformed.pose_query),
            )
        return self._decode(x, features)

    def _feature_forward_values(
        self,
        x: Tensor,
        features: CourtFeatures,
    ) -> CourtFeatures:
        if self.encoder.requires_prepared_features:
            if any(feature is None for feature in features):
                raise ValueError(
                    "Prepared-feature DINOv3 route requires all four feature maps."
                )
            return features
        _ = features
        return cast(CourtFeatures, self.encoder(x))

    def _decode_with_transformer(
        self,
        x: Tensor,
        features: CourtFeatures,
        patch_valid_mask: Tensor | None,
    ) -> tuple[dict[CourtTargetKind, Tensor], TransformerEncoderOutput]:
        deepest = features[-1]
        if deepest is None:
            raise ValueError(
                "Enabled intermediate Transformer requires the deepest feature map."
            )
        transformed = self.transformer_encoder(
            deepest,
            patch_valid_mask=patch_valid_mask,
        )
        if transformed.pose_query is None:
            raise RuntimeError("Enabled Transformer unexpectedly returned no pose query.")
        transformed_features: CourtFeatures = (
            features[0],
            features[1],
            features[2],
            transformed.spatial,
        )
        return self._decode(x, transformed_features), transformed

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


__all__ = [
    "CourtHierarchicalModel",
    "CourtHierarchicalOutput",
    "CourtPose10DHead",
]

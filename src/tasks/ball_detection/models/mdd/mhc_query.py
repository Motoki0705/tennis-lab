"""Four image-query streams and one compressed temporal/FFN mHC span per layer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd.attention import TimeAttention
from src.tasks.ball_detection.models.mdd.config import (
    MDDMHCQueryConfig,
    MDDPretrainConfig,
)
from src.tasks.ball_detection.models.mdd.detectors import DeepMDDQueryDetector
from src.utils.models.components.ffn_layers import SwiGLU
from src.utils.models.components.mhc import ManifoldConstrainedHyperConnection


class MHCQueryBlock(nn.Module):
    def __init__(self, config: MDDMHCQueryConfig) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(config.dim, config.heads, dropout=config.dropout, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(config.dim) for _ in range(4))
        self.mhc = ManifoldConstrainedHyperConnection(config.mhc_config())
        self.temporal = TimeAttention(config.dim, config.heads, config.rope_base, config.dropout)
        self.ffn = nn.Sequential(SwiGLU(config.dim, config.ffn_dim), nn.Dropout(config.dropout))

    def forward(self, streams: Tensor, patches: Tensor, times: Tensor) -> Tensor:
        b, t, count, dim = streams.shape
        queries = self.norms[0](streams).reshape(b * t, count, dim)
        image = self.norms[1](patches).reshape(b * t, -1, dim)
        delta, _ = self.cross(queries, image, image, need_weights=False)
        residual = streams + delta.reshape_as(streams)
        return self.temporal_span(residual, times)

    def temporal_span(self, residual: Tensor, times: Tensor) -> Tensor:
        compressed, state = self.mhc.pre(residual, torch.ones_like(residual[..., 0], dtype=torch.bool))
        original = compressed.squeeze(-2)
        value = original + self.temporal(self.norms[2](original), times)
        value = value + self.ffn(self.norms[3](value))
        # One span includes both residual sublayers. Passing value itself would
        # count the compressed residual twice in the four output streams.
        return self.mhc.post((value - original).unsqueeze(-2), residual, state)


class DeepMDDMHCQueryDetector(DeepMDDQueryDetector):
    """Mean four streams -> LayerNorm -> Linear(2) -> FP32 sigmoid coordinates."""

    config: MDDMHCQueryConfig

    def _build_query_modules(self, config: MDDPretrainConfig) -> None:
        if not isinstance(config, MDDMHCQueryConfig):
            raise ValueError("mHC detector requires an explicit MDDMHCQueryConfig")
        self.ball_query = nn.Parameter(torch.randn(config.num_streams, config.dim) * .02)
        self.blocks = nn.ModuleList(MHCQueryBlock(config) for _ in range(config.layers))

    def encode_queries(self, rgb: Tensor, timestamps: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        features, streams = super().encode_queries(rgb, timestamps)
        return features, streams.mean(dim=-2)

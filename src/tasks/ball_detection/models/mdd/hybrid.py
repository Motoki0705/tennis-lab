"""Temporal query context fused into a fresh, frame-local DPT heatmap decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd.config import (
    MDDHeatmapHybridConfig,
    MDDHybridConfig,
)
from src.tasks.ball_detection.models.mdd.detectors import DeepMDDQueryTrunk
from src.tasks.ball_detection.models.mdd.execution import decode_frame_heatmaps
from src.utils.models.dpt import DPTDecoder


@dataclass(frozen=True)
class MDDHybridOutput:
    """Differentiable logits: heatmap B,T,ceil(H/4),ceil(W/4), presence B,T.

    No presence gate is applied here: absent frames can supervise every heatmap
    cell to zero, independently of the query's binary presence objective.
    """

    heatmap_logits: Tensor
    presence_logits: Tensor


class _HybridHeatmapTrunk(DeepMDDQueryTrunk):
    def __init__(self, config: MDDHybridConfig | MDDHeatmapHybridConfig, *, mdd_a: float, mdd_b: float) -> None:
        super().__init__(config, mdd_a=mdd_a, mdd_b=mdd_b)
        channels = (*self.encoder.feature_channels[:-1], self.encoder.feature_channels[-1] + config.dim)
        # DPT already projects each input with 1x1 Conv2d. Widen only its deepest
        # input; a separate compression layer would add a redundant bottleneck.
        self.decoder = DPTDecoder(encoder_channels=channels, decoder_channels=config.decoder_channels,
                                  reassemble_factors=(1., 1., 1., 1.), block_style="residual")
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(config.decoder_channels, config.decoder_channels // 2, 3, padding=1),
            nn.GELU(), nn.Conv2d(config.decoder_channels // 2, 1, 1))

    def decode_heatmaps(self, rgb: Tensor, features: tuple[Tensor, ...], query: Tensor) -> Tensor:
        deepest = features[-1]
        broadcast = query[..., None, None].expand(-1, -1, -1, *deepest.shape[-2:])
        conditioned = torch.cat((deepest, broadcast), dim=2)
        return decode_frame_heatmaps(self.decoder, self.heatmap_head, (*features[:-1], conditioned),
            output_size=((rgb.shape[-2] + 3) // 4, (rgb.shape[-1] + 3) // 4),
            frame_chunk_size=self.config.frame_chunk_size, activation_checkpointing=self.config.activation_checkpointing)


class DeepMDDHybridDetector(_HybridHeatmapTrunk):
    config: MDDHybridConfig

    def __init__(self, config: MDDHybridConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        if not isinstance(config, MDDHybridConfig):
            raise ValueError("Hybrid detector requires an explicit MDDHybridConfig")
        super().__init__(config, mdd_a=mdd_a, mdd_b=mdd_b)
        self.presence_head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 1))

    def forward(self, rgb: Tensor, timestamps: Tensor) -> MDDHybridOutput:
        features, query = self.encode_queries(rgb, timestamps)
        return MDDHybridOutput(
            heatmap_logits=self.decode_heatmaps(rgb, features, query),
            presence_logits=self.presence_head(query).squeeze(-1).float(),
        )


class DeepMDDHeatmapHybridDetector(_HybridHeatmapTrunk):
    config: MDDHeatmapHybridConfig

    def __init__(self, config: MDDHeatmapHybridConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        if not isinstance(config, MDDHeatmapHybridConfig):
            raise ValueError("Heatmap-only hybrid requires an explicit MDDHeatmapHybridConfig")
        super().__init__(config, mdd_a=mdd_a, mdd_b=mdd_b)

    def forward(self, rgb: Tensor, timestamps: Tensor) -> Tensor:
        features, query = self.encode_queries(rgb, timestamps)
        return self.decode_heatmaps(rgb, features, query)

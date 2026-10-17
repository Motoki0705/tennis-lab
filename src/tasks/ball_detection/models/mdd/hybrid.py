"""Temporal query context fused into a fresh, frame-local DPT heatmap decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.ball_detection.models.mdd.config import MDDHybridConfig
from src.tasks.ball_detection.models.mdd.detectors import DeepMDDQueryTrunk
from src.utils.models.dpt import DPTDecoder


@dataclass(frozen=True)
class MDDHybridOutput:
    """Differentiable logits: heatmap B,T,ceil(H/4),ceil(W/4), presence B,T.

    No presence gate is applied here: absent frames can supervise every heatmap
    cell to zero, independently of the query's binary presence objective.
    """

    heatmap_logits: Tensor
    presence_logits: Tensor


class DeepMDDHybridDetector(DeepMDDQueryTrunk):
    config: MDDHybridConfig

    def __init__(self, config: MDDHybridConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        if not isinstance(config, MDDHybridConfig):
            raise ValueError("Hybrid detector requires an explicit MDDHybridConfig")
        super().__init__(config, mdd_a=mdd_a, mdd_b=mdd_b)
        channels = (*self.encoder.feature_channels[:-1], self.encoder.feature_channels[-1] + config.dim)
        # DPT already projects each input with 1x1 Conv2d. Widen only its deepest
        # input; a separate compression layer would add a redundant bottleneck.
        self.decoder = DPTDecoder(encoder_channels=channels, decoder_channels=config.decoder_channels,
                                  reassemble_factors=(1., 1., 1., 1.), block_style="residual")
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(config.decoder_channels, config.decoder_channels // 2, 3, padding=1),
            nn.GELU(), nn.Conv2d(config.decoder_channels // 2, 1, 1))
        self.presence_head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 1))

    def forward(self, rgb: Tensor, timestamps: Tensor) -> MDDHybridOutput:
        features, query = self.encode_queries(rgb, timestamps)
        deepest = features[-1]
        broadcast = query[..., None, None].expand(-1, -1, -1, *deepest.shape[-2:])
        conditioned = torch.cat((deepest, broadcast), dim=2)
        fused = self.decoder([f.flatten(0, 1) for f in (*features[:-1], conditioned)])
        logits = self.heatmap_head(F.interpolate(
            fused, size=((rgb.shape[-2] + 3) // 4, (rgb.shape[-1] + 3) // 4),
            mode="bilinear", align_corners=False))
        return MDDHybridOutput(
            heatmap_logits=logits.reshape(*rgb.shape[:2], *logits.shape[-2:]),
            presence_logits=self.presence_head(query).squeeze(-1).float(),
        )

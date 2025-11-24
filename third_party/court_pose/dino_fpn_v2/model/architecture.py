from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DinoBackboneConfig, DinoFpnBackbone
from .decoder import FpnHeatmapDecoder, HeatmapDecoderConfig


@dataclass
class DinoFpnModelConfig:
    backbone: DinoBackboneConfig
    decoder: HeatmapDecoderConfig


class DinoFpnHeatmapNet(nn.Module):
    """Full model: DINOv3 backbone + FPN decoder that outputs dense heatmaps."""

    def __init__(self, cfg: DinoFpnModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DinoFpnBackbone(cfg.backbone)
        self.decoder = FpnHeatmapDecoder(cfg.decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyramid = self.backbone(x)
        heatmaps = self.decoder(pyramid)
        if heatmaps.shape[-2:] != x.shape[-2:]:
            heatmaps = F.interpolate(
                heatmaps,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return heatmaps


__all__ = ["DinoFpnHeatmapNet", "DinoFpnModelConfig"]

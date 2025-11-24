from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeatmapDecoderConfig:
    num_keypoints: int
    base_channels: int = 256
    inner_channels: Sequence[int] = (256, 128, 64)
    upsample_mode: str = "bilinear"
    final_activation: str | None = None  # e.g. "sigmoid" or "softmax"


class FpnHeatmapDecoder(nn.Module):
    """Fuse FPN features and upsample to produce dense heatmaps."""

    def __init__(self, cfg: HeatmapDecoderConfig) -> None:
        super().__init__()
        if cfg.num_keypoints is None:
            raise ValueError("HeatmapDecoderConfig.num_keypoints must be provided.")
        self.cfg = cfg
        self.upsample_mode = cfg.upsample_mode

        layers: list[nn.Module] = []
        in_channels = cfg.base_channels
        for hidden in cfg.inner_channels:
            layers.append(nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Upsample(scale_factor=2.0, mode=cfg.upsample_mode, align_corners=False))
            in_channels = hidden
        self.head = nn.Conv2d(in_channels, cfg.num_keypoints, kernel_size=1)
        self.blocks = nn.Sequential(*layers) if layers else nn.Identity()

        if cfg.final_activation is None:
            self._activation: nn.Module | None = None
        elif cfg.final_activation.lower() == "sigmoid":
            self._activation = nn.Sigmoid()
        elif cfg.final_activation.lower() == "softmax":
            self._activation = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported final activation '{cfg.final_activation}'.")

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        if not pyramid:
            raise ValueError("FpnHeatmapDecoder received an empty feature pyramid.")
        if "P3" not in pyramid:
            raise KeyError("FpnHeatmapDecoder expects a 'P3' tensor in the pyramid.")

        p3 = pyramid["P3"]
        fused = p3
        for level in ("P4", "P5"):
            if level not in pyramid:
                continue
            fused = fused + F.interpolate(
                pyramid[level],
                size=p3.shape[-2:],
                mode=self.upsample_mode,
                align_corners=False,
            )

        x = self.blocks(fused)
        heatmaps = self.head(x)
        if self._activation is not None:
            heatmaps = self._activation(heatmaps)
        return heatmaps


__all__ = ["FpnHeatmapDecoder", "HeatmapDecoderConfig"]

"""Reusable convolution blocks for court detection models."""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable 2-D convolution (depthwise to pointwise)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class Conv2dWiseWiseBlock(nn.Module):
    """Conv2d -> DWSepConv -> DWSepConv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.wise_1 = DepthwiseSeparableConv2d(out_channels, out_channels)
        self.wise_2 = DepthwiseSeparableConv2d(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.wise_1(x)
        x = self.wise_2(x)
        return x


__all__ = ["Conv2dWiseWiseBlock", "DepthwiseSeparableConv2d"]
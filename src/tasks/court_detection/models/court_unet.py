"""2D U-Net for court detection tasks.

A purely 2-D U-Net with depthwise-separable convolutions.
Full-resolution output.

Supports three modes via ``num_classes``:

* **Segmentation** (``num_classes=7``): 6 court cells + background.
* **Keypoint heatmap** (``num_classes=14``): per-keypoint Gaussian heatmaps.
* **Line segmentation** (``num_classes=1``): binary white-line mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable 2-D convolution (depthwise → pointwise)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1, groups=in_channels, bias=False,
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
    """``Conv2d → DWSepConv → DWSepConv`` block."""

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


class EncoderBlock2d(nn.Module):
    """Encoder stage: Conv2dWiseWise block producing a skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = Conv2dWiseWiseBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock2d(nn.Module):
    """Decoder stage: concatenate skip then Conv2dWiseWise."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = Conv2dWiseWiseBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class CourtUNet(nn.Module):
    """2-D U-Net for court detection (segmentation / keypoint heatmap / line).

    Parameters
    ----------
    in_channels:
        Number of input channels (default ``3`` for RGB).
    num_classes:
        Number of output channels.
        * ``7`` for segmentation (6 cells + background).
        * ``14`` for keypoint heatmap regression.
        * ``1`` for binary line segmentation.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 7) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.enc1 = EncoderBlock2d(in_channels, 64)
        self.enc2 = EncoderBlock2d(64, 128)
        self.enc3 = EncoderBlock2d(128, 256)

        self.bottleneck_1 = Conv2dWiseWiseBlock(256, 512)
        self.bottleneck_2 = Conv2dWiseWiseBlock(512, 512)

        self.dec3 = DecoderBlock2d(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = DecoderBlock2d(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = DecoderBlock2d(in_channels=128, skip_channels=64, out_channels=64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.pre_head_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            ``[B, in_channels, H, W]`` input tensor.

        Returns
        -------
        torch.Tensor
            ``[B, num_classes, H, W]`` raw logits (full resolution).
        """
        input_hw = x.shape[-2:]
        x = self.stem(x)

        s1 = self.enc1(x)
        x = self.pool(s1)

        s2 = self.enc2(x)
        x = self.pool(s2)

        s3 = self.enc3(x)
        x = self.pool(s3)

        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)

        x = self.upsample(x)
        x = self.dec3(x, s3)

        x = self.upsample(x)
        x = self.dec2(x, s2)

        x = self.upsample(x)
        x = self.dec1(x, s1)

        x = self.pre_head_upsample(x)
        if x.shape[-2:] != input_hw:
            x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)

        return self.final_conv(x)

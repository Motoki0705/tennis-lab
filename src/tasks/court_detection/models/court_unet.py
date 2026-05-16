"""Legacy 2-D U-Net for court detection checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.court_detection.models.blocks import Conv2dWiseWiseBlock


class EncoderBlock2d(nn.Module):
    """Encoder stage producing a skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = Conv2dWiseWiseBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock2d(nn.Module):
    """Decoder stage that fuses an upsampled tensor with a skip tensor."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = Conv2dWiseWiseBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.block(torch.cat([x, skip], dim=1))


class CourtUNet(nn.Module):
    """Legacy full-resolution U-Net used by older court detection checkpoints."""

    def __init__(self, in_channels: int = 3, num_classes: int = 7) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.enc1 = EncoderBlock2d(in_channels, 64)
        self.enc2 = EncoderBlock2d(64, 128)
        self.enc3 = EncoderBlock2d(128, 256)

        self.bottleneck_1 = Conv2dWiseWiseBlock(256, 512)
        self.bottleneck_2 = Conv2dWiseWiseBlock(512, 512)

        self.dec3 = DecoderBlock2d(512, 256, 256)
        self.dec2 = DecoderBlock2d(256, 128, 128)
        self.dec1 = DecoderBlock2d(128, 64, 64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        self.pre_head_upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]

        x = self.stem(x)
        s1 = self.enc1(x)
        x = self.pool(s1)
        s2 = self.enc2(x)
        x = self.pool(s2)
        s3 = self.enc3(x)
        x = self.pool(s3)

        x = self.bottleneck_2(self.bottleneck_1(x))

        x = self.upsample(x)
        x = self.dec3(x, s3)
        x = self.upsample(x)
        x = self.dec2(x, s2)
        x = self.upsample(x)
        x = self.dec1(x, s1)

        x = self.pre_head_upsample(x)
        if x.shape[-2:] != input_hw:
            x = F.interpolate(
                x,
                size=input_hw,
                mode="bilinear",
                align_corners=False,
            )
        return self.final_conv(x)


__all__ = ["CourtUNet", "DecoderBlock2d", "EncoderBlock2d"]

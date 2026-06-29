"""Reusable encoder modules for court detection models."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.tasks.court_detection.models.blocks import Conv2dWiseWiseBlock


class CourtDefaultEncoder(nn.Module):
    """Default native encoder extracted from the original court U-Net."""

    feature_channels = (64, 128, 256, 512)

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.enc1 = Conv2dWiseWiseBlock(self.in_channels, 64)
        self.enc2 = Conv2dWiseWiseBlock(64, 128)
        self.enc3 = Conv2dWiseWiseBlock(128, 256)
        self.bottleneck_1 = Conv2dWiseWiseBlock(256, 512)
        self.bottleneck_2 = Conv2dWiseWiseBlock(512, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_forward_input(x)

        x = self.stem(x)

        feat1 = self.enc1(x)
        x = self.pool(feat1)

        feat2 = self.enc2(x)
        x = self.pool(feat2)

        feat3 = self.enc3(x)
        x = self.pool(feat3)

        feat4 = self.bottleneck_2(self.bottleneck_1(x))
        return (feat1, feat2, feat3, feat4)

    def _validate_forward_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "CourtDefaultEncoder expects input with shape (B, C, H, W), "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )


def build_court_encoder(
    *,
    encoder_name: str = "default",
    in_channels: int = 3,
) -> nn.Module:
    """Build the requested court encoder."""

    if str(encoder_name) != "default":
        raise ValueError(
            f"Unknown court encoder '{encoder_name}'. Supported: ['default']."
        )
    return CourtDefaultEncoder(in_channels=in_channels)


__all__ = ["CourtDefaultEncoder", "build_court_encoder"]

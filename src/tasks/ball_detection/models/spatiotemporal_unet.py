"""Spatio-temporal U-Net for sequence-based ball detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.tasks.ball_detection.data.utils.input_adapter import resolve_model_in_channels

if TYPE_CHECKING:
    from omegaconf import DictConfig


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable 2D convolution block."""

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
        """Apply depthwise and pointwise convolutions."""
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class Conv2dWiseWiseBlock(nn.Module):
    """Conv2d block followed by two depthwise-separable blocks."""

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
        """Apply the 2D convolution stack."""
        x = self.conv2d(x)
        x = self.wise_1(x)
        x = self.wise_2(x)
        return x


class Conv3dBlock(nn.Module):
    """Two Conv3d-BN-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 3D convolution stack."""
        return self.conv3d_2(self.conv3d_1(x))


class EncoderBlock(nn.Module):
    """Spatial 2D encoder followed by a spatio-temporal 3D block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block_2d = Conv2dWiseWiseBlock(in_channels, out_channels)
        self.block_3d = Conv3dBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one resolution level and return 3D and 2D skip tensors."""
        bsz, channels, timesteps, height, width = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(
            bsz * timesteps, channels, height, width
        )
        x_2d = self.block_2d(x_2d)
        skip_2d = x_2d
        _, out_channels, out_h, out_w = skip_2d.shape
        x_3d = skip_2d.view(bsz, timesteps, out_channels, out_h, out_w).permute(
            0, 2, 1, 3, 4
        ).contiguous()
        skip_3d = self.block_3d(x_3d)
        return skip_3d, skip_2d


class BottleneckBlock(nn.Module):
    """2D bottleneck operating after temporal pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block_1 = Conv2dWiseWiseBlock(in_channels, out_channels)
        self.block_2 = Conv2dWiseWiseBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bottleneck block to the pooled feature tensor."""
        bsz, channels, timesteps, height, width = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(
            bsz * timesteps, channels, height, width
        )
        x_2d = self.block_2(self.block_1(x_2d))
        _, out_channels, out_h, out_w = x_2d.shape
        return x_2d.view(bsz, timesteps, out_channels, out_h, out_w).permute(
            0, 2, 1, 3, 4
        ).contiguous()


class DecoderBlock(nn.Module):
    """Symmetric decoder block with 3D and 2D skip fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block_3d = Conv3dBlock(in_channels + skip_channels, out_channels)
        self.block_2d = Conv2dWiseWiseBlock(out_channels + skip_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip_3d: torch.Tensor,
        skip_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse the decoder state with 3D and 2D skip tensors."""
        x_3d = self.block_3d(torch.cat([x, skip_3d], dim=1))
        bsz, channels, timesteps, height, width = x_3d.shape
        x_2d = x_3d.permute(0, 2, 1, 3, 4).contiguous().view(
            bsz * timesteps, channels, height, width
        )
        x_2d = self.block_2d(torch.cat([x_2d, skip_2d], dim=1))
        _, out_channels, out_h, out_w = x_2d.shape
        return x_2d.view(bsz, timesteps, out_channels, out_h, out_w).permute(
            0, 2, 1, 3, 4
        ).contiguous()


class SpatioTemporalUNet(nn.Module):
    """Spatio-temporal U-Net producing per-frame half-resolution logits.

    Input shape:
        ``(B, C, T, H, W)``

    Output shape:
        ``(B, num_classes, T, H/2, W/2)``
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        self.stem = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                self.in_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.enc1 = EncoderBlock(self.in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.bottleneck = BottleneckBlock(256, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.final_conv = nn.Conv3d(64, self.num_classes, kernel_size=1)

    @classmethod
    def from_config(cls, config: DictConfig) -> SpatioTemporalUNet:
        """Create the model from a composed Hydra config."""
        model_cfg = config.get("model", {}) or {}
        return cls(
            in_channels=resolve_model_in_channels(model_cfg),
            num_classes=int(model_cfg.get("num_classes", 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the spatio-temporal U-Net."""
        if x.ndim != 5:
            raise ValueError(
                "SpatioTemporalUNet expects input with shape (B, C, T, H, W), "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )
        if x.shape[2] < 8:
            raise ValueError(
                "SpatioTemporalUNet expects at least 8 frames because the temporal axis "
                "is pooled three times. "
                f"Received T={x.shape[2]}."
            )

        x = self.stem(x)
        s3d_1, s2d_1 = self.enc1(x)
        x = self.pool(s3d_1)
        s3d_2, s2d_2 = self.enc2(x)
        x = self.pool(s3d_2)
        s3d_3, s2d_3 = self.enc3(x)
        x = self.pool(s3d_3)
        x = self.bottleneck(x)
        x = self.upsample(x)
        x = self.dec3(x, s3d_3, s2d_3)
        x = self.upsample(x)
        x = self.dec2(x, s3d_2, s2d_2)
        x = self.upsample(x)
        x = self.dec1(x, s3d_1, s2d_1)
        return self.final_conv(x)


__all__ = ["SpatioTemporalUNet"]

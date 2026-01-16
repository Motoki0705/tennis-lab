"""Output heads for court keypoint detection."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class HeatmapHead(nn.Module):
    """Heatmap prediction head for keypoint detection.

    Uses transposed convolutions to upsample features and predict
    per-keypoint heatmaps.

    Args:
        in_channels: Number of input channels from backbone.
        num_keypoints: Number of keypoints to predict.
        heatmap_size: Output heatmap size [H, W].
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 20,
        heatmap_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()

        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        # Deconvolution layers to upsample
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final convolution to predict heatmaps
        self.final_conv = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features from backbone, shape (B, C, H, W).

        Returns:
            Heatmaps of shape (B, K, Hm, Wm).
        """
        x = self.deconv_layers(x)
        heatmaps = self.final_conv(x)

        # Interpolate to exact heatmap size if needed
        if heatmaps.shape[-2:] != self.heatmap_size:
            heatmaps = nn.functional.interpolate(
                heatmaps,
                size=self.heatmap_size,
                mode="bilinear",
                align_corners=False,
            )

        return heatmaps

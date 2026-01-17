"""Court keypoint detection model.

Detects 20 court keypoints (CourtKP20) from tennis images using
heatmap regression with various backbone architectures.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.court_detection.models.components.backbones import build_backbone
from src.court_detection.models.components.heads import HeatmapHead

NUM_KEYPOINTS = 20


class CourtKeypointModel(nn.Module):
    """Court keypoint detection model.

    Uses a CNN backbone (ResNet, HRNet, etc.) followed by a heatmap head
    to predict 2D locations of 20 court keypoints.

    Args:
        backbone: Backbone configuration dict with 'name' and 'pretrained'.
        head: Head configuration dict with 'type', 'num_keypoints', etc.
        input_size: Input image size [H, W].
    """

    def __init__(
        self,
        backbone: dict[str, Any],
        head: dict[str, Any],
        input_size: list[int] | tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        self.input_size = tuple(input_size)
        self.num_keypoints = head.get("num_keypoints", NUM_KEYPOINTS)
        self.heatmap_size = tuple(head.get("heatmap_size", [64, 64]))

        # Build backbone
        self.backbone, backbone_channels = build_backbone(
            name=backbone.get("name", "resnet50"),
            pretrained=backbone.get("pretrained", True),
        )

        # Build head
        self.head = HeatmapHead(
            in_channels=backbone_channels,
            num_keypoints=self.num_keypoints,
            heatmap_size=self.heatmap_size,
        )

        # Visibility classifier (binary: visible or not)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_keypoints),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input images, shape (B, 3, H, W).

        Returns:
            Dictionary with:
                - 'heatmaps': Predicted heatmaps, shape (B, K, Hm, Wm)
                - 'visibility': Visibility logits, shape (B, K)
                - 'keypoints': Predicted keypoint coordinates, shape (B, K, 2)
        """
        # Extract features
        features = self.backbone(x)

        # Predict heatmaps
        heatmaps = self.head(features)

        # Predict visibility
        visibility = self.visibility_head(features)

        # Extract keypoint coordinates from heatmaps
        keypoints = self._heatmaps_to_coords(heatmaps)

        return {
            "heatmaps": heatmaps,
            "visibility": visibility,
            "keypoints": keypoints,
        }

    def _heatmaps_to_coords(self, heatmaps: Tensor) -> Tensor:
        """Convert heatmaps to keypoint coordinates using soft-argmax.

        Args:
            heatmaps: Heatmaps of shape (B, K, H, W).

        Returns:
            Coordinates of shape (B, K, 2) in normalized [0, 1] range.
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device

        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, K, -1)

        # Apply softmax to get probability distribution
        probs = F.softmax(heatmaps_flat, dim=-1)

        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Flatten coordinate grids
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        # Compute expected coordinates (soft-argmax)
        x = (probs * xx_flat.view(1, 1, -1)).sum(dim=-1)
        y = (probs * yy_flat.view(1, 1, -1)).sum(dim=-1)

        # Stack to (B, K, 2)
        coords = torch.stack([x, y], dim=-1)

        return coords

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict keypoints and visibility.

        Args:
            x: Input images, shape (B, 3, H, W).

        Returns:
            Tuple of:
                - keypoints: Predicted coordinates in pixel space, shape (B, K, 2)
                - visibility: Visibility probabilities, shape (B, K)
        """
        output = self.forward(x)

        # Scale keypoints to input image size
        keypoints = output["keypoints"].clone()
        keypoints[..., 0] *= self.input_size[1]  # x * W
        keypoints[..., 1] *= self.input_size[0]  # y * H

        # Apply sigmoid to visibility logits
        visibility = torch.sigmoid(output["visibility"])

        return keypoints, visibility

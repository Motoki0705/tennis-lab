"""Court keypoint embeddings with invisible-token substitution."""

from __future__ import annotations

from math import isqrt
from typing import cast

import torch
from torch import Tensor, nn

from src.utils.models.components.norm import RMSNorm
from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding
from src.utils.models.embeddings.projection import (
    CoordinateProjection,
    apply_visibility_mask,
)


class CourtKPUVEmbedding(nn.Module):
    """Embed court keypoints with invisible-token substitution.

    Args:
        dim: Embedding dimension.
        dropout: Retained for API compatibility; ignored by the current projection stack.
        invisible_token: Shared invisible token module.
    """

    def __init__(
        self,
        *,
        dim: int,
        dropout: float = 0.0,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.proj = CoordinateProjection(input_dim=2, dim=int(dim))
        self.invisible_token = invisible_token

    def forward(self, court_kp: Tensor, court_vis: Tensor | None = None) -> Tensor:
        """Embed court keypoints.

        Args:
            court_kp: Court keypoints, shape (B, N*2) or (B, N, 2).
            court_vis: Visibility flags, shape (B, N). Optional.

        Returns:
            Tensor: Embedded tokens, shape (B, N, D).
        """
        batch_size = int(court_kp.shape[0])
        if court_kp.dim() == 2:
            court_kp = court_kp.reshape(batch_size, -1, 2)

        feat = self.proj(court_kp)
        return apply_visibility_mask(feat, court_vis, self.invisible_token)


class _LineMapConvStage(nn.Module):
    """Stride-2 depthwise-separable stage for the line-map encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.layers(inputs))


class CourtLineMapEmbedding(nn.Module):
    """Compress a court-line map into a square grid of feature tokens."""

    def __init__(
        self,
        *,
        dim: int,
        channels: tuple[int, ...] = (16, 32, 64),
        num_tokens: int = 1,
    ) -> None:
        super().__init__()
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive integers.")
        grid_size = isqrt(int(num_tokens))
        if num_tokens <= 0 or grid_size * grid_size != num_tokens:
            raise ValueError("num_tokens must be a positive perfect square.")
        self.num_tokens = int(num_tokens)
        self.grid_size = grid_size
        stages: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            stages.append(_LineMapConvStage(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*stages)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, int(dim)),
            RMSNorm(int(dim)),
            nn.GELU(),
        )

    def _pool_grid(self, features: Tensor) -> Tensor:
        """Pool to ``(B, K, C)`` using deterministic slice reductions."""
        height, width = features.shape[-2:]
        cells: list[Tensor] = []
        for row in range(self.grid_size):
            y_start = row * height // self.grid_size
            y_end = ((row + 1) * height + self.grid_size - 1) // self.grid_size
            for column in range(self.grid_size):
                x_start = column * width // self.grid_size
                x_end = (
                    (column + 1) * width + self.grid_size - 1
                ) // self.grid_size
                cells.append(
                    features[..., y_start:y_end, x_start:x_end].mean(dim=(-2, -1))
                )
        return torch.stack(cells, dim=1)

    def forward(self, court_line_map: Tensor) -> Tensor:
        """Return ``num_tokens`` row-major grid tokens for ``(..., 1, H, W)``."""
        if court_line_map.ndim < 3 or court_line_map.shape[-3] != 1:
            raise ValueError(
                "court_line_map must have shape (..., 1, H, W), got "
                f"{tuple(court_line_map.shape)}."
            )
        height, width = court_line_map.shape[-2:]
        if height < 8 or width < 8:
            raise ValueError("court_line_map height and width must be at least 8.")
        leading_shape = court_line_map.shape[:-3]
        flattened = court_line_map.reshape(-1, 1, height, width)
        tokens = self.proj(self._pool_grid(self.encoder(flattened)))
        return cast(
            Tensor,
            tokens.reshape(*leading_shape, self.num_tokens, tokens.shape[-1]),
        )

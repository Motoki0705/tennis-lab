"""Rotary Position Embedding (RoPE) implementations.

This module provides positional encoding mechanisms:
- RoPE: 1D Rotary Position Embedding
- RoPE2D: 2D Rotary Position Embedding for Vision Transformers

Reference:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - 2D RoPE in ViT: https://arxiv.org/abs/2306.15195
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RoPEConfig:
    """Configuration for Rotary Position Embedding."""

    rope_dim: int
    rope_theta: float = 10000.0


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary position embedding to query and key tensors.
    Works on tensors of shape (B, H, S, D_head).

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, cfg: RoPEConfig) -> None:
        """Initialize RoPE.

        Args:
            cfg: RoPE configuration with rope_dim and rope_theta.

        """
        super().__init__()
        assert cfg.rope_dim % 2 == 0, "rope_dim must be even"
        self.cfg = cfg

    def _build_inv_freq(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build inverse frequency tensor for RoPE."""
        half = self.cfg.rope_dim // 2
        i = torch.arange(half, device=device, dtype=dtype)
        return self.cfg.rope_theta ** (-i / half)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """Apply rotary position embedding.

        Args:
            x: Input tensor, shape (B, H, S, Dh).
            pos: Position indices, shape (S,) or (B, S).

        Returns:
            Tensor with rotary position embedding applied.

        """
        B, H, S, Dh = x.shape
        rope_dim = min(self.cfg.rope_dim, Dh)
        if rope_dim <= 0:
            return x

        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]

        device, dtype = x.device, x.dtype
        inv_freq = self._build_inv_freq(device, dtype)

        if pos.dim() == 2:
            pos_ = pos[0]
        else:
            pos_ = pos
        pos_ = pos_.to(device=device, dtype=dtype)

        angles = torch.outer(pos_, inv_freq)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        y = torch.empty_like(x_rope)
        y[..., 0::2] = y1
        y[..., 1::2] = y2

        return torch.cat([y, x_pass], dim=-1)


@dataclass
class RoPE2DConfig:
    """Configuration for 2D Rotary Position Embedding.

    Used for Vision Transformers where positions are 2D (height, width).

    Attributes:
        rope_dim: Dimension for RoPE (applied per spatial dimension).
        rope_theta: Base frequency for RoPE.
        interleave: If True, interleave h/w frequencies. If False, split dimensions.

    """

    rope_dim: int
    rope_theta: float = 10000.0
    interleave: bool = True


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for Vision Transformers.

    Applies rotary position embedding to 2D spatial positions.
    Supports both interleaved and split dimension modes.

    For a ViT with patches at (h, w) positions:
    - Interleaved: Alternates between h and w frequencies in the dim
    - Split: First half uses h, second half uses w

    Reference:
        - RoFormer: https://arxiv.org/abs/2104.09864
        - 2D RoPE in ViT: https://arxiv.org/abs/2306.15195

    """

    def __init__(self, cfg: RoPE2DConfig) -> None:
        """Initialize 2D RoPE.

        Args:
            cfg: Configuration with rope_dim, rope_theta, and interleave mode.

        """
        super().__init__()
        assert cfg.rope_dim % 4 == 0, "rope_dim must be divisible by 4 for 2D RoPE"
        self.cfg = cfg

    def _build_inv_freq(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build inverse frequency tensor for one spatial dimension."""
        # Each spatial dim uses rope_dim/2 dimensions
        half_per_dim = self.cfg.rope_dim // 4
        i = torch.arange(half_per_dim, device=device, dtype=dtype)
        return self.cfg.rope_theta ** (-i / half_per_dim)

    def _apply_rope_pairs(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply RoPE to paired dimensions.

        Args:
            x: Input tensor of shape (..., D) where D is even.
            cos: Cosine values of shape (..., D/2).
            sin: Sine values of shape (..., D/2).

        Returns:
            Rotated tensor of same shape as x.

        """
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y = torch.empty_like(x)
        y[..., 0::2] = y1
        y[..., 1::2] = y2
        return y

    def forward(
        self,
        x: Tensor,
        pos_h: Tensor,
        pos_w: Tensor,
    ) -> Tensor:
        """Apply 2D rotary position embedding.

        Args:
            x: Input tensor, shape (B, H, S, Dh).
            pos_h: Height position indices, shape (S,).
            pos_w: Width position indices, shape (S,).

        Returns:
            Tensor with 2D rotary position embedding applied.

        """
        B, H, S, Dh = x.shape
        rope_dim = min(self.cfg.rope_dim, Dh)
        if rope_dim <= 0:
            return x

        device, dtype = x.device, x.dtype
        inv_freq = self._build_inv_freq(device, dtype)
        half_rope = rope_dim // 2

        pos_h = pos_h.to(device=device, dtype=dtype)
        pos_w = pos_w.to(device=device, dtype=dtype)

        # Compute angles for h and w
        angles_h = torch.outer(pos_h, inv_freq)  # (S, half_rope/2)
        angles_w = torch.outer(pos_w, inv_freq)  # (S, half_rope/2)

        cos_h = angles_h.cos()[None, None, :, :]  # (1, 1, S, half_rope/2)
        sin_h = angles_h.sin()[None, None, :, :]
        cos_w = angles_w.cos()[None, None, :, :]
        sin_w = angles_w.sin()[None, None, :, :]

        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]

        if self.cfg.interleave:
            # Interleaved mode: alternate h and w
            # First half of rope_dim for h, second half for w
            x_h = x_rope[..., :half_rope]
            x_w = x_rope[..., half_rope:]

            y_h = self._apply_rope_pairs(x_h, cos_h, sin_h)
            y_w = self._apply_rope_pairs(x_w, cos_w, sin_w)
            y = torch.cat([y_h, y_w], dim=-1)
        else:
            # Split mode: first quarter h, second quarter w (repeated)
            quarter = rope_dim // 4
            x_h = x_rope[..., : quarter * 2]
            x_w = x_rope[..., quarter * 2 :]

            y_h = self._apply_rope_pairs(x_h, cos_h, sin_h)
            y_w = self._apply_rope_pairs(x_w, cos_w, sin_w)
            y = torch.cat([y_h, y_w], dim=-1)

        return torch.cat([y, x_pass], dim=-1)

# ==============================================================================
# NOTE ON ORIGIN / LICENSE
#
# This file is derived from (and/or inspired by) DeepSeek's inference reference:
#   https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py
#
# MIT License
#
# Copyright (c) 2025 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
# ==============================================================================
# 2D RoPE REFERENCE
#
# The 2D RoPE implementation below is adapted from (Apache-2.0):
#   https://huggingface.co/spaces/facebook/vggt/blob/0740a7a5fdbace9873ed2ce3d4d44ff2dbfa1dec/vggt/layers/rope.py
# which itself cites:
#   - https://github.com/naver-ai/rope-vit
#   - https://github.com/meta-llama/codellama/blob/main/llama/model.py
#
# The implementation here is rewritten to fit this project's interfaces, but it
# follows the same axial (y/x) split strategy and caching approach.
# ==============================================================================

"""
rope.py (pure PyTorch)

Includes:
- 1D RoPE utilities (complex cis) compatible with many LLM implementations
- 2D RoPE module for ViT-like patch grids (axial split: y/x)

Design notes:
- 1D RoPE is implemented using complex multiplication (cis).
- 2D RoPE uses cosine/sine tables and axial feature split (half for y, half for x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# -------------------------
# 1D RoPE (complex cis)
# -------------------------
@dataclass(frozen=True)
class YaRNConfig:
    """
    Optional YaRN-style frequency correction config for long context extrapolation.

    If seqlen <= original_seq_len, YaRN correction is not applied.
    """
    original_seq_len: int
    rope_factor: float
    beta_fast: int = 32
    beta_slow: int = 1


def precompute_freqs_cis(
    *,
    dim: int,
    seqlen: int,
    base: float = 10000.0,
    yarn: Optional[YaRNConfig] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precompute complex cis tensor for 1D RoPE.

    Returns:
        freqs_cis: (seqlen, dim//2) complex64/complex128 depending on torch defaults.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")

    # inv_freq: (dim/2,)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    # Optional YaRN correction for extrapolation
    if yarn is not None and seqlen > yarn.original_seq_len:
        def find_correction_dim(num_rotations: float, dim_: int, base_: float, max_seq_len: int) -> float:
            return dim_ * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2.0 * math.log(base_))

        def find_correction_range(low_rot: float, high_rot: float, dim_: int, base_: float, max_seq_len: int) -> Tuple[int, int]:
            low = math.floor(find_correction_dim(low_rot, dim_, base_, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim_, base_, max_seq_len))
            return max(low, 0), min(high, dim_ - 1)

        def linear_ramp_factor(min_: float, max_: float, dim_: int) -> torch.Tensor:
            if min_ == max_:
                max_ += 1e-3
            t = (torch.arange(dim_, dtype=torch.float32, device=device) - min_) / (max_ - min_)
            return torch.clamp(t, 0.0, 1.0)

        low, high = find_correction_range(yarn.beta_fast, yarn.beta_slow, dim, base, yarn.original_seq_len)
        smooth = 1.0 - linear_ramp_factor(low, high, dim // 2)  # (dim/2,)
        inv_freq = inv_freq / yarn.rope_factor * (1.0 - smooth) + inv_freq * smooth

    t = torch.arange(seqlen, device=device, dtype=torch.float32)  # (seqlen,)
    freqs = torch.outer(t, inv_freq)  # (seqlen, dim/2)
    # cis = cos + i sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """
    Apply 1D RoPE to the last dimension of x.

    Args:
        x: (..., rope_dim) where rope_dim is even
        freqs_cis: (T, rope_dim/2) complex; T must match x's sequence dimension.
                  This function assumes x shape is (B, T, H, D) or (B, T, D) etc,
                  i.e. the sequence dimension is x.shape[-3] for 4D, or x.shape[-2] for 3D.
        interleaved: matches DeepSeek's interleaving mode.

    Returns:
        Tensor with same shape/dtype as x.
    """
    dtype = x.dtype
    shape = x.shape

    if x.size(-1) % 2 != 0:
        raise ValueError(f"RoPE expects even dim, got {x.size(-1)}")

    # If not interleaved, convert to (.., 2, D/2) pattern used by some implementations.
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))  # (..., D/2)
    # Broadcast freqs: (1, T, 1, D/2) is the common case.
    # We detect the sequence dimension as x_complex.shape[-3] when rank>=3.
    if x_complex.dim() < 2:
        raise ValueError("Unexpected x rank for RoPE application.")

    # The sequence dimension is the second-to-last "token" axis in common layouts.
    # For (B,T,H,D/2): x_complex.dim()==4 and seq axis is 1.
    # For (B,T,D/2): x_complex.dim()==3 and seq axis is 1.
    T = x_complex.size(1)
    freqs_cis = freqs_cis.view(1, T, *([1] * (x_complex.dim() - 3)), x_complex.size(-1))
    y = torch.view_as_real(x_complex * freqs_cis).flatten(-2)  # back to (..., D)

    if not interleaved:
        # Undo the transpose convention
        y = y.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous().view(*shape)

    return y.to(dtype)


# -------------------------
# 2D RoPE (axial split)
# -------------------------
class PositionGetter:
    """
    Generates and caches 2D (y,x) integer positions for a patch grid.

    Returns positions of shape (B, H*W, 2) for a batch size B.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device)
        if key not in self._cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            pos = torch.cartesian_prod(y, x)  # (H*W, 2), columns: (y,x)
            self._cache[key] = pos
        pos = self._cache[key]
        return pos.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """
    2D RoPE module (axial split strategy).

    Expected input:
        tokens: (B, n_heads, n_tokens, rope_dim)  where rope_dim divisible by 4
        positions: (B, n_tokens, 2)  integer y/x coordinates

    It splits rope_dim into two halves:
        - first half rotated by y positions
        - second half rotated by x positions
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0) -> None:
        super().__init__()
        self.base_frequency = float(frequency)
        self.scaling_factor = float(scaling_factor)
        self._freq_cache: Dict[Tuple[int, int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_cos_sin(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds cos/sin tables of shape (seq_len, dim), where dim is even.

        We follow the referenced vggt implementation:
            inv_freq = 1 / base_frequency^(arange(0, dim, 2)/dim)
            angles = outer(positions, inv_freq)
            angles are duplicated to produce dim features
        """
        key = (dim, seq_len, device, dtype)
        if key in self._freq_cache:
            return self._freq_cache[key]

        if dim % 2 != 0:
            raise ValueError(f"2D RoPE per-axis dim must be even, got {dim}")

        exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        inv_freq = 1.0 / (self.base_frequency ** exponents)  # (dim/2,)

        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, dim/2)
        angles = angles.to(dtype)
        angles = torch.cat([angles, angles], dim=-1)  # (seq_len, dim)

        cos = angles.cos().to(dtype)
        sin = angles.sin().to(dtype)
        self._freq_cache[key] = (cos, sin)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_1d(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies 1D RoPE to tokens along a single axis.

        tokens: (B, H, T, D_axis)
        positions: (B, T) int
        cos/sin: (max_pos, D_axis)
        """
        # Embed positions with precomputed cos/sin, then broadcast across heads
        cos_t = F.embedding(positions, cos)[:, None, :, :]  # (B, 1, T, D)
        sin_t = F.embedding(positions, sin)[:, None, :, :]
        return tokens * cos_t + self._rotate_half(tokens) * sin_t

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, n_heads, n_tokens, rope_dim)
            positions: (B, n_tokens, 2) integer y/x

        Returns:
            tokens with 2D RoPE applied, same shape as input.
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens rank=4 (B,H,T,D), got {tokens.dim()}")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError(f"positions must have shape (B,T,2), got {tuple(positions.shape)}")

        rope_dim = tokens.size(-1)
        if rope_dim % 4 != 0:
            raise ValueError(f"rope_dim must be divisible by 4 for 2D RoPE, got {rope_dim}")

        # Split feature dim into y/x halves
        axis_dim = rope_dim // 2
        y_tokens, x_tokens = tokens.split(axis_dim, dim=-1)  # each (B,H,T,axis_dim)

        # cos/sin tables depend on max coordinate (height/width)
        max_pos = int(positions.max().item()) + 1
        cos, sin = self._compute_cos_sin(axis_dim, max_pos, tokens.device, tokens.dtype)

        y_pos = positions[..., 0].to(torch.long)  # (B,T)
        x_pos = positions[..., 1].to(torch.long)

        y_tokens = self._apply_1d(y_tokens, y_pos, cos, sin)
        x_tokens = self._apply_1d(x_tokens, x_pos, cos, sin)

        return torch.cat([y_tokens, x_tokens], dim=-1)

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

"""Rotary position embeddings (RoPE) utilities (pure PyTorch).

This module provides:
- 1D RoPE utilities based on complex cis (DeepSeek-style).
- 2D axial RoPE utilities (y/x split) built on the same complex cis machinery.

The 2D API is offered in two layers:
- Functional utilities: `precompute_freqs_cis_2d`, `apply_rotary_emb_2d`
- A small wrapper module: `RotaryPositionEmbedding2D` (kept for compatibility with
  existing attention code that expects a callable `nn.Module`).
"""

from __future__ import annotations

import torch
from torch import nn


# -------------------------
# 1D RoPE (complex cis)
# -------------------------
def precompute_freqs_cis(
    *,
    dim: int,
    seqlen: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Precompute complex cis tensor for 1D RoPE.

    Returns:
        freqs_cis: (seqlen, dim//2) complex64/complex128 depending on torch defaults.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    t = torch.arange(seqlen, device=device, dtype=torch.float32)  # (seqlen,)
    freqs = torch.outer(t, inv_freq)  # (seqlen, dim/2)
    # cis = cos + i sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """
    Apply 1D RoPE (complex cis) to the last dimension of `x`.

    Args:
        x: (B, T, ..., rope_dim) where rope_dim is even and sequence axis is dim=1.
        freqs_cis: (T, rope_dim/2) complex.
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
    if x_complex.dim() < 2:
        raise ValueError(f"Expected x rank>=2, got x.shape={tuple(shape)}")

    T = x_complex.size(1)
    if freqs_cis.size(0) != T:
        raise ValueError(f"freqs_cis length mismatch: freqs_cis.T={freqs_cis.size(0)} vs x.T={T}")

    freqs_cis = freqs_cis.view(1, T, *([1] * (x_complex.dim() - 3)), x_complex.size(-1))
    y = torch.view_as_real(x_complex * freqs_cis).flatten(-2)  # back to (..., D)

    if not interleaved:
        # Undo the transpose convention
        y = y.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous().view(*shape)

    return y.to(dtype)


# -------------------------
# 2D RoPE (axial split, cis-based)
# -------------------------
class PositionGetter:
    """Generate and cache 2D (y, x) integer positions for a patch grid.

    Returns:
        positions: (B, H*W, 2)
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device)
        if key not in self._cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            pos = torch.cartesian_prod(y, x)  # (H*W, 2), columns: (y, x)
            self._cache[key] = pos
        pos = self._cache[key]
        return pos.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


def _apply_rotary_emb_indexed_1d_canonical(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    *,
    interleaved: bool,
) -> torch.Tensor:
    """Apply 1D RoPE to `x` using per-token absolute positions.

    Args:
        x: (B, T, ..., D) where D is even.
        freqs_cis: (max_pos, D/2) complex cis table.
        positions: (B, T) integer absolute positions in [0, max_pos).
        interleaved: whether `x` uses interleaved (even/odd) pairs.
    """
    if x.size(-1) % 2 != 0:
        raise ValueError(f"RoPE expects even dim, got {x.size(-1)}")
    if positions.ndim != 2:
        raise ValueError(f"positions must be (B,T), got {tuple(positions.shape)}")

    dtype = x.dtype
    shape = x.shape

    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))  # (B,T,...,D/2)

    pos = positions.to(torch.long)
    if pos.min().item() < 0:
        raise ValueError("positions must be non-negative.")

    # Gather cis per token: (B,T,D/2)
    gathered = freqs_cis[pos]  # complex
    # Broadcast across extra dims (e.g., heads)
    gathered = gathered.view(shape[0], shape[1], *([1] * (x_complex.dim() - 3)), x_complex.size(-1))

    y = torch.view_as_real(x_complex * gathered).flatten(-2)  # (..., D)

    if not interleaved:
        y = y.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous().view(*shape)

    return y.to(dtype)


class RotaryPositionEmbedding2D(nn.Module):
    """2D axial RoPE wrapper (kept for attention/VisionTransformer compatibility).

    The core functionality is implemented by `apply_rotary_emb_2d`.

    Args:
        frequency: RoPE base (theta). For ViT-style 2D RoPE this is often 100.0.
        scaling_factor: kept for backward-compatible configs (currently unused).
        interleaved: whether to treat features as interleaved (even/odd) pairs.
    """

    def __init__(
        self,
        frequency: float = 100.0,
        scaling_factor: float = 1.0,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.base_frequency = float(frequency)
        self.scaling_factor = float(scaling_factor)
        self.interleaved = bool(interleaved)
        self._cache: dict[tuple[int, int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_freqs(
        self,
        *,
        dim: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (dim, height, width, device)
        if key not in self._cache:
            freqs_y, freqs_x = precompute_freqs_cis_2d(
                dim=dim,
                height=height,
                width=width,
                base=self.base_frequency,
                device=device,
            )
            self._cache[key] = (freqs_y, freqs_x)
        return self._cache[key]

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 3 or positions.size(-1) != 2:
            raise ValueError(f"positions must be (B,T,2), got {tuple(positions.shape)}")
        if positions.min().item() < 0:
            raise ValueError("positions must be non-negative.")
        if tokens.size(0) != positions.size(0):
            raise ValueError(f"Batch mismatch: tokens.B={tokens.size(0)} vs positions.B={positions.size(0)}")

        # Determine grid extents from the provided coordinates.
        y_max = int(positions[..., 0].max().item())
        x_max = int(positions[..., 1].max().item())
        height = y_max + 1
        width = x_max + 1

        freqs_y, freqs_x = self._get_freqs(dim=tokens.size(-1), height=height, width=width, device=tokens.device)
        return apply_rotary_emb_2d(
            tokens,
            freqs_y,
            freqs_x,
            positions,
            interleaved=self.interleaved,
        )


# -------------------------
# 2D precompute (axial split, cis-based)
# -------------------------
def precompute_freqs_cis_2d(
    *,
    dim: int,
    height: int,
    width: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute complex cis tables for 2D axial RoPE.

    We split dim into y/x halves:
        dim = rope_dim, must be divisible by 4
        axis_dim = dim // 2  (per-axis feature dim; must be even)
        freqs_cis_y: (height, axis_dim//2) complex
        freqs_cis_x: (width,  axis_dim//2) complex

    """
    if dim % 4 != 0:
        raise ValueError(f"2D axial RoPE requires dim % 4 == 0, got {dim}")
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be positive, got height={height}, width={width}")

    axis_dim = dim // 2  # y half and x half each has axis_dim

    freqs_cis_y = precompute_freqs_cis(dim=axis_dim, seqlen=height, base=base, device=device)
    freqs_cis_x = precompute_freqs_cis(dim=axis_dim, seqlen=width, base=base, device=device)
    return freqs_cis_y, freqs_cis_x


def apply_rotary_emb_2d(
    x: torch.Tensor,
    freqs_cis_y: torch.Tensor,
    freqs_cis_x: torch.Tensor,
    positions: torch.Tensor,
    *,
    interleaved: bool = True,
) -> torch.Tensor:
    """
    Apply 2D axial RoPE using the same "complex cis" structure as 1D.

    Supports x layouts:
      - (B, n_heads, T, D)  (common)
      - (B, T, n_heads, D)
      - (B, T, D)

    positions:
      - (B, T, 2) where last dim is (y, x)

    Requirements:
      - D % 4 == 0  (because we split into y/x halves and each half must be even for RoPE)
      - max(y) < freqs_cis_y.size(0)
      - max(x) < freqs_cis_x.size(0)
    """
    if positions.ndim != 3 or positions.size(-1) != 2:
        raise ValueError(f"positions must be (B,T,2), got {tuple(positions.shape)}")

    D = x.size(-1)
    if D % 4 != 0:
        raise ValueError(f"2D axial RoPE requires last-dim % 4 == 0, got {D}")

    B = positions.size(0)
    T = positions.size(1)

    # ---- Canonicalize x to (B, T, H, D) or (B, T, D)
    layout = None
    x_can = x

    if x.dim() == 4:
        if x.size(0) != B:
            raise ValueError(f"Batch mismatch: x.B={x.size(0)} vs positions.B={B}")

        # Detect whether x is (B, H, T, D) or (B, T, H, D)
        if x.size(1) == T:
            # (B, T, H, D)
            layout = "BTHD"
        elif x.size(2) == T:
            # (B, H, T, D) -> (B, T, H, D)
            layout = "BHTD"
            x_can = x.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Cannot infer token axis: x.shape={tuple(x.shape)} vs T={T}")

    elif x.dim() == 3:
        if x.size(0) != B or x.size(1) != T:
            raise ValueError(f"x must be (B,T,D). Got x={tuple(x.shape)}, positions={tuple(positions.shape)}")
        layout = "BTD"
    else:
        raise ValueError(f"Unsupported x rank={x.dim()}. Expected 3 or 4.")

    axis_dim = D // 2
    y_half = x_can[..., :axis_dim]
    x_half = x_can[..., axis_dim:]

    y_pos = positions[..., 0].to(torch.long)  # (B,T)
    x_pos = positions[..., 1].to(torch.long)  # (B,T)

    # Apply 1D RoPE (indexed) to each half
    y_half = _apply_rotary_emb_indexed_1d_canonical(y_half, freqs_cis_y, y_pos, interleaved=interleaved)
    x_half = _apply_rotary_emb_indexed_1d_canonical(x_half, freqs_cis_x, x_pos, interleaved=interleaved)

    out = torch.cat([y_half, x_half], dim=-1)

    # ---- Restore original layout
    if layout == "BHTD":
        out = out.permute(0, 2, 1, 3).contiguous()
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_seq_len = 4
    demo_dim = 8
    demo_freqs = precompute_freqs_cis(dim=demo_dim, seqlen=demo_seq_len, device=demo_device)
    demo_input = torch.randn(1, demo_seq_len, demo_dim, device=demo_device)

    with torch.no_grad():
        demo_output = apply_rotary_emb(demo_input, demo_freqs)

    print("1D RoPE output:")
    print(demo_output)

    demo_height = 2
    demo_width = 3
    demo_dim_2d = 8
    demo_freqs_y, demo_freqs_x = precompute_freqs_cis_2d(
        dim=demo_dim_2d,
        height=demo_height,
        width=demo_width,
        device=demo_device,
    )
    demo_positions = PositionGetter()(1, demo_height, demo_width, demo_device)
    demo_input_2d = torch.randn(1, demo_height * demo_width, demo_dim_2d, device=demo_device)

    with torch.no_grad():
        demo_output_2d = apply_rotary_emb_2d(
            demo_input_2d,
            freqs_cis_y=demo_freqs_y,
            freqs_cis_x=demo_freqs_x,
            positions=demo_positions,
        )

    print("2D RoPE output:")
    print(demo_output_2d)

"""Temporal encoder with RoPE attention."""
from __future__ import annotations

import torch
from typing import cast

from torch import Tensor, nn
from torch.nn import functional as F


class TemporalEncoderRoPE(nn.Module):
    """Stack of temporal self-attention layers with RoPE."""

    def __init__(self, dim: int, depth: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            msg = "dim must be divisible by heads"
            raise ValueError(msg)
        self.layers = nn.ModuleList([
            TemporalBlock(dim, heads) for _ in range(depth)
        ])

    def forward(self, cls_tokens: Tensor) -> Tensor:
        """Run temporal self-attention over CLS tokens."""
        x = cls_tokens
        for layer in self.layers:
            x = layer(x)
        return x


class TemporalBlock(nn.Module):
    """Transformer-style block with RoPE attention."""
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2 != 0:
            msg = "head_dim must be even for RoPE"
            raise ValueError(msg)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE attention followed by an MLP update."""
        B, T, _ = x.shape
        normed = self.norm1(x)
        qkv = self.qkv(normed)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        cos, sin = _rope_cache(T, self.head_dim, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.dim)
        proj = cast(Tensor, self.out_proj(attn))
        residual = x + proj
        ffn_out = cast(Tensor, self.ffn(self.norm2(residual)))
        return residual + ffn_out


def _rope_cache(T: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    half = head_dim // 2
    freq_seq = torch.arange(half, device=device, dtype=dtype)
    freq = 1.0 / (10000 ** (freq_seq / half))
    t = torch.arange(T, device=device, dtype=dtype)
    angles = torch.einsum("t,f->tf", t, freq)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary positional embeddings to q/k."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : cos.size(-1)], x[..., cos.size(-1) : 2 * cos.size(-1)]
    rotated = torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )
    return torch.cat([rotated, x[..., 2 * cos.size(-1) :]], dim=-1)

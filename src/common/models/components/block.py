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

"""
block.py (pure PyTorch)

Includes:
- TransformerBlock: DeepSeek-style block structure (RMSNorm + residual accumulator)
- ViTBlock: standard Vision Transformer block (LayerNorm + residual inside the block)

This module assumes single-GPU / non-distributed execution and uses the pure PyTorch
attention / norm / MoE implementations from sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from src.common.models.components.attention import MultiHeadSelfAttention, KVCache  # type: ignore
from src.common.models.components.norm import RMSNorm, LayerNorm  # type: ignore
from src.common.models.components.moe import SwiGLU, MoE, MoEArgs  # type: ignore
from src.common.models.components.rope import YaRNConfig, precompute_freqs_cis, RotaryPositionEmbedding2D, PositionGetter  # type: ignore


@dataclass
class TransformerBlockArgs:
    dim: int
    n_heads: int
    mlp_inter_dim: int
    # attention
    head_dim: Optional[int] = None
    rope_dim: Optional[int] = None
    attn_dropout: float = 0.0
    # RoPE
    rope_base: float = 10000.0
    yarn: Optional[YaRNConfig] = None
    # MoE (optional)
    use_moe: bool = False
    moe_args: Optional[MoEArgs] = None


class TransformerBlock(nn.Module):
    """
    DeepSeek-style Transformer block with a residual accumulator.

    forward returns (x, residual):
      - x is the "current" stream
      - residual is the running residual stream

    Typical usage (prefill):
        x, residual = block(x, residual=None, start_pos=0, freqs_cis=freqs, is_causal=True)
    """

    def __init__(self, args: TransformerBlockArgs) -> None:
        super().__init__()
        self.args = args

        self.attn_norm = RMSNorm(args.dim)
        self.attn = MultiHeadSelfAttention(
            dim=args.dim,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            rope_dim=args.rope_dim,
            attn_dropout=args.attn_dropout,
        )

        self.ffn_norm = RMSNorm(args.dim)
        if args.use_moe:
            if args.moe_args is None:
                raise ValueError("use_moe=True requires moe_args.")
            self.ffn: nn.Module = MoE(args.moe_args)
        else:
            self.ffn = SwiGLU(args.dim, args.mlp_inter_dim)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        *,
        start_pos: int,
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = True,
    ):
        if residual is None:
            x_norm = self.attn_norm(x)
            residual = x
        else:
            x_norm, residual = self.attn_norm(x, residual)

        x = self.attn(
            x_norm,
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            kv_cache=kv_cache,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        x_norm, residual = self.ffn_norm(x, residual)
        x = self.ffn(x_norm)
        return x, residual


@dataclass
class ViTBlockArgs:
    dim: int
    n_heads: int
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    # optional 2D RoPE
    use_2d_rope: bool = False
    rope2d_frequency: float = 100.0
    rope2d_scaling_factor: float = 1.0
    rope_dim: Optional[int] = None  # by default full head_dim


class ViTMLP(nn.Module):
    """Standard ViT MLP (GELU)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class ViTBlock(nn.Module):
    """
    Vision Transformer block (pre-norm).

    This block is meant for non-causal attention (is_causal=False).
    Optionally applies 2D RoPE to q/k using positions_2d.

    Expected input:
        x: (B, N, C)
        positions_2d: (B, N, 2) if use_2d_rope is enabled
    """

    def __init__(self, args: ViTBlockArgs) -> None:
        super().__init__()
        self.args = args

        self.norm1 = nn.LayerNorm(args.dim)
        self.attn = MultiHeadSelfAttention(
            dim=args.dim,
            n_heads=args.n_heads,
            attn_dropout=args.attn_dropout,
            rope_dim=args.rope_dim,
        )
        self.norm2 = nn.LayerNorm(args.dim)

        hidden_dim = int(args.dim * args.mlp_ratio)
        self.mlp = ViTMLP(args.dim, hidden_dim, dropout=args.mlp_dropout)

        self.rope2d: Optional[RotaryPositionEmbedding2D]
        self.pos_getter: Optional[PositionGetter]
        if args.use_2d_rope:
            self.rope2d = RotaryPositionEmbedding2D(
                frequency=args.rope2d_frequency,
                scaling_factor=args.rope2d_scaling_factor,
            )
            self.pos_getter = PositionGetter()
        else:
            self.rope2d = None
            self.pos_getter = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions_2d: Optional[torch.Tensor] = None,
        grid_hw: Optional[tuple[int, int]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            positions_2d: (B, N, 2) integer y/x coordinates for each token
            grid_hw: convenience for patch tokens: if provided and positions_2d is None,
                     positions are generated for a HxW grid and assume N == H*W.
            attn_mask: optional SDPA mask (usually None for ViT)

        Returns:
            (B, N, C)
        """
        bsz, n, _ = x.shape

        rope2d = self.rope2d
        if rope2d is not None:
            if positions_2d is None:
                if grid_hw is None or self.pos_getter is None:
                    raise ValueError("use_2d_rope=True requires positions_2d or grid_hw.")
                h, w = grid_hw
                positions_2d = self.pos_getter(bsz, h, w, x.device)
                if positions_2d.shape[1] != n:
                    raise ValueError(f"grid_hw produced {positions_2d.shape[1]} tokens, but x has N={n}")

        x = x + self.attn(
            self.norm1(x),
            start_pos=0,
            rope2d=rope2d,
            positions_2d=positions_2d,
            attn_mask=attn_mask,
            is_causal=False,
        )
        x = x + self.mlp(self.norm2(x))
        return x

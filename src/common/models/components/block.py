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

import torch
from torch import nn

from src.common.models.components.attention import (
    KVCache,
    MultiHeadSelfAttention,
)
from src.common.models.components.moe import MoE, MoEConfig, SwiGLU
from src.common.models.components.norm import RMSNorm
from src.common.models.components.rope import (
    PositionGetter,
    RotaryPositionEmbedding2D,
    YaRNConfig,
)


@dataclass
class TransformerBlockConfig:
    dim: int
    n_heads: int
    mlp_inter_dim: int
    # attention
    head_dim: int | None = None
    rope_dim: int | None = None
    attn_dropout: float = 0.0
    # RoPE
    rope_base: float = 10000.0
    yarn: YaRNConfig | None = None
    # MoE (optional)
    use_moe: bool = False
    moe_config: MoEConfig | None = None


class TransformerBlock(nn.Module):
    """
    DeepSeek-style Transformer block with a residual accumulator.

    forward returns (x, residual):
      - x is the "current" stream
      - residual is the running residual stream

    Typical usage (prefill):
        x, residual = block(x, residual=None, start_pos=0, freqs_cis=freqs, is_causal=True)
    """

    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = MultiHeadSelfAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            rope_dim=cfg.rope_dim,
            attn_dropout=cfg.attn_dropout,
        )

        self.ffn_norm = RMSNorm(cfg.dim)
        if cfg.use_moe:
            if cfg.moe_config is None:
                raise ValueError("use_moe=True requires moe_config.")
            self.ffn: nn.Module = MoE(cfg.moe_config)
        else:
            self.ffn = SwiGLU(cfg.dim, cfg.mlp_inter_dim)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        *,
        start_pos: int,
        freqs_cis: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool | None = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
class ViTBlockConfig:
    dim: int
    n_heads: int
    mlp_inter_dim: int
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    # optional 2D RoPE
    use_2d_rope: bool = False
    rope2d_frequency: float = 100.0
    rope2d_scaling_factor: float = 1.0
    rope_dim: int | None = None  # by default full head_dim
    # MoE (optional)
    use_moe: bool = False
    moe_config: MoEConfig | None = None


class ViTBlock(nn.Module):
    """
    Vision Transformer block (pre-norm).

    This block is meant for non-causal attention (is_causal=False).
    Optionally applies 2D RoPE to q/k using positions_2d.

    Expected input:
        x: (B, N, C)
        positions_2d: (B, N, 2) if use_2d_rope is enabled
    """

    def __init__(self, cfg: ViTBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadSelfAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_dim=cfg.rope_dim,
        )
        self.norm2 = nn.LayerNorm(cfg.dim)

        if cfg.use_moe:
            if cfg.moe_config is None:
                raise ValueError("use_moe=True requires moe_config.")
            self.ffn: nn.Module = MoE(cfg.moe_config)
        else:
            self.ffn = SwiGLU(cfg.dim, cfg.mlp_inter_dim)

        self.rope2d: RotaryPositionEmbedding2D | None
        self.pos_getter: PositionGetter | None
        if cfg.use_2d_rope:
            self.rope2d = RotaryPositionEmbedding2D(
                frequency=cfg.rope2d_frequency,
                scaling_factor=cfg.rope2d_scaling_factor,
            )
            self.pos_getter = PositionGetter()
        else:
            self.rope2d = None
            self.pos_getter = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions_2d: torch.Tensor | None = None,
        grid_hw: tuple[int, int] | None = None,
        attn_mask: torch.Tensor | None = None,
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
        if rope2d is not None and positions_2d is None:
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
        x = x + self.ffn(self.norm2(x))
        return x

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

from src.utils.models.components.attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from src.utils.models.components.moe import MoE, MoEConfig, SwiGLU
from src.utils.models.components.norm import RMSNorm
from src.utils.models.components.rope import (
    PositionGetter,
    YaRNConfig,
    precompute_freqs_cis_2d,
)


@dataclass
class TransformerBlockConfig:
    """Configuration for TransformerBlock.

    Args:
        dim: Token embedding dimension.
        n_heads: Number of attention heads.
        mlp_inter_dim: Hidden dimension for the MLP/FFN.
        head_dim: Per-head dimension (defaults to dim // n_heads).
        rope_dim: Rotary dimension per head for 1D RoPE.
        attn_dropout: Dropout probability for attention.
        rope_base: Base theta for 1D RoPE.
        yarn: Optional YaRN correction config.
        use_moe: Whether to use MoE FFN.
        moe_config: MoE configuration when `use_moe=True`.
    """

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
        x, residual = block(x, residual=None, freqs_cis=freqs)
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
        freqs_cis: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            x_norm = self.attn_norm(x)
            residual = x
        else:
            x_norm, residual = self.attn_norm(x, residual)

        x = self.attn(
            x_norm,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
        )

        x_norm, residual = self.ffn_norm(x, residual)
        x = self.ffn(x_norm)
        return x, residual


@dataclass
class CrossAttnBlockConfig:
    """Configuration for CrossAttnBlock."""

    dim: int
    n_heads: int
    mlp_inter_dim: int
    # attention
    head_dim: int | None = None
    rope_dim: int | None = None
    attn_dropout: float = 0.0
    # MoE (optional)
    use_moe: bool = False
    moe_config: MoEConfig | None = None


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block with optional RoPE on query/key."""

    def __init__(self, cfg: CrossAttnBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.q_norm = RMSNorm(cfg.dim)
        self.kv_norm = RMSNorm(cfg.dim)
        self.attn = MultiHeadCrossAttention(
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
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        key_valid: torch.Tensor | None = None,
        freqs_q_cis: torch.Tensor | None = None,
        freqs_k_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = q.shape
        _, k_len, _ = kv.shape

        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(kv)

        attn_mask: torch.Tensor | None = None
        if key_valid is not None:
            if key_valid.shape != (bsz, k_len):
                raise ValueError(
                    f"key_valid must have shape {(bsz, k_len)}, got {tuple(key_valid.shape)}"
                )
            key_keep = key_valid > 0
            fully_masked = ~key_keep.any(dim=1)
            if fully_masked.any():
                key_keep = key_keep.clone()
                key_keep[fully_masked, 0] = True
                kv_norm = kv_norm.clone()
                kv_norm[fully_masked] = 0.0
            attn_mask = key_keep[:, None, :].expand(bsz, q_len, k_len)

        q = q + self.attn(
            q_norm,
            kv_norm,
            freqs_q_cis=freqs_q_cis,
            freqs_k_cis=freqs_k_cis,
            attn_mask=attn_mask,
        )
        q = q + self.ffn(self.ffn_norm(q))
        return q

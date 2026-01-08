"""Transformer block implementations."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from src.common.models.attention import GQASelfAttention, RoPE
from src.common.models.mlp import SwiGLUMLP
from src.common.models.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with GQA attention and SwiGLU MLP.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLUMLP(RMSNorm(x))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float,
        rope: Optional[RoPE],
        causal: bool,
    ) -> None:
        """Initialize Transformer block.

        Args:
            dim: Model dimension.
            num_heads: Number of query heads.
            num_kv_heads: Number of key/value heads (for GQA).
            ffn_dim: FFN intermediate dimension.
            dropout: Dropout probability.
            rope: Optional RoPE module for positional encoding.
            causal: Whether to use causal attention mask.

        """
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = GQASelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rope=rope,
            causal=causal,
        )
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim=dim, ffn_dim=ffn_dim, dropout=dropout)

    def forward(
        self, x: Tensor, pos: Tensor, key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, S, D).
            pos: Position indices for RoPE, shape (S,).
            key_padding_mask: Mask where True = keep, False = mask out, shape (B, S).

        Returns:
            Output tensor, shape (B, S, D).

        """
        x = x + self.attn(self.attn_norm(x), pos=pos, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

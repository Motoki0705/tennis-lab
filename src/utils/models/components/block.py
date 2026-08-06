from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn

from src.utils.models.components.attention import (
    GroupedQuerySelfAttention,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from src.utils.models.components.ffn_layers import MLP, SwiGLU
from src.utils.models.components.norm import RMSNorm


@dataclass
class TransformerBlockConfig:
    """Configuration for TransformerBlock.

    Args:
        dim: Token embedding dimension.
        n_heads: Number of attention heads.
        ffn_dim: Hidden dimension for the FFN. Defaults to the repository-wide transformer FFN heuristic.
        head_dim: Per-head dimension (defaults to dim // n_heads).
        rope_dim: Rotary dimension per head for 1D RoPE.
        attn_dropout: Dropout probability for attention.
        attention_type: Self-attention implementation to use.
        n_kv_heads: Number of key/value heads for GQA.
        rope_base: Base theta for 1D RoPE.
        ffn_type: FFN implementation to use.
    """

    dim: int
    n_heads: int
    ffn_dim: int
    # attention
    head_dim: int
    rope_dim: int
    attn_dropout: float
    attention_type: Literal["mha", "gqa"]
    n_kv_heads: int | None
    # RoPE
    rope_base: float
    # FFN
    ffn_type: Literal["swiglu", "mlp"]


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with explicit residual additions.
    """

    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.attn_norm = RMSNorm(cfg.dim)
        self.attn: MultiHeadSelfAttention | GroupedQuerySelfAttention
        if cfg.attention_type == "mha":
            self.attn = MultiHeadSelfAttention(
                dim=cfg.dim,
                n_heads=cfg.n_heads,
                head_dim=cfg.head_dim,
                rope_dim=cfg.rope_dim,
                attn_dropout=cfg.attn_dropout,
                bias=False,
            )
        elif cfg.attention_type == "gqa":
            if cfg.n_kv_heads is None:
                raise ValueError("n_kv_heads must be set when attention_type='gqa'")
            self.attn = GroupedQuerySelfAttention(
                dim=cfg.dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_dim=cfg.rope_dim,
                attn_dropout=cfg.attn_dropout,
                bias=False,
            )
        else:
            raise ValueError(f"Unsupported attention_type={cfg.attention_type}")

        self.ffn_norm = RMSNorm(cfg.dim)
        if cfg.ffn_type == "swiglu":
            self.ffn: nn.Module = SwiGLU(cfg.dim, cfg.ffn_dim)
        elif cfg.ffn_type == "mlp":
            self.ffn = MLP(cfg.dim, cfg.ffn_dim)
        else:
            raise ValueError(f"Unsupported ffn_type={cfg.ffn_type}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_attn = x + self.attn(
            self.attn_norm(x),
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
        )
        ffn_output = self.ffn(self.ffn_norm(x_attn))
        x_fnn = cast(torch.Tensor, x_attn + ffn_output)
        return x_fnn


@dataclass
class CrossAttnBlockConfig:
    """Configuration for CrossAttnBlock."""

    dim: int
    n_heads: int
    ffn_dim: int
    # attention
    head_dim: int
    rope_dim: int
    attn_dropout: float
    # FFN
    ffn_type: Literal["swiglu", "mlp"]


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block over boundary-prepared tensors."""

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
            bias=False,
        )
        self.ffn_norm = RMSNorm(cfg.dim)
        if cfg.ffn_type == "swiglu":
            self.ffn: nn.Module = SwiGLU(cfg.dim, cfg.ffn_dim)
        elif cfg.ffn_type == "mlp":
            self.ffn = MLP(cfg.dim, cfg.ffn_dim)
        else:
            raise ValueError(f"Unsupported ffn_type={cfg.ffn_type}")

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        freqs_q_cis: torch.Tensor,
        freqs_k_cis: torch.Tensor,
    ) -> torch.Tensor:
        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(kv)

        q = q + self.attn(
            q_norm,
            kv_norm,
            freqs_q_cis=freqs_q_cis,
            freqs_k_cis=freqs_k_cis,
            attn_mask=attn_mask,
        )
        q = q + self.ffn(self.ffn_norm(q))
        return q

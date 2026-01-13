"""Transformer block implementations.

This module provides Transformer blocks for various architectures:
- TransformerBlock: Configurable block with selectable attention (MHA/GQA/MLA)
- ViTBlock: Vision Transformer block with 2D RoPE support

All blocks support attention type selection via configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models.components.attention import (
    GQA,
    GQAConfig,
    MHA,
    MHAConfig,
    MLA,
    MLAConfig,
    AttentionType,
    build_attention,
)
from src.common.models.components.mlp import MoEConfig, MoELayer, SwiGLUMLP
from src.common.models.components.norm import RMSNorm
from src.common.models.components.rope import RoPE, RoPE2D, RoPEConfig


@dataclass
class BlockConfig:
    """Configuration for Transformer block.

    Attributes:
        dim: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads (for GQA). Ignored for MHA.
        ffn_dim: FFN intermediate dimension.
        dropout: Dropout probability.
        attn_type: Attention type ('mha', 'gqa', 'mla').
        rope_dim: RoPE dimension. None to disable.
        rope_theta: RoPE base frequency.
        causal: Whether to use causal masking.
        use_moe: Whether to use MoE for FFN.
        moe_config: MoE configuration (if use_moe=True).
        # MLA-specific
        kv_lora_rank: KV compression rank (MLA only).
        q_lora_rank: Q compression rank (MLA only).

    """

    dim: int
    num_heads: int
    num_kv_heads: int = 4
    ffn_dim: int | None = None
    dropout: float = 0.0
    attn_type: Literal["mha", "gqa", "mla"] = "gqa"
    rope_dim: int | None = None
    rope_theta: float = 10000.0
    causal: bool = False
    use_moe: bool = False
    moe_config: MoEConfig | None = None
    # MLA-specific
    kv_lora_rank: int = 64
    q_lora_rank: int | None = None


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with configurable attention.

    Supports MHA, GQA, and MLA attention mechanisms.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))  # FFN can be SwiGLU or MoE
    """

    def __init__(self, cfg: BlockConfig) -> None:
        """Initialize Transformer block.

        Args:
            cfg: Block configuration.

        """
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim

        # FFN dimension
        ffn_dim = cfg.ffn_dim
        if ffn_dim is None:
            ffn_dim = int((8 * dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        # Attention
        self.attn_norm = RMSNorm(dim)
        self.attn = build_attention(
            attn_type=cfg.attn_type,
            dim=dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            dropout=cfg.dropout,
            rope_dim=cfg.rope_dim,
            rope_theta=cfg.rope_theta,
            causal=cfg.causal,
            kv_lora_rank=cfg.kv_lora_rank,
            q_lora_rank=cfg.q_lora_rank,
        )

        # FFN (or MoE)
        self.mlp_norm = RMSNorm(dim)
        self.use_moe = cfg.use_moe

        if cfg.use_moe and cfg.moe_config is not None:
            self.mlp = MoELayer(cfg.moe_config)
        else:
            self.mlp = SwiGLUMLP(dim=dim, ffn_dim=ffn_dim, dropout=cfg.dropout)

    def forward(
        self,
        x: Tensor,
        pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, S, D).
            pos: Position indices for RoPE, shape (S,).
            key_padding_mask: Mask where True = keep, False = mask out.

        Returns:
            Output tensor, shape (B, S, D).

        """
        x = x + self.attn(self.attn_norm(x), pos=pos, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def get_aux_loss(self) -> Tensor:
        """Get auxiliary loss from MoE layer if applicable."""
        if self.use_moe and hasattr(self.mlp, "get_aux_loss"):
            return self.mlp.get_aux_loss()
        return torch.tensor(0.0)


@dataclass
class ViTBlockConfig:
    """Configuration for Vision Transformer block.

    Attributes:
        dim: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads (for GQA).
        ffn_dim: FFN intermediate dimension.
        dropout: Dropout probability.
        attn_type: Attention type ('mha', 'gqa', 'mla').
        rope_dim: 2D RoPE dimension. None to disable.
        rope_theta: RoPE base frequency.
        rope_interleave: Whether to interleave h/w in RoPE.
        use_moe: Whether to use MoE for FFN.
        moe_config: MoE configuration (if use_moe=True).
        # MLA-specific
        kv_lora_rank: KV compression rank (MLA only).
        q_lora_rank: Q compression rank (MLA only).

    """

    dim: int
    num_heads: int
    num_kv_heads: int = 4
    ffn_dim: int | None = None
    dropout: float = 0.0
    attn_type: Literal["mha", "gqa", "mla"] = "gqa"
    rope_dim: int | None = None
    rope_theta: float = 10000.0
    rope_interleave: bool = True
    use_moe: bool = False
    moe_config: MoEConfig | None = None
    # MLA-specific
    kv_lora_rank: int = 64
    q_lora_rank: int | None = None


class ViTBlock(nn.Module):
    """Vision Transformer block with 2D RoPE and configurable attention.

    Supports MHA, GQA, and MLA attention with 2D positional encoding.

    Architecture (Pre-Norm):
        x = x + Attention(RMSNorm(x), pos_h, pos_w)
        x = x + FFN(RMSNorm(x))  # FFN can be SwiGLU or MoE

    """

    def __init__(self, cfg: ViTBlockConfig) -> None:
        """Initialize ViT block.

        Args:
            cfg: ViT block configuration.

        """
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim

        # FFN dimension
        ffn_dim = cfg.ffn_dim
        if ffn_dim is None:
            ffn_dim = int((8 * dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        # 2D RoPE (for manual application)
        self.rope_2d: RoPE2D | None = None
        if cfg.rope_dim is not None and cfg.rope_dim > 0:
            from src.common.models.components.rope import RoPE2D, RoPE2DConfig

            self.rope_2d = RoPE2D(
                RoPE2DConfig(
                    rope_dim=cfg.rope_dim,
                    rope_theta=cfg.rope_theta,
                    interleave=cfg.rope_interleave,
                )
            )

        # Attention (without RoPE - we apply 2D RoPE manually)
        self.attn_norm = RMSNorm(dim)
        self.attn_type = AttentionType(cfg.attn_type)

        self.attn = build_attention(
            attn_type=cfg.attn_type,
            dim=dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            dropout=cfg.dropout,
            rope_dim=None,  # Handle RoPE manually for 2D
            rope_theta=cfg.rope_theta,
            causal=False,
            kv_lora_rank=cfg.kv_lora_rank,
            q_lora_rank=cfg.q_lora_rank,
        )

        # Store for access
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = dim // cfg.num_heads

        # FFN (or MoE)
        self.mlp_norm = RMSNorm(dim)
        self.use_moe = cfg.use_moe

        if cfg.use_moe and cfg.moe_config is not None:
            self.mlp = MoELayer(cfg.moe_config)
        else:
            self.mlp = SwiGLUMLP(dim=dim, ffn_dim=ffn_dim, dropout=cfg.dropout)

    def forward(
        self,
        x: Tensor,
        pos_h: Optional[Tensor] = None,
        pos_w: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, S, D).
            pos_h: Height position indices, shape (S,).
            pos_w: Width position indices, shape (S,).
            key_padding_mask: Optional attention mask.

        Returns:
            Output tensor, shape (B, S, D).

        """
        attn_in = self.attn_norm(x)

        # For MLA, create combined 1D position
        if self.attn_type == AttentionType.MLA:
            pos = None
            if pos_h is not None and pos_w is not None:
                pos = pos_h * 10000 + pos_w
            attn_out = self.attn(attn_in, pos=pos, key_padding_mask=key_padding_mask)
        elif self.rope_2d is not None and pos_h is not None and pos_w is not None:
            # Manual 2D RoPE for GQA/MHA
            attn_out = self._forward_with_2d_rope(
                attn_in, pos_h, pos_w, key_padding_mask
            )
        else:
            attn_out = self.attn(attn_in, pos=None, key_padding_mask=key_padding_mask)

        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def _forward_with_2d_rope(
        self,
        x: Tensor,
        pos_h: Tensor,
        pos_w: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Forward pass with manual 2D RoPE application.

        This method manually applies 2D RoPE to Q/K before attention.

        Args:
            x: Normalized input tensor, shape (B, S, D).
            pos_h: Height positions, shape (S,).
            pos_w: Width positions, shape (S,).
            key_padding_mask: Optional attention mask.

        Returns:
            Attention output, shape (B, S, D).

        """
        import torch.nn.functional as F

        B, S, D = x.shape

        # Access internal projections
        attn = self.attn
        if hasattr(attn, "attn"):
            # For MHA wrapper
            attn = attn.attn

        # Project Q, K, V
        q = attn.wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.wk(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.wv(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply 2D RoPE
        assert self.rope_2d is not None
        q = self.rope_2d(q, pos_h, pos_w)

        # Expand K for RoPE then average back
        num_groups = self.num_heads // self.num_kv_heads
        k_expanded = (
            k.unsqueeze(2)
            .expand(B, self.num_kv_heads, num_groups, S, self.head_dim)
            .reshape(B, self.num_heads, S, self.head_dim)
        )
        k_expanded = self.rope_2d(k_expanded, pos_h, pos_w)
        k = k_expanded.view(B, self.num_kv_heads, num_groups, S, self.head_dim)[
            :, :, 0, :, :
        ]

        # Expand K/V for GQA
        k = (
            k.unsqueeze(2)
            .expand(B, self.num_kv_heads, num_groups, S, self.head_dim)
            .reshape(B, self.num_heads, S, self.head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(B, self.num_kv_heads, num_groups, S, self.head_dim)
            .reshape(B, self.num_heads, S, self.head_dim)
        )

        # Build attention mask
        attn_mask = None
        if key_padding_mask is not None:
            mask = ~key_padding_mask
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(
                mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # SDPA
        dropout_p = attn.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return attn.wo(out)

    def get_aux_loss(self) -> Tensor:
        """Get auxiliary loss from MoE if applicable."""
        if self.use_moe and hasattr(self.mlp, "get_aux_loss"):
            return self.mlp.get_aux_loss()
        return torch.tensor(0.0)

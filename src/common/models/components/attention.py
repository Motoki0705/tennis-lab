"""Attention mechanisms for Transformer models.

This module provides attention mechanisms including:
- MHA: Multi-Head Attention (standard)
- GQA: Grouped-Query Attention
- MLA: Multi-head Latent Attention (DeepSeek-V2/V3)

All attention classes support a common interface for easy switching.

Reference:
    - MHA: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
    - GQA: https://arxiv.org/abs/2305.13245 (GQA: Training Generalized Multi-Query)
    - MLA: https://arxiv.org/abs/2405.04434 (DeepSeek-V2)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.common.models.components.rope import RoPE


class AttentionType(str, Enum):
    """Attention mechanism type."""

    MHA = "mha"
    GQA = "gqa"
    MLA = "mla"


@dataclass
class GQAConfig:
    """Configuration for Grouped-Query Attention.

    Attributes:
        dim: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (must divide num_heads).
        dropout: Dropout probability.
        rope_dim: Dimension for RoPE. None to disable.
        rope_theta: Base frequency for RoPE.
        causal: Whether to use causal attention.

    """

    dim: int
    num_heads: int
    num_kv_heads: int = 4
    dropout: float = 0.0
    rope_dim: int | None = None
    rope_theta: float = 10000.0
    causal: bool = False


class GQA(nn.Module):
    """Grouped-Query Attention with Scaled Dot-Product Attention (SDPA).

    Uses F.scaled_dot_product_attention for efficiency:
      - Q has num_heads query heads
      - K/V have num_kv_heads key/value heads (GQA)
      - K/V are expanded via view/expand (no memory copy) to match num_heads

    Special cases:
      - num_kv_heads == num_heads: Standard MHA
      - num_kv_heads == 1: Multi-Query Attention (MQA)

    Reference: https://arxiv.org/abs/2305.13245
    """

    def __init__(self, cfg: GQAConfig) -> None:
        """Initialize GQA.

        Args:
            cfg: GQA configuration.

        """
        super().__init__()
        dim = cfg.dim
        num_heads = cfg.num_heads
        num_kv_heads = cfg.num_kv_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert (
            num_heads % num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = cfg.dropout
        self.causal = cfg.causal

        # RoPE (optional)
        self.rope: RoPE | None = None
        if cfg.rope_dim is not None and cfg.rope_dim > 0:
            from src.common.models.components.rope import RoPE, RoPEConfig

            self.rope = RoPE(RoPEConfig(rope_dim=cfg.rope_dim, rope_theta=cfg.rope_theta))

        # Projections
        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with SDPA.

        Args:
            x: Input tensor, shape (B, S, D).
            pos: Position indices for RoPE, shape (S,).
            key_padding_mask: Mask where True = keep, False = mask out, shape (B, S).

        Returns:
            Output tensor, shape (B, S, D).

        """
        B, S, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # GQA: expand K/V to match num_heads using view/expand (no memory copy)
        k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.num_groups, S, self.head_dim)
        k = k.reshape(B, self.num_heads, S, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.num_groups, S, self.head_dim)
        v = v.reshape(B, self.num_heads, S, self.head_dim)

        # Build attention mask for SDPA
        attn_mask: Optional[Tensor] = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) True=keep, False=mask
            # SDPA expects additive mask where -inf = masked
            mask = ~key_padding_mask
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Use SDPA for efficient attention computation
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.causal and attn_mask is None,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.wo(out)


@dataclass
class MHAConfig:
    """Configuration for Multi-Head Attention.

    Attributes:
        dim: Model dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        rope_dim: Dimension for RoPE. None to disable.
        rope_theta: Base frequency for RoPE.
        causal: Whether to use causal attention.

    """

    dim: int
    num_heads: int
    dropout: float = 0.0
    rope_dim: int | None = None
    rope_theta: float = 10000.0
    causal: bool = False


class MHA(nn.Module):
    """Multi-Head Attention (standard implementation).

    This is equivalent to GQA with num_kv_heads == num_heads.
    Provided for clarity and backward compatibility.

    Reference: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, cfg: MHAConfig) -> None:
        """Initialize MHA.

        Args:
            cfg: MHA configuration.

        """
        super().__init__()
        # MHA is GQA with num_kv_heads == num_heads
        gqa_cfg = GQAConfig(
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_heads,  # Full attention
            dropout=cfg.dropout,
            rope_dim=cfg.rope_dim,
            rope_theta=cfg.rope_theta,
            causal=cfg.causal,
        )
        self.attn = GQA(gqa_cfg)

        # Expose attributes for compatibility
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.dim // cfg.num_heads
        self.dropout = cfg.dropout
        self.causal = cfg.causal

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
        return self.attn(x, pos=pos, key_padding_mask=key_padding_mask)


@dataclass
class MLAConfig:
    """Configuration for Multi-head Latent Attention.

    Attributes:
        dim: Model dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head. Defaults to dim // num_heads.
        kv_lora_rank: Rank for KV compression. Lower = more compression.
        q_lora_rank: Rank for Q compression (optional). None = no compression.
        rope_dim: Dimension for RoPE. Applied to decoupled position keys.
        rope_theta: Base frequency for RoPE.
        dropout: Dropout probability.
        causal: Whether to use causal attention.

    """

    dim: int
    num_heads: int
    head_dim: int | None = None
    kv_lora_rank: int = 64
    q_lora_rank: int | None = None
    rope_dim: int = 64
    rope_theta: float = 10000.0
    dropout: float = 0.0
    causal: bool = False


class MLA(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2/V3).

    Key innovations:
    1. KV cache compression: Instead of caching full K/V, we cache a
       compressed latent vector c_kv which is projected back to K/V.
    2. Decoupled RoPE: Position information is kept in separate "rope keys"
       that don't go through the compression, ensuring positional info
       is preserved.
    3. Optional Q compression for further efficiency.

    The attention computation:
        c_kv = W_dkv(x)  # Compress to latent
        k_c, v_c = split(W_ukv(c_kv))  # Decompress content K/V
        k_pe = W_kpe(x)  # Decoupled position key
        k = [k_c; k_pe]  # Concatenate content and position keys

    Memory savings:
        - Standard MHA: O(S * D)
        - GQA: O(S * D / G) where G is group size
        - MLA: O(S * r) where r << D is the latent rank

    Reference: https://arxiv.org/abs/2405.04434
    """

    def __init__(self, cfg: MLAConfig) -> None:
        """Initialize MLA.

        Args:
            cfg: MLA configuration.

        """
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim or (cfg.dim // cfg.num_heads)
        self.kv_lora_rank = cfg.kv_lora_rank
        self.q_lora_rank = cfg.q_lora_rank
        self.rope_dim = cfg.rope_dim
        self.dropout = cfg.dropout
        self.causal = cfg.causal

        # Total dimension for content K/V per head
        self.kv_head_dim = self.head_dim

        # Q projection (optionally with LoRA compression)
        if self.q_lora_rank is not None:
            self.q_down = nn.Linear(cfg.dim, self.q_lora_rank, bias=False)
            self.q_up = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.head_dim,
                bias=False,
            )
        else:
            self.wq = nn.Linear(cfg.dim, self.num_heads * self.head_dim, bias=False)

        # KV compression: down-project to latent, up-project to K and V
        self.kv_down = nn.Linear(cfg.dim, self.kv_lora_rank, bias=False)
        self.kv_up = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.kv_head_dim + self.kv_head_dim),
            bias=False,
        )

        # Decoupled position key (separate from content)
        self.k_pe = nn.Linear(cfg.dim, self.num_heads * self.rope_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(self.num_heads * self.head_dim, cfg.dim, bias=False)

        # RoPE inverse frequencies (precomputed)
        self._init_rope()

    def _init_rope(self) -> None:
        """Initialize RoPE inverse frequencies."""
        half = self.rope_dim // 2
        inv_freq = self.cfg.rope_theta ** (
            -torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: Tensor, pos: Tensor) -> Tensor:
        """Apply RoPE to the position key dimensions.

        Args:
            x: Tensor of shape (B, H, S, rope_dim).
            pos: Position indices of shape (S,).

        Returns:
            Rotated tensor of same shape.

        """
        B, H, S, D = x.shape
        device, dtype = x.device, x.dtype

        inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        pos = pos.to(device=device, dtype=dtype)

        angles = torch.outer(pos, inv_freq)  # (S, rope_dim/2)
        cos = angles.cos()[None, None, :, :]  # (1, 1, S, rope_dim/2)
        sin = angles.sin()[None, None, :, :]

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
        pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for MLA.

        Args:
            x: Input tensor of shape (B, S, D).
            pos: Position indices of shape (S,). Required for RoPE.
            key_padding_mask: Boolean mask where True = keep, False = mask.

        Returns:
            Output tensor of shape (B, S, D).

        """
        B, S, _ = x.shape

        # Compute Q
        if self.q_lora_rank is not None:
            q = self.q_up(self.q_down(x))
        else:
            q = self.wq(x)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute compressed KV
        c_kv = self.kv_down(x)
        kv = self.kv_up(c_kv)
        kv = kv.view(B, S, self.num_heads, 2 * self.kv_head_dim)

        # Split into K_content and V
        k_c, v = kv.split([self.kv_head_dim, self.kv_head_dim], dim=-1)
        k_c = k_c.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute decoupled position key
        k_pe = self.k_pe(x)
        k_pe = k_pe.view(B, S, self.num_heads, self.rope_dim).transpose(1, 2)

        # Apply RoPE to Q and position key
        if pos is not None:
            q_rope = q[..., : self.rope_dim]
            q_pass = q[..., self.rope_dim :]
            q_rope = self._apply_rope(q_rope, pos)
            q = torch.cat([q_rope, q_pass], dim=-1)
            k_pe = self._apply_rope(k_pe, pos)

        # Attention with content + position keys
        scale = (self.head_dim) ** -0.5

        # Build attention mask
        attn_mask: Optional[Tensor] = None
        if key_padding_mask is not None:
            mask = ~key_padding_mask
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Compute attention scores
        attn_content = torch.matmul(q, k_c.transpose(-2, -1)) * scale
        q_for_pe = q[..., : self.rope_dim]
        attn_pos = torch.matmul(q_for_pe, k_pe.transpose(-2, -1)) * (self.rope_dim**-0.5)
        attn = attn_content + attn_pos

        if attn_mask is not None:
            attn = attn + attn_mask

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)


def build_attention(
    attn_type: AttentionType | str,
    dim: int,
    num_heads: int,
    num_kv_heads: int | None = None,
    dropout: float = 0.0,
    rope_dim: int | None = None,
    rope_theta: float = 10000.0,
    causal: bool = False,
    # MLA-specific
    kv_lora_rank: int = 64,
    q_lora_rank: int | None = None,
) -> nn.Module:
    """Factory function to build attention module.

    Args:
        attn_type: Type of attention ('mha', 'gqa', 'mla').
        dim: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads (for GQA). Defaults to num_heads.
        dropout: Dropout probability.
        rope_dim: RoPE dimension. None to disable.
        rope_theta: RoPE base frequency.
        causal: Whether to use causal masking.
        kv_lora_rank: KV compression rank (MLA only).
        q_lora_rank: Q compression rank (MLA only).

    Returns:
        Attention module.

    """
    attn_type = AttentionType(attn_type) if isinstance(attn_type, str) else attn_type

    if attn_type == AttentionType.MHA:
        return MHA(
            MHAConfig(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                rope_dim=rope_dim,
                rope_theta=rope_theta,
                causal=causal,
            )
        )
    elif attn_type == AttentionType.GQA:
        return GQA(
            GQAConfig(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads or num_heads,
                dropout=dropout,
                rope_dim=rope_dim,
                rope_theta=rope_theta,
                causal=causal,
            )
        )
    elif attn_type == AttentionType.MLA:
        return MLA(
            MLAConfig(
                dim=dim,
                num_heads=num_heads,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                rope_dim=rope_dim or 64,
                rope_theta=rope_theta,
                dropout=dropout,
                causal=causal,
            )
        )
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

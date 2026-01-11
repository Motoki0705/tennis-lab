"""Attention mechanisms for Transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RoPEConfig:
    """Configuration for Rotary Position Embedding."""

    rope_dim: int
    rope_theta: float = 10000.0


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary position embedding to query and key tensors.
    Works on tensors of shape (B, H, S, D_head).

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, cfg: RoPEConfig) -> None:
        """Initialize RoPE.

        Args:
            cfg: RoPE configuration with rope_dim and rope_theta.

        """
        super().__init__()
        assert cfg.rope_dim % 2 == 0, "rope_dim must be even"
        self.cfg = cfg

    def _build_inv_freq(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build inverse frequency tensor for RoPE."""
        half = self.cfg.rope_dim // 2
        i = torch.arange(half, device=device, dtype=dtype)
        return self.cfg.rope_theta ** (-i / half)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """Apply rotary position embedding.

        Args:
            x: Input tensor, shape (B, H, S, Dh).
            pos: Position indices, shape (S,) or (B, S).

        Returns:
            Tensor with rotary position embedding applied.

        """
        B, H, S, Dh = x.shape
        rope_dim = min(self.cfg.rope_dim, Dh)
        if rope_dim <= 0:
            return x

        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]

        device, dtype = x.device, x.dtype
        inv_freq = self._build_inv_freq(device, dtype)

        if pos.dim() == 2:
            pos_ = pos[0]
        else:
            pos_ = pos
        pos_ = pos_.to(device=device, dtype=dtype)

        angles = torch.outer(pos_, inv_freq)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        y = torch.empty_like(x_rope)
        y[..., 0::2] = y1
        y[..., 1::2] = y2

        return torch.cat([y, x_pass], dim=-1)


class GQASelfAttention(nn.Module):
    """Grouped-Query Self-Attention with Scaled Dot-Product Attention (SDPA).

    Uses F.scaled_dot_product_attention for efficiency:
      - Q has num_heads query heads
      - K/V have num_kv_heads key/value heads (GQA)
      - K/V are expanded via view/expand (no memory copy) to match num_heads

    Reference: https://arxiv.org/abs/2305.13245
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float,
        rope: Optional[RoPE] = None,
        causal: bool = False,
    ) -> None:
        """Initialize GQA Self-Attention.

        Args:
            dim: Model dimension.
            num_heads: Number of query heads.
            num_kv_heads: Number of key/value heads (must divide num_heads).
            dropout: Dropout probability for attention weights.
            rope: Optional RoPE module for positional encoding.
            causal: Whether to use causal attention mask.

        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.rope = rope
        self.causal = causal

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

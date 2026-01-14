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
attention.py (pure PyTorch)

- Implements Multi-Head Self-Attention (MHA) using torch.nn.functional.scaled_dot_product_attention.
- Supports:
  - Optional KV-cache (for autoregressive decoding)
  - Optional 1D RoPE (complex cis) via rope.apply_rotary_emb
  - Optional 2D RoPE (for ViT-like patch grids) via rope.RotaryPositionEmbedding2D

Mask conventions (important):
- torch.nn.functional.scaled_dot_product_attention supports:
  - float mask: added to attention scores (use 0 for keep, -inf for mask)
  - bool mask: True means "take part in attention" (KEEP), False means MASK
  See official docs for SDPA for details.
- To avoid ambiguity with nn.MultiheadAttention (where bool True often means MASK),
  this module primarily uses float additive masks internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.common.models.components.rope import apply_rotary_emb, RotaryPositionEmbedding2D


def _make_additive_causal_mask(
    q_len: int,
    k_len: int,
    *,
    start_pos: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Builds an additive causal mask for absolute-positioned queries.

    Allowed attention: key_pos <= query_pos_abs
      - query_pos_abs spans [start_pos, start_pos + q_len - 1]
      - key_pos spans   [0, k_len - 1]

    Returns:
        Tensor of shape (q_len, k_len) with 0 for allowed and -inf for disallowed.
    """
    q_pos = torch.arange(start_pos, start_pos + q_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    disallow = k_pos[None, :] > q_pos[:, None]  # (q_len, k_len) bool
    mask = torch.zeros((q_len, k_len), device=device, dtype=dtype)
    mask = mask.masked_fill(disallow, torch.finfo(dtype).min)
    return mask


def _normalize_attn_mask(
    attn_mask: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Normalizes a user-provided mask into a float additive mask broadcastable to (B, H, q_len, k_len).

    Accepts:
      - (q_len, k_len)
      - (B, q_len, k_len)
      - (B, 1, q_len, k_len)
      - (B, H, q_len, k_len)

    For bool masks: SDPA semantics are used (True=KEEP, False=MASK).
    """
    if attn_mask.device != device:
        attn_mask = attn_mask.to(device)

    if attn_mask.dtype == torch.bool:
        # SDPA expects True=KEEP; convert to additive float.
        keep = attn_mask
        add = torch.zeros_like(keep, dtype=dtype)
        add = add.masked_fill(~keep, torch.finfo(dtype).min)
        attn_mask = add
    else:
        attn_mask = attn_mask.to(dtype)

    if attn_mask.dim() == 2:
        if attn_mask.shape != (q_len, k_len):
            raise ValueError(f"attn_mask shape must be {(q_len, k_len)}, got {tuple(attn_mask.shape)}")
        attn_mask = attn_mask[None, None, :, :]  # (1,1,q,k)
    elif attn_mask.dim() == 3:
        if attn_mask.shape[1:] != (q_len, k_len):
            raise ValueError(f"attn_mask shape must be (B,{q_len},{k_len}), got {tuple(attn_mask.shape)}")
        attn_mask = attn_mask[:, None, :, :]  # (B,1,q,k)
    elif attn_mask.dim() == 4:
        if attn_mask.shape[-2:] != (q_len, k_len):
            raise ValueError(f"attn_mask last dims must be ({q_len},{k_len}), got {tuple(attn_mask.shape)}")
        # keep as-is; should be broadcastable to (B,H,q,k)
    else:
        raise ValueError(f"Unsupported attn_mask rank: {attn_mask.dim()}")

    return attn_mask


@dataclass
class KVCache:
    """
    Simple KV cache for autoregressive decoding.

    Stores keys/values as:
        k: (B, max_seq_len, n_heads, head_dim)
        v: (B, max_seq_len, n_heads, head_dim)

    Notes:
    - This cache is optional. For ViT / bidirectional encoders, pass kv_cache=None.
    - This implementation is "pure PyTorch" and trades performance for clarity.
    """
    max_batch_size: int
    max_seq_len: int
    n_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        self.k = torch.empty(
            (self.max_batch_size, self.max_seq_len, self.n_heads, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.v = torch.empty_like(self.k)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inserts new keys/values into the cache and returns the cached prefix up to end_pos.

        Args:
            k_new: (B, T, H, D)
            v_new: (B, T, H, D)
            start_pos: starting absolute position to write
        """
        bsz, t, h, d = k_new.shape
        end_pos = start_pos + t
        if bsz > self.max_batch_size:
            raise ValueError(f"bsz={bsz} exceeds cache max_batch_size={self.max_batch_size}")
        if end_pos > self.max_seq_len:
            raise ValueError(f"end_pos={end_pos} exceeds cache max_seq_len={self.max_seq_len}")
        if h != self.n_heads or d != self.head_dim:
            raise ValueError(f"Head shape mismatch: cache (H,D)=({self.n_heads},{self.head_dim}), got ({h},{d})")
        self.k[:bsz, start_pos:end_pos] = k_new
        self.v[:bsz, start_pos:end_pos] = v_new
        return self.k[:bsz, :end_pos], self.v[:bsz, :end_pos]


class MultiHeadSelfAttention(nn.Module):
    """
    Pure PyTorch Multi-Head Self-Attention using SDPA.

    Supports:
      - Optional KV cache (decode)
      - Optional 1D RoPE (freqs_cis) applied to first `rope_dim` of head_dim
      - Optional 2D RoPE (rope2d + positions_2d) applied to first `rope_dim` of head_dim

    Args:
        dim: model dimension
        n_heads: number of attention heads
        head_dim: per-head dimension (defaults to dim // n_heads)
        rope_dim: rotary-embedded sub-dimension of head_dim (defaults to head_dim)
        attn_dropout: dropout probability used inside SDPA (training)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        head_dim: Optional[int] = None,
        rope_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if dim % n_heads != 0:
                raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
            head_dim = dim // n_heads
        if rope_dim is None:
            rope_dim = head_dim
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.attn_dropout = float(attn_dropout)

        self.wqkv = nn.Linear(self.dim, 3 * self.n_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=bias)

    def _shape_qkv(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = qkv.shape
        qkv = qkv.view(bsz, seqlen, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)
        return q, k, v

    def _apply_rope_1d(self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope_dim == 0:
            return q, k
        q_pe, q_rest = q[..., : self.rope_dim], q[..., self.rope_dim :]
        k_pe, k_rest = k[..., : self.rope_dim], k[..., self.rope_dim :]
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)
        k_pe = apply_rotary_emb(k_pe, freqs_cis, interleaved=True)
        q = torch.cat([q_pe, q_rest], dim=-1)
        k = torch.cat([k_pe, k_rest], dim=-1)
        return q, k

    def _apply_rope_2d(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        rope2d: RotaryPositionEmbedding2D,
        positions_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope_dim == 0:
            return q, k

        # vggt rope2d expects (B, H, T, D)
        q_pe, q_rest = q[..., : self.rope_dim], q[..., self.rope_dim :]
        k_pe, k_rest = k[..., : self.rope_dim], k[..., self.rope_dim :]

        q_pe = q_pe.transpose(1, 2)  # (B, H, T, D)
        k_pe = k_pe.transpose(1, 2)  # (B, H, T, D)
        q_pe = rope2d(q_pe, positions_2d)
        k_pe = rope2d(k_pe, positions_2d)
        q_pe = q_pe.transpose(1, 2)  # (B, T, H, D)
        k_pe = k_pe.transpose(1, 2)

        q = torch.cat([q_pe, q_rest], dim=-1)
        k = torch.cat([k_pe, k_rest], dim=-1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        *,
        start_pos: int = 0,
        freqs_cis: Optional[torch.Tensor] = None,
        rope2d: Optional[RotaryPositionEmbedding2D] = None,
        positions_2d: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            start_pos: absolute start position for RoPE and/or causal masking (decode)
            freqs_cis: (T, rope_dim//2) complex cis for 1D RoPE (when used)
            rope2d: 2D rotary module (when used)
            positions_2d: (B, T, 2) y/x integer coordinates for each token (when used with rope2d)
            kv_cache: KVCache for autoregressive decode (optional)
            attn_mask: optional user mask; see module docstring
            is_causal: controls causal masking behavior. If None, it is inferred:
                      - True if no kv_cache and attn_mask is None
                      - False otherwise

        Returns:
            (B, T, dim)
        """
        bsz, q_len, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = self._shape_qkv(qkv)

        if (freqs_cis is not None) and (rope2d is not None or positions_2d is not None):
            raise ValueError("Use either 1D RoPE (freqs_cis) OR 2D RoPE (rope2d + positions_2d), not both.")

        # Apply RoPE (optional)
        if freqs_cis is not None:
            q, k = self._apply_rope_1d(q, k, freqs_cis)
        if rope2d is not None:
            if positions_2d is None:
                raise ValueError("positions_2d must be provided when rope2d is provided.")
            q, k = self._apply_rope_2d(q, k, rope2d=rope2d, positions_2d=positions_2d)

        # Cache (optional)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos=start_pos)

        k_len = k.shape[1]

        # SDPA expects (B, H, L, D)
        q_ = q.transpose(1, 2)  # (B, H, q_len, D)
        k_ = k.transpose(1, 2)  # (B, H, k_len, D)
        v_ = v.transpose(1, 2)  # (B, H, k_len, D)

        # Decide causal vs explicit mask
        if is_causal is None:
            # Default: causal only when "simple" (no external mask, no cache offset).
            is_causal = (attn_mask is None) and (kv_cache is None)

        sdpa_is_causal = bool(is_causal)
        sdpa_mask: Optional[torch.Tensor] = None

        if attn_mask is not None:
            # If any explicit mask is provided, avoid is_causal=True to maximize backend compatibility.
            sdpa_is_causal = False
            sdpa_mask = _normalize_attn_mask(attn_mask, q_len=q_len, k_len=k_len, device=x.device, dtype=x.dtype)

        if bool(is_causal) and (kv_cache is not None or start_pos != 0 or k_len != q_len):
            # Causal with cache/offset requires explicit additive mask.
            sdpa_is_causal = False
            causal = _make_additive_causal_mask(q_len, k_len, start_pos=start_pos, device=x.device, dtype=x.dtype)
            causal = causal[None, None, :, :]  # (1,1,q,k)
            sdpa_mask = causal if sdpa_mask is None else (sdpa_mask + causal)

        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=sdpa_is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.n_heads * self.head_dim)
        return self.wo(out)

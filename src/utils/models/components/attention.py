from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.models.components.rope import apply_rotary_emb


class MultiHeadSelfAttention(nn.Module):
    """
    Pure PyTorch Multi-Head Self-Attention using SDPA.

    Applies boundary-prepared RoPE frequencies to the first ``rope_dim`` of
    ``head_dim`` and consumes a boundary-prepared boolean keep-mask.

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
        head_dim: int,
        rope_dim: int,
        attn_dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        if rope_dim <= 0 or rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be positive and even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.attn_dropout = float(attn_dropout)

        self.wqkv = nn.Linear(self.dim, 3 * self.n_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=bias)

    def _shape_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = qkv.shape
        qkv = qkv.view(bsz, seqlen, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)
        return q, k, v

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_pe, q_rest = q[..., : self.rope_dim], q[..., self.rope_dim :]
        k_pe, k_rest = k[..., : self.rope_dim], k[..., self.rope_dim :]
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        k_pe = apply_rotary_emb(k_pe, freqs_cis)
        q = torch.cat([q_pe, q_rest], dim=-1)
        k = torch.cat([k_pe, k_rest], dim=-1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            freqs_cis: complex cis frequencies with a singleton head axis.
            attn_mask: prepared ``(B, T, T)`` boolean keep-mask.

        Returns:
            (B, T, dim)
        """
        bsz, q_len, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = self._shape_qkv(qkv)

        q, k = self._apply_rope(q, k, freqs_cis)

        # SDPA expects (B, H, L, D)
        q_ = q.transpose(1, 2)  # (B, H, q_len, D)
        k_ = k.transpose(1, 2)  # (B, H, k_len, D)
        v_ = v.transpose(1, 2)  # (B, H, k_len, D)

        out = F.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=attn_mask[:, None, :, :],
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.n_heads * self.head_dim)
        )
        return cast(Tensor, self.wo(out))


class GroupedQuerySelfAttention(nn.Module):
    """
    Pure PyTorch Grouped Query Self-Attention using SDPA.

    Query heads use ``n_heads`` while key/value heads use ``n_kv_heads``.
    The public forward interface matches ``MultiHeadSelfAttention`` so blocks
    can switch between MHA and GQA without changing call sites.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        head_dim: int,
        rope_dim: int,
        attn_dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be positive, got {n_kv_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
            )
        if rope_dim <= 0 or rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be positive and even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.enable_gqa = self.n_kv_heads != self.n_heads
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.attn_dropout = float(attn_dropout)

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=bias)

    def _shape_query(self, tensor: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = tensor.shape
        return tensor.view(bsz, seqlen, self.n_heads, self.head_dim)

    def _shape_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = tensor.shape
        return tensor.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_pe, query_rest = query[..., : self.rope_dim], query[..., self.rope_dim :]
        key_pe, key_rest = key[..., : self.rope_dim], key[..., self.rope_dim :]
        query_pe = apply_rotary_emb(query_pe, freqs_cis)
        key_pe = apply_rotary_emb(key_pe, freqs_cis)
        query = torch.cat([query_pe, query_rest], dim=-1)
        key = torch.cat([key_pe, key_rest], dim=-1)
        return query, key

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            freqs_cis: complex cis frequencies with a singleton head axis.
            attn_mask: prepared ``(B, T, T)`` boolean keep-mask.

        Returns:
            (B, T, dim)
        """
        bsz, q_len, _ = x.shape
        query = self._shape_query(self.wq(x))
        key = self._shape_kv(self.wk(x))
        value = self._shape_kv(self.wv(x))

        query, key = self._apply_rope(query, key, freqs_cis)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask[:, None, :, :],
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=self.enable_gqa,
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.n_heads * self.head_dim)
        )
        return cast(Tensor, self.wo(out))


class MultiHeadCrossAttention(nn.Module):
    """
    Pure PyTorch Multi-Head Cross-Attention using SDPA.

    Consumes boundary-prepared 1D RoPE frequencies and keep-masks for the query
    and key streams.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        head_dim: int,
        rope_dim: int,
        attn_dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        if rope_dim <= 0 or rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be positive and even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.attn_dropout = float(attn_dropout)

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=bias)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        return x.view(bsz, seqlen, self.n_heads, self.head_dim)

    def _apply_rope(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        x_pe, x_rest = x[..., : self.rope_dim], x[..., self.rope_dim :]
        x_pe = apply_rotary_emb(x_pe, freqs_cis)
        return torch.cat([x_pe, x_rest], dim=-1)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        *,
        freqs_q_cis: torch.Tensor,
        freqs_k_cis: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q_in: query tokens (B, Q, D)
            kv_in: key/value tokens (B, K, D)
            freqs_q_cis: prepared complex query frequencies.
            freqs_k_cis: prepared complex key frequencies.
            attn_mask: prepared ``(B, Q, K)`` boolean keep-mask.

        Returns:
            (B, Q, D)
        """
        bsz, q_len, _ = q_in.shape
        _, k_len, _ = kv_in.shape

        q = self._shape(self.wq(q_in))
        k = self._shape(self.wk(kv_in))
        v = self._shape(self.wv(kv_in))

        q = self._apply_rope(q, freqs_q_cis)
        k = self._apply_rope(k, freqs_k_cis)

        q_ = q.transpose(1, 2)  # (B, H, Q, D)
        k_ = k.transpose(1, 2)  # (B, H, K, D)
        v_ = v.transpose(1, 2)  # (B, H, K, D)

        out = F.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=attn_mask[:, None, :, :],
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.n_heads * self.head_dim)
        )
        return cast(Tensor, self.wo(out))

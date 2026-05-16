from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.models.components.ops.time_local import time_local_attention
from src.utils.models.components.rope import apply_rotary_emb


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
            raise ValueError(
                f"attn_mask shape must be {(q_len, k_len)}, got {tuple(attn_mask.shape)}"
            )
        attn_mask = attn_mask[None, None, :, :]  # (1,1,q,k)
    elif attn_mask.dim() == 3:
        if attn_mask.shape[1:] != (q_len, k_len):
            raise ValueError(
                f"attn_mask shape must be (B,{q_len},{k_len}), got {tuple(attn_mask.shape)}"
            )
        attn_mask = attn_mask[:, None, :, :]  # (B,1,q,k)
    elif attn_mask.dim() == 4:
        if attn_mask.shape[-2:] != (q_len, k_len):
            raise ValueError(
                f"attn_mask last dims must be ({q_len},{k_len}), got {tuple(attn_mask.shape)}"
            )
        # keep as-is; should be broadcastable to (B,H,q,k)
    else:
        raise ValueError(f"Unsupported attn_mask rank: {attn_mask.dim()}")

    return attn_mask


class MultiHeadSelfAttention(nn.Module):
    """
    Pure PyTorch Multi-Head Self-Attention using SDPA.

    Supports optional RoPE (freqs_cis) applied to first `rope_dim` of head_dim.

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
        head_dim: int | None = None,
        rope_dim: int | None = None,
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
        if self.rope_dim == 0:
            return q, k
        q_pe, q_rest = q[..., : self.rope_dim], q[..., self.rope_dim :]
        k_pe, k_rest = k[..., : self.rope_dim], k[..., self.rope_dim :]
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)
        k_pe = apply_rotary_emb(k_pe, freqs_cis, interleaved=True)
        q = torch.cat([q_pe, q_rest], dim=-1)
        k = torch.cat([k_pe, k_rest], dim=-1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        local_valid_mask: torch.Tensor | None = None,
        local_window_radius: int | None = None,
        local_use_cuda: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            freqs_cis: complex cis frequencies for RoPE. Supports `(T, rope_dim//2)`
                or batched `(B, T, rope_dim//2)`.
            attn_mask: optional user mask; see module docstring

        Returns:
            (B, T, dim)
        """
        bsz, q_len, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = self._shape_qkv(qkv)

        if freqs_cis is not None:
            q, k = self._apply_rope(q, k, freqs_cis)

        k_len = q_len

        # SDPA expects (B, H, L, D)
        q_ = q.transpose(1, 2)  # (B, H, q_len, D)
        k_ = k.transpose(1, 2)  # (B, H, k_len, D)
        v_ = v.transpose(1, 2)  # (B, H, k_len, D)

        if local_valid_mask is not None:
            if local_window_radius is None:
                raise ValueError(
                    "local_window_radius must be set when local_valid_mask is provided"
                )
            out = time_local_attention(
                q_,
                k_,
                v_,
                valid_mask=local_valid_mask,
                window_radius=local_window_radius,
                dropout_p=self.attn_dropout,
                training=self.training,
                use_cuda=local_use_cuda,
            )
        else:
            sdpa_mask: torch.Tensor | None = None
            if attn_mask is not None:
                sdpa_mask = _normalize_attn_mask(
                    attn_mask,
                    q_len=q_len,
                    k_len=k_len,
                    device=x.device,
                    dtype=x.dtype,
                )
            out = F.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.n_heads * self.head_dim)
        )
        return self.wo(out)

    def _sdpa_mask(
        self,
        attn_mask: torch.Tensor | None,
        *,
        q_len: int,
        k_len: int,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        sdpa_mask: torch.Tensor | None = None
        if attn_mask is not None:
            sdpa_mask = _normalize_attn_mask(
                attn_mask,
                q_len=q_len,
                k_len=k_len,
                device=x.device,
                dtype=x.dtype,
            )
        return sdpa_mask


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
        head_dim: int | None = None,
        rope_dim: int | None = None,
        attn_dropout: float = 0.0,
        bias: bool = False,
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
        self.n_kv_heads = int(n_kv_heads)
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
        if self.rope_dim == 0:
            return query, key
        query_pe, query_rest = query[..., : self.rope_dim], query[..., self.rope_dim :]
        key_pe, key_rest = key[..., : self.rope_dim], key[..., self.rope_dim :]
        query_pe = apply_rotary_emb(query_pe, freqs_cis, interleaved=True)
        key_pe = apply_rotary_emb(key_pe, freqs_cis, interleaved=True)
        query = torch.cat([query_pe, query_rest], dim=-1)
        key = torch.cat([key_pe, key_rest], dim=-1)
        return query, key

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        local_valid_mask: torch.Tensor | None = None,
        local_window_radius: int | None = None,
        local_use_cuda: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            freqs_cis: complex cis frequencies for RoPE. Supports `(T, rope_dim//2)`
                or batched `(B, T, rope_dim//2)`.
            attn_mask: optional user mask; see module docstring

        Returns:
            (B, T, dim)
        """
        bsz, q_len, _ = x.shape
        query = self._shape_query(self.wq(x))
        key = self._shape_kv(self.wk(x))
        value = self._shape_kv(self.wv(x))

        if freqs_cis is not None:
            query, key = self._apply_rope(query, key, freqs_cis)

        k_len = q_len
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if local_valid_mask is not None:
            if local_window_radius is None:
                raise ValueError(
                    "local_window_radius must be set when local_valid_mask is provided"
                )
            key_local = key
            value_local = value
            if self.n_kv_heads != self.n_heads:
                repeats = self.n_heads // self.n_kv_heads
                key_local = key_local.repeat_interleave(repeats, dim=1)
                value_local = value_local.repeat_interleave(repeats, dim=1)
            out = time_local_attention(
                query,
                key_local,
                value_local,
                valid_mask=local_valid_mask,
                window_radius=local_window_radius,
                dropout_p=self.attn_dropout,
                training=self.training,
                use_cuda=local_use_cuda,
            )
        else:
            sdpa_mask: torch.Tensor | None = None
            if attn_mask is not None:
                sdpa_mask = _normalize_attn_mask(
                    attn_mask,
                    q_len=q_len,
                    k_len=k_len,
                    device=x.device,
                    dtype=x.dtype,
                )
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
                enable_gqa=self.n_kv_heads != self.n_heads,
            )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.n_heads * self.head_dim)
        )
        return self.wo(out)

    def _sdpa_mask(
        self,
        attn_mask: torch.Tensor | None,
        *,
        q_len: int,
        k_len: int,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        sdpa_mask: torch.Tensor | None = None
        if attn_mask is not None:
            sdpa_mask = _normalize_attn_mask(
                attn_mask,
                q_len=q_len,
                k_len=k_len,
                device=x.device,
                dtype=x.dtype,
            )
        return sdpa_mask


class MultiHeadCrossAttention(nn.Module):
    """
    Pure PyTorch Multi-Head Cross-Attention using SDPA.

    Supports optional 1D RoPE for query and key streams independently.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        head_dim: int | None = None,
        rope_dim: int | None = None,
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
        freqs_cis: torch.Tensor | None,
    ) -> torch.Tensor:
        if freqs_cis is None or self.rope_dim == 0:
            return x
        if freqs_cis.ndim == 2:
            if freqs_cis.size(0) != x.size(1):
                raise ValueError(
                    f"freqs_cis length mismatch: freqs_cis.T={freqs_cis.size(0)} vs x.T={x.size(1)}"
                )
        elif freqs_cis.ndim == 3:
            if freqs_cis.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "freqs_cis batch/length mismatch: "
                    f"freqs={tuple(freqs_cis.shape[:2])} vs x={tuple(x.shape[:2])}"
                )
        else:
            raise ValueError(
                f"freqs_cis must have rank 2 or 3, got shape {tuple(freqs_cis.shape)}"
            )
        x_pe, x_rest = x[..., : self.rope_dim], x[..., self.rope_dim :]
        x_pe = apply_rotary_emb(x_pe, freqs_cis, interleaved=True)
        return torch.cat([x_pe, x_rest], dim=-1)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        *,
        freqs_q_cis: torch.Tensor | None = None,
        freqs_k_cis: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q_in: query tokens (B, Q, D)
            kv_in: key/value tokens (B, K, D)
            freqs_q_cis: complex cis frequencies for query RoPE. Supports
                `(Q, rope_dim//2)` or batched `(B, Q, rope_dim//2)`.
            freqs_k_cis: complex cis frequencies for key RoPE. Supports
                `(K, rope_dim//2)` or batched `(B, K, rope_dim//2)`.
            attn_mask: optional mask broadcastable to (B, H, Q, K)

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

        sdpa_mask: torch.Tensor | None = None
        if attn_mask is not None:
            sdpa_mask = _normalize_attn_mask(
                attn_mask,
                q_len=q_len,
                k_len=k_len,
                device=q_in.device,
                dtype=q_in.dtype,
            )

        out = F.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.n_heads * self.head_dim)
        )
        return self.wo(out)

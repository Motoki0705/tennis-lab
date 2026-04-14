"""Interleaved rotary position embedding utilities.

This module keeps the existing 1D helper API while making the core frequency
construction generic over arbitrary coordinate ranks. Frequencies are assigned in
an interleaved manner across axes, following the Qwen3-VL-style MROPE layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch

RopeBaseLike: TypeAlias = float | Sequence[float] | torch.Tensor


def _normalize_rope_bases(
    base: RopeBaseLike,
    *,
    n_axes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(base, torch.Tensor):
        base_tensor = base.to(device=device, dtype=dtype).flatten()
    elif isinstance(base, Sequence):
        base_tensor = torch.tensor(list(base), device=device, dtype=dtype).flatten()
    else:
        base_tensor = torch.full((n_axes,), float(base), device=device, dtype=dtype)

    if base_tensor.numel() == 1:
        base_tensor = base_tensor.expand(n_axes)
    if base_tensor.numel() != n_axes:
        raise ValueError(f"Expected {n_axes} RoPE bases, got {base_tensor.numel()}")
    if (base_tensor <= 0).any():
        raise ValueError("RoPE bases must be positive.")
    return base_tensor


def precompute_freqs_cis(
    *,
    dim: int,
    seqlen: int,
    base: RopeBaseLike = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute complex cis frequencies for standard 1D RoPE."""
    if seqlen < 0:
        raise ValueError(f"seqlen must be non-negative, got {seqlen}")
    positions = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(-1)
    return precompute_freqs_cis_nd(dim=dim, pos=positions, base=base)


def precompute_freqs_cis_nd(
    dim: int,
    pos: torch.Tensor,
    base: RopeBaseLike = 10000.0,
) -> torch.Tensor:
    """Precompute complex cis frequencies for interleaved N-dimensional MROPE."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")

    if pos.ndim == 0:
        raise ValueError("pos must have at least one dimension.")
    if pos.ndim == 1:
        pos = pos.unsqueeze(-1)
    if pos.size(-1) <= 0:
        raise ValueError(f"pos must have a positive axis dimension, got shape {tuple(pos.shape)}")

    device = pos.device
    dtype = torch.float32
    half_dim = dim // 2
    n_axes = int(pos.size(-1))

    pair_indices = torch.arange(half_dim, device=device, dtype=dtype)
    axis_indices = torch.arange(half_dim, device=device) % n_axes
    base_tensor = _normalize_rope_bases(base, n_axes=n_axes, device=device, dtype=dtype)
    base_per_pair = base_tensor[axis_indices]

    # Each axis receives rotary pairs cyclically across the full spectrum.
    inv_freq = 1.0 / (base_per_pair ** ((2.0 * pair_indices) / dim))

    pos_interleaved = pos.to(dtype)[..., axis_indices]
    angles = pos_interleaved * inv_freq
    return torch.polar(torch.ones_like(angles), angles)


def _reshape_freqs_cis_for_broadcast(
    x_complex: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    if freqs_cis.shape[-1] != x_complex.shape[-1]:
        raise ValueError(
            "RoPE frequency dim mismatch: "
            f"freqs={freqs_cis.shape[-1]} vs x={x_complex.shape[-1]}"
        )

    if x_complex.ndim == 3:
        if freqs_cis.ndim == 2:
            if freqs_cis.shape[0] != x_complex.shape[1]:
                raise ValueError(
                    f"RoPE length mismatch: freqs.T={freqs_cis.shape[0]} vs x.T={x_complex.shape[1]}"
                )
            return freqs_cis.view(1, x_complex.shape[1], x_complex.shape[2])
        if freqs_cis.ndim == 3:
            if freqs_cis.shape[:2] != x_complex.shape[:2]:
                raise ValueError(
                    "RoPE batch/length mismatch: "
                    f"freqs={tuple(freqs_cis.shape[:2])} vs x={tuple(x_complex.shape[:2])}"
                )
            return freqs_cis
    elif x_complex.ndim == 4:
        if freqs_cis.ndim == 2:
            if freqs_cis.shape[0] != x_complex.shape[1]:
                raise ValueError(
                    f"RoPE length mismatch: freqs.T={freqs_cis.shape[0]} vs x.T={x_complex.shape[1]}"
                )
            return freqs_cis.view(1, x_complex.shape[1], 1, x_complex.shape[3])
        if freqs_cis.ndim == 3:
            if freqs_cis.shape[:2] != x_complex.shape[:2]:
                raise ValueError(
                    "RoPE batch/length mismatch: "
                    f"freqs={tuple(freqs_cis.shape[:2])} vs x={tuple(x_complex.shape[:2])}"
                )
            return freqs_cis.view(x_complex.shape[0], x_complex.shape[1], 1, x_complex.shape[3])

    raise ValueError(
        "Unsupported RoPE broadcast combination: "
        f"x_complex.shape={tuple(x_complex.shape)}, freqs_cis.shape={tuple(freqs_cis.shape)}"
    )


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    interleaved: bool = True,
) -> torch.Tensor:
    """Apply rotary embeddings to `(B, T, D)` or `(B, T, H, D)` tensors."""
    if x.ndim not in {3, 4}:
        raise ValueError(f"Unsupported input rank for RoPE: {x.ndim}")
    if x.size(-1) % 2 != 0:
        raise ValueError(f"RoPE expects an even dim, got {x.size(-1)}")

    dtype = x.dtype
    shape = x.shape
    x_work = x

    if not interleaved:
        x_work = x_work.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    x_complex = torch.view_as_complex(x_work.float().view(*x_work.shape[:-1], -1, 2))
    freqs_cis = _reshape_freqs_cis_for_broadcast(x_complex, freqs_cis.to(device=x.device))

    y = torch.view_as_real(x_complex * freqs_cis).flatten(-2)

    if not interleaved:
        y = y.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous().view(*shape)

    return y.to(dtype)
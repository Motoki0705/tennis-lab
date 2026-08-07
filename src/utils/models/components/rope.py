"""Interleaved rotary position embedding utilities.

This module keeps the existing 1D helper API while making the core frequency
construction generic over arbitrary coordinate ranks. Frequencies are assigned in
an interleaved manner across axes, following the Qwen3-VL-style MROPE layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch
from torch import nn

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
        raise ValueError(
            f"RoPE base must contain one value or {n_axes} axis values, "
            f"got {base_tensor.numel()}."
        )
    if not torch.isfinite(base_tensor).all() or (base_tensor <= 0).any():
        raise ValueError("RoPE bases must be finite and positive.")
    return base_tensor


class RotaryFrequencyComputer(nn.Module):
    """Constructor-prepared interleaved N-D rotary frequencies.

    Base representation and axis assignment are resolved once in ``__init__``.
    ``forward`` accepts only boundary-validated position tensors and returns
    frequencies already shaped for broadcasting over the attention head axis.
    """

    def __init__(self, *, dim: int, base: RopeBaseLike, n_axes: int) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim must be a positive even int, got {dim!r}.")
        if type(n_axes) is not int or n_axes <= 0:
            raise ValueError(f"n_axes must be a positive int, got {n_axes!r}.")

        half_dim = dim // 2
        pair_indices = torch.arange(half_dim, dtype=torch.float32)
        axis_indices = torch.arange(half_dim, dtype=torch.long) % n_axes
        bases = _normalize_rope_bases(
            base,
            n_axes=n_axes,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        base_per_pair = bases[axis_indices]
        inverse_frequencies = 1.0 / (base_per_pair ** ((2.0 * pair_indices) / dim))
        self.n_axes = n_axes
        self.axis_indices: torch.Tensor
        self.inverse_frequencies: torch.Tensor
        self.register_buffer("axis_indices", axis_indices, persistent=False)
        self.register_buffer(
            "inverse_frequencies", inverse_frequencies, persistent=False
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Return complex freqs for validated ``(..., T, n_axes)`` positions."""
        interleaved_positions = positions[..., self.axis_indices]
        angles = interleaved_positions * self.inverse_frequencies
        return torch.polar(torch.ones_like(angles), angles).unsqueeze(-2)


def precompute_freqs_cis(
    *,
    dim: int,
    seqlen: int,
    base: RopeBaseLike = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Prepare static standard 1D RoPE frequencies outside model execution."""
    positions = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(-1)
    computer = RotaryFrequencyComputer(dim=dim, base=base, n_axes=1).to(
        device=positions.device
    )
    return computer.forward(positions)


def precompute_freqs_cis_nd(
    dim: int,
    pos: torch.Tensor,
    base: RopeBaseLike = 10000.0,
) -> torch.Tensor:
    """Prepare N-D frequencies at a construction or data boundary."""
    computer = RotaryFrequencyComputer(
        dim=dim,
        base=base,
        n_axes=int(pos.shape[-1]),
    ).to(device=pos.device)
    return computer.forward(pos)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply interleaved RoPE to prepared ``(B,T,H,D)`` tensors/frequencies."""
    dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    y = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return y.to(dtype)

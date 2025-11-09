"""Positional encodings used by the scene model."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn


class AbsTimePE(nn.Module):
    """Fixed frequency sinusoidal encoder with learnable scale/phase."""

    def __init__(self, K: int, fps: float, Thorizon: float) -> None:
        super().__init__()
        if K <= 0:
            msg = "K must be positive"
            raise ValueError(msg)
        if Thorizon <= 0:
            msg = "Thorizon must be positive"
            raise ValueError(msg)
        w_min = 2 * math.pi / Thorizon
        w_max = 2 * math.pi * 0.45 * fps
        if w_min <= 0 or w_max <= 0:
            msg = "Frequencies must be positive"
            raise ValueError(msg)
        self.register_buffer(
            "freqs",
            torch.logspace(
                math.log10(w_min),
                math.log10(w_max),
                steps=K,
            ),
            persistent=False,
        )
        self.scale = nn.Parameter(torch.ones(K))
        self.phase = nn.Parameter(torch.zeros(K))

    def forward(self, t_sec: Tensor) -> Tensor:
        """Return the sine/cosine absolute time encoding."""
        freqs = cast(Tensor, self.freqs).to(device=t_sec.device, dtype=t_sec.dtype)
        scale = self.scale.to(dtype=t_sec.dtype)
        phase = self.phase.to(device=t_sec.device, dtype=t_sec.dtype)
        t = t_sec.unsqueeze(-1)
        arg = t * freqs * scale + phase
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)


class RelTimePE(nn.Module):
    """Relative time encoding used inside the deformable decoder."""

    def __init__(self, Q: int, wmin: float, wmax: float) -> None:
        super().__init__()
        if Q <= 0:
            msg = "Q must be positive"
            raise ValueError(msg)
        if wmin <= 0 or wmax <= 0:
            msg = "Frequencies must be positive"
            raise ValueError(msg)
        self.register_buffer(
            "freqs",
            torch.logspace(
                math.log10(wmin),
                math.log10(wmax),
                steps=Q,
            ),
            persistent=False,
        )

    def forward(self, delta_sec: Tensor) -> Tensor:
        """Return the sine/cosine relative time encoding."""
        freqs = cast(Tensor, self.freqs).to(
            device=delta_sec.device, dtype=delta_sec.dtype
        )
        d = delta_sec.unsqueeze(-1)
        arg = d * freqs
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)


def fixed_2d_sincos(H: int, W: int, D: int) -> Tensor:
    """Generate a fixed 2D sine/cosine grid."""
    if H <= 0 or W <= 0 or D <= 0:
        msg = "H, W, and D must be positive"
        raise ValueError(msg)
    half = D // 2
    pe_y = _sincos_1d(H, half)
    pe_x = _sincos_1d(W, D - half)
    grid = torch.zeros(H * W, D)
    idx = 0
    for y in range(H):
        for x in range(W):
            grid[idx, :half] = pe_y[y]
            grid[idx, half:] = pe_x[x]
            idx += 1
    return grid.unsqueeze(0)


def _sincos_1d(length: int, dim: int) -> Tensor:
    if dim <= 0:
        return torch.zeros(length, 0)
    position = torch.linspace(-1.0, 1.0, steps=length).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=position.dtype) * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term)[:, : dim // 2]
    return pe

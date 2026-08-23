"""Patch-only two-dimensional RoPE for query-encoder self-attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.utils.models.components.rope import (
    RotaryFrequencyComputer,
    apply_rotary_emb,
)


def build_patch_positions(
    grid_hw: tuple[int, int],
    *,
    device: torch.device,
) -> Tensor:
    """Return row-major ``(x,y)`` coordinates for one strict patch grid."""
    grid_h, grid_w = grid_hw
    if (
        type(grid_h) is not int
        or type(grid_w) is not int
        or grid_h <= 0
        or grid_w <= 0
    ):
        raise ValueError("grid_hw must contain two positive integers.")
    rows, columns = torch.meshgrid(
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    return torch.stack((columns, rows), dim=-1).reshape(-1, 2)


def apply_patch_only_rope(
    query: Tensor,
    key: Tensor,
    *,
    grid_hw: tuple[int, int],
    rope_dim: int,
    frequency_computer: RotaryFrequencyComputer,
) -> tuple[Tensor, Tensor]:
    """Rotate patch Q/K positions while leaving the leading pose query unchanged."""
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("Query/key must share shape (B,H,1+N,D).")
    grid_h, grid_w = grid_hw
    expected_tokens = 1 + grid_h * grid_w
    if query.shape[2] != expected_tokens:
        raise ValueError(
            "Attention sequence must contain one pose query plus the exact patch grid."
        )
    if rope_dim <= 0 or rope_dim % 4 or rope_dim > query.shape[-1]:
        raise ValueError(
            "rope_dim must be positive, divisible by four, and <= head dimension."
        )
    positions = build_patch_positions(grid_hw, device=query.device)
    frequencies = frequency_computer(positions)

    def rotate(value: Tensor) -> Tensor:
        pose = value[:, :, :1]
        patch = value[:, :, 1:].transpose(1, 2)
        rotated = apply_rotary_emb(patch[..., :rope_dim], frequencies)
        patch = torch.cat((rotated, patch[..., rope_dim:]), dim=-1).transpose(1, 2)
        return torch.cat((pose, patch), dim=2)

    return rotate(query), rotate(key)


class PatchRoPEMultiheadAttention(nn.Module):
    """Self-attention with one positionless query and patch-only 2-D RoPE."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        rope_dim: int,
        rope_theta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if rope_dim <= 0 or rope_dim % 4 or rope_dim > self.head_dim:
            raise ValueError("Invalid patch RoPE dimension.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Attention dropout must be in [0,1).")
        self.rope_dim = rope_dim
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.frequency_computer = RotaryFrequencyComputer(
            dim=rope_dim,
            base=rope_theta,
            n_axes=2,
        )

    def forward(self, tokens: Tensor, *, grid_hw: tuple[int, int]) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError("Attention tokens must have shape (B,1+N,hidden_dim).")
        batch_size, token_count, _ = tokens.shape
        if token_count != 1 + grid_hw[0] * grid_hw[1]:
            raise ValueError("Attention token count disagrees with the patch grid.")
        qkv = self.qkv(tokens).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_patch_only_rope(
            query,
            key,
            grid_hw=grid_hw,
            rope_dim=self.rope_dim,
            frequency_computer=self.frequency_computer,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            token_count,
            self.hidden_dim,
        )
        return self.output(attended)


__all__ = [
    "PatchRoPEMultiheadAttention",
    "apply_patch_only_rope",
    "build_patch_positions",
]

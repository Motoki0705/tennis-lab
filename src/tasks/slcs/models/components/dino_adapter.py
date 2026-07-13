"""DINOv3 patch-token encoder for the SLCS fusion model.

Projects raw patch tokens ``(B, T_d, S, C_in)`` into the model width and adds
a fixed 2D sin-cos spatial position embedding over the patch grid. Temporal
position is *not* encoded here — the fusion model applies time-axis RoPE using
the explicit ``dino_frame_idx`` during cross-attention, so the sparse sampling
cadence stays visible to the attention mechanism instead of being blurred by
interpolation.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.utils.models import RMSNorm


def sincos_position_embedding_2d(grid_h: int, grid_w: int, dim: int) -> Tensor:
    """Fixed 2D sin-cos embedding of shape ``(grid_h * grid_w, dim)``.

    Half of the channels encode the row coordinate, half the column coordinate
    (each as interleaved sin/cos of geometrically spaced frequencies).
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sin-cos embedding, got {dim}.")
    dim_axis = dim // 2

    def encode(pos: Tensor) -> Tensor:
        half = dim_axis // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
        )
        angles = pos[:, None].to(torch.float32) * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    row_embed = encode(rows)  # (grid_h, dim/2)
    col_embed = encode(cols)  # (grid_w, dim/2)
    out = torch.cat(
        [
            row_embed[:, None, :].expand(grid_h, grid_w, dim_axis),
            col_embed[None, :, :].expand(grid_h, grid_w, dim_axis),
        ],
        dim=-1,
    )
    return out.reshape(grid_h * grid_w, dim)


class DinoTokenEncoder(nn.Module):
    """Spatially downsample, then project DINOv3 patch tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        dim: int,
        grid_h: int,
        grid_w: int,
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or dim <= 0:
            raise ValueError(f"input_dim and dim must be positive, got {input_dim}, {dim}.")
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"grid must be positive, got ({grid_h}, {grid_w}).")
        if downsample_factor <= 0:
            raise ValueError(
                f"downsample_factor must be positive, got {downsample_factor}."
            )
        if grid_h % downsample_factor != 0 or grid_w % downsample_factor != 0:
            raise ValueError(
                "DINO grid dimensions must be divisible by downsample_factor, "
                f"got grid=({grid_h}, {grid_w}) and factor={downsample_factor}."
            )
        self.input_dim = int(input_dim)
        self.dim = int(dim)
        self.input_grid_h = int(grid_h)
        self.input_grid_w = int(grid_w)
        self.downsample_factor = int(downsample_factor)
        self.grid_h = self.input_grid_h // self.downsample_factor
        self.grid_w = self.input_grid_w // self.downsample_factor
        self.num_input_tokens = self.input_grid_h * self.input_grid_w
        self.num_tokens = self.grid_h * self.grid_w

        self.proj = nn.Linear(self.input_dim, self.dim)
        self.norm = RMSNorm(self.dim)
        self.register_buffer(
            "spatial_pos",
            sincos_position_embedding_2d(self.grid_h, self.grid_w, self.dim),
            persistent=False,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Encode ``(B,T_d,H*W,C_in)`` into the reduced patch grid."""
        if tokens.dim() != 4:
            raise ValueError(
                f"DinoTokenEncoder expects (B, T_d, S, C), got shape {tuple(tokens.shape)}."
            )
        if tokens.shape[2] != self.num_input_tokens:
            raise ValueError(
                f"token count S={tokens.shape[2]} does not match the configured grid "
                f"{self.input_grid_h}x{self.input_grid_w}={self.num_input_tokens}."
            )
        if tokens.shape[3] != self.input_dim:
            raise ValueError(
                f"token width C={tokens.shape[3]} does not match configured "
                f"input_dim={self.input_dim}."
            )
        batch_size, num_samples = tokens.shape[:2]
        if self.downsample_factor > 1:
            # Interpolate in the original DINO feature space. Projection after
            # reduction keeps the requested spatial compression independent of
            # the learned channel compression.
            tokens = tokens.reshape(
                batch_size,
                num_samples,
                self.input_grid_h,
                self.input_grid_w,
                self.input_dim,
            )
            tokens = tokens.permute(0, 1, 4, 2, 3).reshape(
                batch_size * num_samples,
                self.input_dim,
                self.input_grid_h,
                self.input_grid_w,
            )
            tokens = F.interpolate(
                tokens,
                size=(self.grid_h, self.grid_w),
                mode="bilinear",
                align_corners=False,
            )
            tokens = tokens.reshape(
                batch_size, num_samples, self.input_dim, self.grid_h, self.grid_w
            ).permute(0, 1, 3, 4, 2)
            tokens = tokens.reshape(
                batch_size, num_samples, self.num_tokens, self.input_dim
            )

        x = self.norm(self.proj(tokens))
        spatial_pos = cast(Tensor, self.spatial_pos)
        return x + spatial_pos.to(dtype=x.dtype)[None, None, :, :]


__all__ = ["DinoTokenEncoder", "sincos_position_embedding_2d"]

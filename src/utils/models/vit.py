"""Vision Transformer (ViT) encoder built from DeepSeek-style components.

This module intentionally uses the "pure PyTorch / DeepSeek-style" building blocks in
`src.common.models.components` (MultiHeadSelfAttention + 2D RoPE + SwiGLU/MoE).

It is primarily consumed by `src/mae/models/mae_model.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models.components import MoEConfig, RMSNorm, ViTBlock, ViTBlockConfig


@dataclass
class ViTConfig:
    """Configuration for ViTEncoder.

    Args:
        patch_size: Patch size for image tokenization.
        in_channels: Number of input image channels.
        max_resolution: Maximum input resolution (assumed square).
        hidden_dim: Token embedding dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        ffn_dim: MLP hidden dimension (defaults to 8/3 rule if None).
        dropout: Dropout probability for attention/MLP.
        num_register_tokens: Number of register tokens (inserted after CLS).
        use_cls_token: Whether to prepend a CLS token.
        rope_dim: Rotary dimension per head for 2D RoPE.
        rope_theta: Base theta for 2D RoPE.
        use_moe: Whether to use MoE FFNs.
        moe_config: MoE configuration when `use_moe=True`.
        moe_layer_freq: Apply MoE every N layers.
        pooling: Output pooling mode for encoder outputs.
    """

    # Patch embedding
    patch_size: int = 16
    in_channels: int = 3
    max_resolution: int = 224

    # Architecture
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int | None = None
    dropout: float = 0.1

    # Special tokens
    num_register_tokens: int = 4
    use_cls_token: bool = True

    # 2D RoPE
    rope_dim: int | None = None
    rope_theta: float = 1000.0  # RoPE base (theta) for 2D axial RoPE

    # MoE (optional)
    use_moe: bool = True
    moe_config: MoEConfig | None = None
    moe_layer_freq: int = 2

    # Output selection
    pooling: Literal["cls", "mean", "all"] = "all"


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, patch_size: int, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)  # (B, D, H', W')
        bsz, dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x, h, w


class ViTEncoder(nn.Module):
    """ViT encoder with 2D RoPE using DeepSeek-style ViTBlock."""

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = int(cfg.patch_size)
        self.max_resolution = int(cfg.max_resolution)

        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            in_channels=cfg.in_channels,
            hidden_dim=cfg.hidden_dim,
        )

        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if cfg.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, cfg.num_register_tokens, cfg.hidden_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        ffn_dim = cfg.ffn_dim
        if ffn_dim is None:
            ffn_dim = int((8 * cfg.hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        head_dim = cfg.hidden_dim // cfg.num_heads
        rope_dim = cfg.rope_dim or head_dim
        rope_dim = min(int(rope_dim), int(head_dim))
        rope_dim = (rope_dim // 4) * 4  # 2D RoPE requires divisible by 4
        if rope_dim <= 0:
            rope_dim = 0

        self.blocks = nn.ModuleList()
        for layer_idx in range(cfg.num_layers):
            use_moe = cfg.use_moe and (layer_idx % cfg.moe_layer_freq == 0)
            moe_cfg: MoEConfig | None = None
            if use_moe:
                moe_cfg = cfg.moe_config or MoEConfig(
                    dim=cfg.hidden_dim,
                    moe_inter_dim=ffn_dim,
                    n_routed_experts=8,
                    n_shared_experts=1,
                    n_activated_experts=2,
                )

            block_cfg = ViTBlockConfig(
                dim=cfg.hidden_dim,
                n_heads=cfg.num_heads,
                mlp_inter_dim=ffn_dim,
                attn_dropout=cfg.dropout,
                mlp_dropout=cfg.dropout,
                use_2d_rope=(rope_dim > 0),
                rope2d_frequency=cfg.rope_theta,
                rope_dim=rope_dim if rope_dim > 0 else None,
                use_moe=use_moe,
                moe_config=moe_cfg,
            )
            self.blocks.append(ViTBlock(block_cfg))

        self.norm = RMSNorm(cfg.hidden_dim)

        max_patches_h = self.max_resolution // self.patch_size
        max_patches_w = self.max_resolution // self.patch_size
        if max_patches_h <= 0 or max_patches_w <= 0:
            raise ValueError(
                f"max_resolution={self.max_resolution} must be >= patch_size={self.patch_size}."
            )

        positions_2d_grid = self._build_positions_2d_grid(max_patches_h, max_patches_w)
        self.register_buffer("_positions_2d_grid", positions_2d_grid, persistent=False)

    @staticmethod
    def _build_positions_2d_grid(num_patches_h: int, num_patches_w: int) -> Tensor:
        """Build a (H, W, 2) integer (y,x) position grid for patch tokens."""
        y = torch.arange(num_patches_h, dtype=torch.long)
        x = torch.arange(num_patches_w, dtype=torch.long)
        return torch.cartesian_prod(y, x).view(num_patches_h, num_patches_w, 2)

    def _slice_positions_2d(self, bsz: int, num_patches_h: int, num_patches_w: int, device: torch.device) -> Tensor:
        """Slice cached (y,x) integer positions for patch tokens.

        Special tokens ([CLS]/[REG...]) are treated as a prefix and are not rotated by 2D RoPE.
        """
        max_patches_h, max_patches_w, _ = self._positions_2d_grid.shape
        if num_patches_h > max_patches_h or num_patches_w > max_patches_w:
            raise ValueError(
                f"Patch grid (h={num_patches_h}, w={num_patches_w}) exceeds cached "
                f"max grid (h={max_patches_h}, w={max_patches_w}). "
                f"Increase max_resolution={self.max_resolution}."
            )

        patch_pos = self._positions_2d_grid[:num_patches_h, :num_patches_w].reshape(1, -1, 2)
        if patch_pos.device != device:
            patch_pos = patch_pos.to(device)
        return patch_pos.expand(bsz, -1, -1).contiguous()

    def forward(self, x: Tensor, *, return_all_tokens: bool = False) -> Tensor:
        """Forward pass.

        Args:
            x: image tensor, shape (B, C, H, W)
            return_all_tokens: kept for compatibility; same as pooling='all'
        """
        bsz = x.shape[0]
        tok, h, w = self.patch_embed(x)  # (B, N, D), (H', W')

        if self.cfg.use_cls_token:
            cls = self.cls_token.expand(bsz, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
        if self.cfg.num_register_tokens > 0:
            reg = self.register_tokens.expand(bsz, -1, -1)
            insert_at = 1 if self.cfg.use_cls_token else 0
            tok = torch.cat([tok[:, :insert_at], reg, tok[:, insert_at:]], dim=1)

        positions_2d = self._slice_positions_2d(bsz, h, w, x.device)

        for block in self.blocks:
            tok = block(tok, positions_2d=positions_2d, grid_hw=(h, w))

        tok = self.norm(tok)

        pooling = "all" if return_all_tokens else self.cfg.pooling
        if pooling == "all":
            return tok
        if pooling == "cls":
            if not self.cfg.use_cls_token:
                raise ValueError("pooling='cls' requires use_cls_token=True.")
            return tok[:, 0]
        if pooling == "mean":
            start = 1 if self.cfg.use_cls_token else 0
            start += self.cfg.num_register_tokens
            return tok[:, start:].mean(dim=1)
        raise ValueError(f"Unknown pooling={pooling!r}")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_cfg = ViTConfig(
        max_resolution=64,
        patch_size=8,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_register_tokens=2,
        use_moe=False,
        pooling="cls",
    )
    demo_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_model = ViTEncoder(demo_cfg).eval().to(demo_device)
    demo_input = torch.randn(1, demo_cfg.in_channels, 56, 56).to(demo_device)

    with torch.no_grad():
        demo_output = demo_model(demo_input)

    print(demo_output)

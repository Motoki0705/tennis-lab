"""Vision backbone built around a lightweight ViT."""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import init


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, img_size: int, patch_size: int, dim: int) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            msg = "img_size must be divisible by patch_size"
            raise ValueError(msg)
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """Convert an image batch into a sequence of flattened patch tokens."""
        feat = cast(Tensor, self.proj(x))
        return feat.flatten(2).transpose(1, 2)


class ViTBackbone(nn.Module):
    """Minimal ViT that emits patch and CLS tokens per frame."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, dim)
        self.dim = dim
        self.num_patches = self.patch_embed.grid_h * self.patch_embed.grid_w
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.norm = nn.LayerNorm(dim)
        init.trunc_normal_(self.cls_token, std=0.02)
        init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        """Encode frames into patch tokens and CLS tokens."""
        if frames.ndim != 5:
            msg = "frames must be [B,T,3,H,W]"
            raise ValueError(msg)
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        patches = cast(Tensor, self.patch_embed(flat))
        cls_tokens = self.cls_token.expand(patches.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + self.pos_embed
        encoded = cast(Tensor, self.encoder(tokens))
        encoded = cast(Tensor, self.norm(encoded))
        cls = encoded[:, 0]
        patch_tokens = encoded[:, 1:]
        patch_tokens = patch_tokens.view(B, T, self.num_patches, self.dim)
        cls = cls.view(B, T, self.dim)
        return patch_tokens, cls

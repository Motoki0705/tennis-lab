"""Modern Vision Transformer (ViT) with latest architectural innovations.

This module implements a state-of-the-art Vision Transformer incorporating:
- 2D Rotary Position Embedding (RoPE2D) for spatial positions
- Register tokens for improved attention patterns
- CLS token + Register tokens + Patch tokens structure
- Optional MoE (Mixture of Experts) for FFN
- Optional MLA (Multi-head Latent Attention)
- GQA (Grouped-Query Attention) for efficiency
- RMSNorm (Pre-Norm) for stability
- SwiGLU activation in FFN

Token Order:
    [CLS, Register_1, ..., Register_R, Patch_1, ..., Patch_N]

    - CLS token: Global representation, gets RoPE position 0
    - Register tokens: Learnable tokens between CLS and patches (no RoPE)
    - Patch tokens: Image patches with 2D RoPE positions

References:
    - ViT: https://arxiv.org/abs/2010.11929
    - RoPE: https://arxiv.org/abs/2104.09864
    - Register tokens: https://arxiv.org/abs/2309.16588
    - DeepSeek MoE: https://arxiv.org/abs/2401.06066
    - MLA: https://arxiv.org/abs/2405.04434
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.common.models.components import (
    GQA,
    GQAConfig,
    MLA,
    MLAConfig,
    MoEConfig,
    MoELayer,
    RMSNorm,
    RoPE2D,
    RoPE2DConfig,
    SwiGLUMLP,
    ViTBlock,
    ViTBlockConfig,
)


@dataclass
class ViTConfig:
    """Configuration for Modern Vision Transformer.

    Attributes:
        # Patch embedding
        patch_size: Size of image patches (patch_size x patch_size).
        in_channels: Number of input image channels.
        image_size: Default image size (can vary at runtime).

        # Architecture
        hidden_dim: Transformer hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        ffn_dim: FFN intermediate dimension. Auto-computed if None.
        dropout: Dropout probability.

        # Special tokens
        num_register_tokens: Number of register tokens (0 to disable).
        use_cls_token: Whether to use CLS token.

        # Positional encoding
        rope_dim: Dimension for 2D RoPE.
        rope_theta: Base frequency for RoPE.
        rope_interleave: Whether to interleave h/w in RoPE dimensions.

        # Advanced options
        use_moe: Whether to use MoE for FFN layers.
        moe_config: Configuration for MoE (if use_moe=True).
        moe_layer_freq: Apply MoE every N layers (e.g., 2 = every other layer).
        use_mla: Whether to use MLA for attention.
        mla_kv_lora_rank: KV compression rank for MLA.

        # Pooling
        pooling: How to pool output. 'cls', 'mean', or 'all'.

    """

    # Patch embedding
    patch_size: int = 16
    in_channels: int = 3
    image_size: int = 224

    # Architecture
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4
    ffn_dim: int | None = None
    dropout: float = 0.1
    attn_type: Literal["mha", "gqa", "mla"] = "gqa"

    # Special tokens
    num_register_tokens: int = 4
    use_cls_token: bool = True

    # Positional encoding
    rope_dim: int | None = None
    rope_theta: float = 10000.0
    rope_interleave: bool = True

    # Advanced options
    use_moe: bool = False
    moe_config: MoEConfig | None = None
    moe_layer_freq: int = 2
    mla_kv_lora_rank: int = 64

    # Pooling
    pooling: Literal["cls", "mean", "all"] = "cls"


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings.

    Splits image into non-overlapping patches and projects each to hidden_dim.
    """

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
    ) -> None:
        """Initialize patch embedding.

        Args:
            patch_size: Size of each patch (P x P).
            in_channels: Number of input channels.
            hidden_dim: Output embedding dimension.

        """
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Extract patch embeddings.

        Args:
            x: Input image, shape (B, C, H, W).

        Returns:
            Tuple of:
                - Patch embeddings, shape (B, num_patches, hidden_dim)
                - Number of patches in height (H // patch_size)
                - Number of patches in width (W // patch_size)

        """
        # x: (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.proj(x)
        B, D, H, W = x.shape
        # Flatten to (B, D, H*W) -> (B, H*W, D)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class ViTEncoder(nn.Module):
    """Vision Transformer encoder with modern architectural innovations.

    Token structure:
        [CLS, Register_1, ..., Register_R, Patch_1, ..., Patch_N]

    RoPE is applied as follows:
        - CLS token: position (0, 0)
        - Register tokens: position (0, 1), (0, 2), ... (virtual column)
        - Patch tokens: 2D positions (h, w) based on grid location

    """

    def __init__(self, cfg: ViTConfig) -> None:
        """Initialize ViT encoder.

        Args:
            cfg: ViT configuration.

        """
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.num_register_tokens = cfg.num_register_tokens
        self.use_cls_token = cfg.use_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            hidden_dim=cfg.hidden_dim,
        )

        # Special tokens
        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if cfg.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, cfg.num_register_tokens, cfg.hidden_dim)
            )
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # Compute FFN dimension
        ffn_dim = cfg.ffn_dim
        if ffn_dim is None:
            ffn_dim = int((8 * cfg.hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        # RoPE 2D
        head_dim = cfg.hidden_dim // cfg.num_heads
        rope_dim = cfg.rope_dim or head_dim
        rope_dim = (rope_dim // 4) * 4  # Must be divisible by 4
        self.rope_2d = RoPE2D(
            RoPE2DConfig(
                rope_dim=rope_dim,
                rope_theta=cfg.rope_theta,
                interleave=cfg.rope_interleave,
            )
        )

        # Build transformer blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(cfg.num_layers):
            # Determine if this layer uses MoE
            use_moe = cfg.use_moe and (layer_idx % cfg.moe_layer_freq == 0)

            # Build MoE config for this layer
            moe_config = None
            if use_moe:
                moe_config = cfg.moe_config or MoEConfig(
                    dim=cfg.hidden_dim,
                    ffn_dim=ffn_dim,
                    num_experts=8,
                    num_shared_experts=1,
                    top_k=2,
                    dropout=cfg.dropout,
                )

            # Build block with configurable attention type
            block_config = ViTBlockConfig(
                dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                ffn_dim=ffn_dim,
                dropout=cfg.dropout,
                attn_type=cfg.attn_type,
                rope_dim=rope_dim,
                rope_theta=cfg.rope_theta,
                rope_interleave=cfg.rope_interleave,
                use_moe=use_moe,
                moe_config=moe_config,
                mla_kv_lora_rank=cfg.mla_kv_lora_rank,
            )
            block = ViTBlock(block_config)
            self.blocks.append(block)

        self.norm = RMSNorm(cfg.hidden_dim)

    def _build_positions(
        self,
        num_patches_h: int,
        num_patches_w: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Build 2D position indices for all tokens.

        Token order: [CLS, Register_1, ..., Register_R, Patch_1, ..., Patch_N]

        Position encoding:
            - CLS: (0, 0)
            - Register tokens: (0, 1), (0, 2), ... (treated as virtual positions)
            - Patches: 2D grid positions starting from (1, 0)

        Args:
            num_patches_h: Number of patches in height.
            num_patches_w: Number of patches in width.
            device: Device for tensors.

        Returns:
            Tuple of (pos_h, pos_w) tensors, each shape (total_tokens,).

        """
        positions_h = []
        positions_w = []

        # CLS token at (0, 0)
        if self.use_cls_token:
            positions_h.append(torch.tensor([0], device=device))
            positions_w.append(torch.tensor([0], device=device))

        # Register tokens at (0, 1), (0, 2), ...
        if self.num_register_tokens > 0:
            reg_h = torch.zeros(self.num_register_tokens, device=device)
            reg_w = torch.arange(1, self.num_register_tokens + 1, device=device)
            positions_h.append(reg_h)
            positions_w.append(reg_w)

        # Patch tokens: 2D grid starting from row 1
        # Patches are in row-major order
        patch_h = torch.arange(num_patches_h, device=device).repeat_interleave(
            num_patches_w
        ) + 1  # Offset by 1 to leave room for CLS/register
        patch_w = torch.arange(num_patches_w, device=device).repeat(num_patches_h)
        positions_h.append(patch_h)
        positions_w.append(patch_w)

        pos_h = torch.cat(positions_h)
        pos_w = torch.cat(positions_w)

        return pos_h, pos_w

    def forward(
        self,
        x: Tensor,
        return_all_tokens: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input image, shape (B, C, H, W).
            return_all_tokens: If True, return all token embeddings.

        Returns:
            If pooling='cls': CLS token embedding, shape (B, D).
            If pooling='mean': Mean-pooled patch embeddings, shape (B, D).
            If pooling='all' or return_all_tokens: All tokens, shape (B, S, D).

        """
        B = x.shape[0]

        # Patch embedding
        patch_tokens, H, W = self.patch_embed(x)  # (B, N, D)

        # Build token sequence: [CLS, Registers, Patches]
        tokens = []

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens.append(cls_tokens)

        if self.num_register_tokens > 0:
            reg_tokens = self.register_tokens.expand(B, -1, -1)
            tokens.append(reg_tokens)

        tokens.append(patch_tokens)
        x = torch.cat(tokens, dim=1)  # (B, 1 + R + N, D)

        # Build position indices
        pos_h, pos_w = self._build_positions(H, W, device=x.device)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, pos_h=pos_h, pos_w=pos_w)

        x = self.norm(x)

        # Pooling
        if return_all_tokens or self.cfg.pooling == "all":
            return x

        num_prefix = (1 if self.use_cls_token else 0) + self.num_register_tokens

        if self.cfg.pooling == "cls" and self.use_cls_token:
            return x[:, 0]  # CLS token
        elif self.cfg.pooling == "mean":
            # Mean pool over patch tokens only (exclude CLS and registers)
            return x[:, num_prefix:].mean(dim=1)
        else:
            # Fallback to all tokens
            return x

    def get_aux_loss(self) -> Tensor:
        """Get total auxiliary loss from MoE layers."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            if hasattr(block, "get_aux_loss"):
                total_loss = total_loss + block.get_aux_loss()
        return total_loss

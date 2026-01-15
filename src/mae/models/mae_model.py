"""Masked Autoencoder (MAE) model implementation.

MAE is a self-supervised learning approach that masks random patches
of the input image and reconstructs the missing patches. This module
implements MAE using our modern ViT architecture with:

- 2D RoPE for positional encoding
- GQA/MLA for efficient attention
- Optional MoE for FFN layers
- Register tokens for improved representations
- Configurable resolution range support

The encoder processes only visible patches (75% masked by default),
making training efficient. The decoder reconstructs all patches
including masked ones.

Reference: https://arxiv.org/abs/2111.06377
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models import ViTConfig, ViTEncoder
from src.common.models.components import MultiHeadSelfAttention, RMSNorm, SwiGLU


@dataclass
class MAEConfig:
    """Configuration for Masked Autoencoder.

    Attributes:
        # Encoder (ViT)
        encoder_config: Configuration for the ViT encoder.

        # Decoder
        decoder_hidden_dim: Hidden dimension for decoder.
        decoder_num_layers: Number of decoder transformer layers.
        decoder_num_heads: Number of attention heads in decoder.
        decoder_num_kv_heads: Number of KV heads in decoder (for GQA).

        # Masking
        mask_ratio: Ratio of patches to mask (default 0.75).
        mask_token_init: How to initialize mask token ('zeros', 'normal').

        # Training
        norm_pix_loss: Normalize pixel values in loss computation.
        patch_size: Patch size (must match encoder).
        in_channels: Number of input channels.

        # Resolution range
        min_resolution: Minimum resolution for training.
        max_resolution: Maximum resolution for training.

    """

    # Encoder
    encoder_config: ViTConfig = field(default_factory=ViTConfig)

    # Decoder
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 8
    decoder_num_heads: int = 16
    decoder_num_kv_heads: int = 4

    # Masking
    mask_ratio: float = 0.75
    mask_token_init: Literal["zeros", "normal"] = "normal"

    # Training
    norm_pix_loss: bool = True
    patch_size: int = 16
    in_channels: int = 3

    # Resolution range (for variable resolution training)
    min_resolution: int = 160
    max_resolution: int = 320


class MAEDecoder(nn.Module):
    """MAE Decoder for patch reconstruction.

    Takes encoded visible patches + mask tokens and reconstructs
    all patches in pixel space.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        patch_size: int,
        in_channels: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize MAE decoder.

        Args:
            encoder_dim: Dimension of encoder output.
            decoder_dim: Hidden dimension of decoder.
            num_layers: Number of decoder layers.
            num_heads: Number of attention heads.
            num_kv_heads: Number of KV heads for GQA.
            patch_size: Size of each patch.
            in_channels: Number of input image channels.
            dropout: Dropout probability.

        """
        super().__init__()
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Project from encoder to decoder dimension
        self.embed_proj = nn.Linear(encoder_dim, decoder_dim, bias=True)

        # Decoder position embedding (learned)
        # We use a maximum expected number of patches
        self.max_patches = 256 * 256  # Support up to 4096x4096 images
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Decoder blocks (simpler than encoder, no MoE)
        ffn_dim = int((8 * decoder_dim) / 3)
        ffn_dim = (ffn_dim + 63) // 64 * 64

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = DecoderBlock(
                dim=decoder_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            self.blocks.append(block)

        self.norm = RMSNorm(decoder_dim)

        # Prediction head: reconstruct patch pixels
        self.pred = nn.Linear(
            decoder_dim,
            patch_size * patch_size * in_channels,
            bias=True,
        )

    def forward(
        self,
        x: Tensor,
        ids_restore: Tensor,
        num_patches_h: int,
        num_patches_w: int,
    ) -> Tensor:
        """Decode and reconstruct patches.

        Args:
            x: Encoded visible patches, shape (B, N_vis + 1, encoder_dim).
                Includes CLS token at position 0.
            ids_restore: Indices to restore full sequence, shape (B, N).
            num_patches_h: Number of patches in height.
            num_patches_w: Number of patches in width.

        Returns:
            Reconstructed patches, shape (B, N, P*P*C).
            Does NOT include CLS token.

        """
        B = x.shape[0]
        N = num_patches_h * num_patches_w

        # Project to decoder dimension
        x = self.embed_proj(x)

        # Append mask tokens
        # x currently has: [CLS, visible_patches]
        # We need: [CLS, all_patches] with mask tokens for missing ones
        num_vis = x.shape[1] - 1  # Exclude CLS
        num_mask = N - num_vis

        # Create mask tokens
        mask_tokens = self.mask_token.expand(B, num_mask, -1)

        # Split CLS and visible tokens
        cls_token = x[:, :1, :]  # (B, 1, D)
        vis_tokens = x[:, 1:, :]  # (B, N_vis, D)

        # Unshuffle: restore original order
        # ids_restore contains the original positions of patches
        x_full = torch.zeros(B, N, self.decoder_dim, device=x.device, dtype=x.dtype)

        # Scatter visible tokens and mask tokens to original positions
        # First, create combined tokens in shuffled order
        x_combined = torch.cat([vis_tokens, mask_tokens], dim=1)  # (B, N, D)

        # Restore to original spatial order
        ids_restore_expand = ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        x_full = torch.zeros_like(x_combined).scatter_(1, ids_restore_expand, x_combined)

        # Add CLS back and add positional embedding
        x = torch.cat([cls_token, x_full], dim=1)  # (B, N+1, D)

        # Add decoder positional embedding
        x = x + self.pos_embed[:, :N + 1, :]

        # Apply decoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Remove CLS token and predict pixels
        x = x[:, 1:, :]  # (B, N, D)
        x = self.pred(x)  # (B, N, P*P*C)

        return x

    @property
    def mask_token(self) -> Tensor:
        """Get mask token (lazy initialization)."""
        if not hasattr(self, "_mask_token"):
            self._mask_token = nn.Parameter(
                torch.zeros(1, 1, self.decoder_dim)
            )
            nn.init.normal_(self._mask_token, std=0.02)
        return self._mask_token


class DecoderBlock(nn.Module):
    """Simple decoder transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize decoder block."""
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            n_heads=num_heads,
            attn_dropout=dropout,
            rope_dim=0,
        )
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, ffn_dim)
        self.dropout = float(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = x + self.attn(self.attn_norm(x), start_pos=0, is_causal=False)
        x = x + torch.nn.functional.dropout(self.mlp(self.mlp_norm(x)), p=self.dropout, training=self.training)
        return x


class MAEModel(nn.Module):
    """Masked Autoencoder with Vision Transformer.

    Self-supervised pre-training approach that:
    1. Splits image into patches
    2. Randomly masks a large portion (75%) of patches
    3. Encodes only visible patches with ViT
    4. Decodes all patches (with mask tokens for missing ones)
    5. Reconstructs the original image

    The encoder can be used for downstream tasks after pre-training.
    """

    def __init__(self, cfg: MAEConfig) -> None:
        """Initialize MAE model.

        Args:
            cfg: MAE configuration.

        """
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.mask_ratio = cfg.mask_ratio
        self.norm_pix_loss = cfg.norm_pix_loss

        # Encoder (ViT)
        self.encoder = ViTEncoder(cfg.encoder_config)

        # Patch embedding (separate from encoder for masking)
        self.patch_embed = nn.Conv2d(
            cfg.in_channels,
            cfg.encoder_config.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

        # CLS token (shared with encoder)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, cfg.encoder_config.hidden_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Decoder
        self.decoder = MAEDecoder(
            encoder_dim=cfg.encoder_config.hidden_dim,
            decoder_dim=cfg.decoder_hidden_dim,
            num_layers=cfg.decoder_num_layers,
            num_heads=cfg.decoder_num_heads,
            num_kv_heads=cfg.decoder_num_kv_heads,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            dropout=cfg.encoder_config.dropout,
        )

        # Initialize mask token in decoder
        self._init_decoder_mask_token()

    def _init_decoder_mask_token(self) -> None:
        """Initialize decoder mask token."""
        mask_token = nn.Parameter(
            torch.zeros(1, 1, self.cfg.decoder_hidden_dim)
        )
        if self.cfg.mask_token_init == "normal":
            nn.init.normal_(mask_token, std=0.02)
        self.decoder._mask_token = mask_token

    def patchify(self, x: Tensor) -> Tensor:
        """Convert image to patches.

        Args:
            x: Image tensor, shape (B, C, H, W).

        Returns:
            Patches, shape (B, N, P*P*C).

        """
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, "Image size must be divisible by patch size"

        h = H // P
        w = W // P
        x = x.reshape(B, C, h, P, w, P)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, P, P, C)
        x = x.reshape(B, h * w, P * P * C)
        return x

    def unpatchify(self, x: Tensor, h: int, w: int) -> Tensor:
        """Convert patches back to image.

        Args:
            x: Patches, shape (B, N, P*P*C).
            h: Number of patches in height.
            w: Number of patches in width.

        Returns:
            Image tensor, shape (B, C, H, W).

        """
        B = x.shape[0]
        P = self.patch_size
        C = self.cfg.in_channels

        x = x.reshape(B, h, w, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, P, w, P)
        x = x.reshape(B, C, h * P, w * P)
        return x

    def random_masking(
        self,
        x: Tensor,
        mask_ratio: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Randomly mask patches.

        Args:
            x: Patch embeddings, shape (B, N, D).
            mask_ratio: Ratio of patches to mask.

        Returns:
            Tuple of:
                - x_masked: Visible patches only, shape (B, N_vis, D).
                - mask: Binary mask (1 = masked), shape (B, N).
                - ids_restore: Indices to restore original order, shape (B, N).

        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]

        # Gather visible patches
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Generate binary mask (1 = masked, 0 = visible)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        # Unshuffle mask to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: Tensor,
        mask_ratio: float,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        """Encode visible patches.

        Args:
            x: Input image, shape (B, C, H, W).
            mask_ratio: Ratio of patches to mask.

        Returns:
            Tuple of:
                - latent: Encoded visible patches, shape (B, N_vis + 1, D).
                - mask: Binary mask, shape (B, N).
                - ids_restore: Indices to restore order, shape (B, N).
                - num_patches_h: Number of patches in height.
                - num_patches_w: Number of patches in width.

        """
        # Patch embedding
        patches = self.patch_embed(x)  # (B, D, H/P, W/P)
        B, D, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)

        # Random masking
        x_masked, mask, ids_restore = self.random_masking(patches, mask_ratio)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_masked = torch.cat([cls_tokens, x_masked], dim=1)

        # Build position indices for visible patches
        # We need to track which patches are visible for RoPE
        # For simplicity, we use learned positional embedding in encoder
        # The encoder will handle positions internally

        # Apply encoder blocks
        # Note: We use a simplified forward that doesn't use encoder.forward
        # because we need to handle masking specially
        # Process through encoder blocks
        x = x_masked
        len_keep = x_masked.shape[1] - 1  # exclude CLS
        positions_2d = self._build_visible_positions_2d(ids_restore, len_keep, H, W, x.device)
        for block in self.encoder.blocks:
            x = block(x, positions_2d=positions_2d)
        x = self.encoder.norm(x)

        return x, mask, ids_restore, H, W

    def _build_visible_positions_2d(
        self,
        ids_restore: Tensor,
        len_keep: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> Tensor:
        """Build position indices for visible patches.

        Args:
            ids_restore: Indices to restore order, shape (B, N).
            len_keep: Number of visible patches per sample.
            H: Number of patches in height.
            W: Number of patches in width.
            device: Device for tensors.

        Returns:
            positions_2d: (B, 1 + N_vis, 2) integer (y,x) coordinates for CLS + visible patches.

        """
        B = ids_restore.shape[0]
        # ids_restore is the inverse permutation of ids_shuffle, so:
        # ids_shuffle = argsort(ids_restore)
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]  # (B, N_vis)

        patch_y = (ids_keep // W) + 1  # offset row 0 for CLS
        patch_x = ids_keep % W

        cls_y = torch.zeros((B, 1), device=device, dtype=torch.long)
        cls_x = torch.zeros((B, 1), device=device, dtype=torch.long)

        pos_y = torch.cat([cls_y, patch_y.to(torch.long)], dim=1)
        pos_x = torch.cat([cls_x, patch_x.to(torch.long)], dim=1)
        return torch.stack([pos_y, pos_x], dim=-1)

    def forward_decoder(
        self,
        x: Tensor,
        ids_restore: Tensor,
        num_patches_h: int,
        num_patches_w: int,
    ) -> Tensor:
        """Decode and reconstruct patches.

        Args:
            x: Encoded visible patches, shape (B, N_vis + 1, D).
            ids_restore: Indices to restore order.
            num_patches_h: Number of patches in height.
            num_patches_w: Number of patches in width.

        Returns:
            Reconstructed patches, shape (B, N, P*P*C).

        """
        return self.decoder(x, ids_restore, num_patches_h, num_patches_w)

    def forward_loss(
        self,
        images: Tensor,
        pred: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Compute reconstruction loss.

        Args:
            images: Original images, shape (B, C, H, W).
            pred: Predicted patches, shape (B, N, P*P*C).
            mask: Binary mask (1 = masked), shape (B, N).

        Returns:
            Mean loss over masked patches.

        """
        target = self.patchify(images)

        if self.norm_pix_loss:
            # Normalize target patches
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Per-patch loss

        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(
        self,
        images: Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            images: Input images, shape (B, C, H, W).
            mask_ratio: Optional override for mask ratio.

        Returns:
            Tuple of:
                - loss: Reconstruction loss.
                - pred: Predicted patches, shape (B, N, P*P*C).
                - mask: Binary mask, shape (B, N).

        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # Encode
        latent, mask, ids_restore, H, W = self.forward_encoder(images, mask_ratio)

        # Decode
        pred = self.forward_decoder(latent, ids_restore, H, W)

        # Compute loss
        loss = self.forward_loss(images, pred, mask)

        return loss, pred, mask

    def get_encoder(self) -> ViTEncoder:
        """Get the pre-trained encoder for downstream tasks."""
        return self.encoder

    @classmethod
    def from_config(cls, config: Any) -> MAEModel:
        """Create model from Hydra config.

        Args:
            config: Hydra configuration.

        Returns:
            Initialized MAE model.

        """
        model_cfg = config.get("model", config)

        # Build encoder config
        encoder_cfg = ViTConfig(
            patch_size=model_cfg.get("patch_size", 16),
            in_channels=model_cfg.get("in_channels", 3),
            max_resolution=model_cfg.get("max_resolution", 224),
            hidden_dim=model_cfg.get("hidden_dim", 768),
            num_layers=model_cfg.get("num_layers", 12),
            num_heads=model_cfg.get("num_heads", 12),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.0),
            num_register_tokens=model_cfg.get("num_register_tokens", 4),
            use_cls_token=True,
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            use_moe=model_cfg.get("use_moe", False),
            moe_layer_freq=model_cfg.get("moe_layer_freq", 2),
            pooling="all",
        )

        mae_cfg = MAEConfig(
            encoder_config=encoder_cfg,
            decoder_hidden_dim=model_cfg.get("decoder_hidden_dim", 512),
            decoder_num_layers=model_cfg.get("decoder_num_layers", 8),
            decoder_num_heads=model_cfg.get("decoder_num_heads", 16),
            decoder_num_kv_heads=model_cfg.get("decoder_num_kv_heads", 4),
            mask_ratio=model_cfg.get("mask_ratio", 0.75),
            norm_pix_loss=model_cfg.get("norm_pix_loss", True),
            patch_size=model_cfg.get("patch_size", 16),
            in_channels=model_cfg.get("in_channels", 3),
            min_resolution=model_cfg.get("min_resolution", 160),
            max_resolution=model_cfg.get("max_resolution", 320),
        )

        return cls(mae_cfg)

"""Masked Autoencoder (MAE) model implementation.

This implementation is based on the original MAE paper and official implementation:
- Paper: "Masked Autoencoders Are Scalable Vision Learners" (https://arxiv.org/abs/2111.06377)
- Official Code: https://github.com/facebookresearch/mae

The encoder uses ViTEncoder from src.utils.models.vit which supports:
- 2D RoPE for positional encoding
- Optional MoE for FFN layers
- Register tokens for improved representations

The decoder uses ViTBlock with 2D RoPE for positional encoding.

References:
    - He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
    - https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.models import ViTConfig, ViTEncoder
from src.utils.models.components import MoEConfig, RMSNorm, ViTBlock, ViTBlockConfig


@dataclass
class MAEConfig:
    """Configuration for Masked Autoencoder.

    Attributes:
        encoder_config: Configuration for the ViT encoder.
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth: Number of decoder transformer layers.
        decoder_num_heads: Number of attention heads in decoder.
        decoder_ffn_dim: Decoder MLP hidden dimension (defaults to 8/3 rule if None).
        decoder_dropout: Dropout probability for decoder attention/MLP.
        decoder_rope_dim: Rotary dimension per head for decoder 2D RoPE.
        decoder_rope_theta: Base theta for decoder 2D RoPE.
        norm_pix_loss: Whether to normalize pixel values in loss computation.
        mask_ratio: Ratio of patches to mask (default 0.75).

    """

    # Encoder (ViT)
    encoder_config: ViTConfig = field(default_factory=ViTConfig)

    # Decoder
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    decoder_ffn_dim: int | None = None
    decoder_dropout: float = 0.0
    decoder_rope_dim: int | None = None
    decoder_rope_theta: float = 1000.0

    # Training
    norm_pix_loss: bool = True
    mask_ratio: float = 0.75

    @property
    def patch_size(self) -> int:
        """Get patch size from encoder config."""
        return int(self.encoder_config.patch_size)

    @property
    def in_channels(self) -> int:
        """Get input channels from encoder config."""
        return int(self.encoder_config.in_channels)

    @property
    def img_size(self) -> int:
        """Get image size from encoder config."""
        return int(self.encoder_config.max_resolution)


class MAEModel(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone.

    The encoder is a ViTEncoder that learns visual representations.
    The decoder uses ViTBlock with 2D RoPE to reconstruct masked patches.

    Self-supervised pre-training approach that:
    1. Splits image into patches
    2. Randomly masks a large portion (75%) of patches
    3. Encodes visible patches only with ViT
    4. Decodes all patches with lightweight decoder
    5. Reconstructs the original image in pixel space

    Reference: https://github.com/facebookresearch/mae
    """

    def __init__(self, cfg: MAEConfig) -> None:
        """Initialize MAE model.

        Args:
            cfg: MAE configuration.

        """
        super().__init__()
        self.cfg = cfg
        self.norm_pix_loss = cfg.norm_pix_loss

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = ViTEncoder(cfg.encoder_config)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(
            cfg.encoder_config.hidden_dim, cfg.decoder_embed_dim, bias=True
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Compute decoder FFN dim
        decoder_ffn_dim = cfg.decoder_ffn_dim
        if decoder_ffn_dim is None:
            decoder_ffn_dim = int((8 * cfg.decoder_embed_dim) / 3)
            decoder_ffn_dim = (decoder_ffn_dim + 63) // 64 * 64

        # Compute decoder rope dim
        decoder_head_dim = cfg.decoder_embed_dim // cfg.decoder_num_heads
        decoder_rope_dim = cfg.decoder_rope_dim or decoder_head_dim
        decoder_rope_dim = min(int(decoder_rope_dim), int(decoder_head_dim))
        decoder_rope_dim = (decoder_rope_dim // 4) * 4  # 2D RoPE requires divisible by 4
        if decoder_rope_dim <= 0:
            decoder_rope_dim = 0

        self.decoder_blocks = nn.ModuleList([
            ViTBlock(ViTBlockConfig(
                dim=cfg.decoder_embed_dim,
                n_heads=cfg.decoder_num_heads,
                mlp_inter_dim=decoder_ffn_dim,
                attn_dropout=cfg.decoder_dropout,
                mlp_dropout=cfg.decoder_dropout,
                use_2d_rope=(decoder_rope_dim > 0),
                rope2d_frequency=cfg.decoder_rope_theta,
                rope_dim=decoder_rope_dim if decoder_rope_dim > 0 else None,
                use_moe=False,
            ))
            for _ in range(cfg.decoder_depth)
        ])

        self.decoder_norm = RMSNorm(cfg.decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            cfg.decoder_embed_dim,
            cfg.patch_size**2 * cfg.in_channels,
            bias=True,
        )
        # --------------------------------------------------------------------------

        # Cache for decoder 2D positions
        max_patches_h = cfg.img_size // cfg.patch_size
        max_patches_w = cfg.img_size // cfg.patch_size
        positions_2d_grid = self._build_positions_2d_grid(max_patches_h, max_patches_w)
        self.register_buffer("_decoder_positions_2d_grid", positions_2d_grid, persistent=False)

        self._initialize_decoder_weights()

    @staticmethod
    def _build_positions_2d_grid(num_patches_h: int, num_patches_w: int) -> Tensor:
        """Build a (H, W, 2) integer (y,x) position grid for patch tokens."""
        y = torch.arange(num_patches_h, dtype=torch.long)
        x = torch.arange(num_patches_w, dtype=torch.long)
        return torch.cartesian_prod(y, x).view(num_patches_h, num_patches_w, 2)

    def _slice_decoder_positions_2d(
        self, bsz: int, num_patches_h: int, num_patches_w: int, device: torch.device
    ) -> Tensor:
        """Slice cached (y,x) integer positions for decoder patch tokens."""
        patch_pos = self._decoder_positions_2d_grid[:num_patches_h, :num_patches_w].reshape(1, -1, 2)
        if patch_pos.device != device:
            patch_pos = patch_pos.to(device)
        return patch_pos.expand(bsz, -1, -1).contiguous()

    def _initialize_decoder_weights(self) -> None:
        """Initialize decoder weights."""
        # Initialize decoder linear layers
        for module in [self.decoder_embed, self.decoder_pred]:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def patchify(self, x: Tensor) -> Tensor:
        """Convert image to patches.

        Args:
            x: Image tensor, shape (B, C, H, W).

        Returns:
            Patches, shape (B, N, P*P*C).

        """
        B, C, H, W = x.shape
        P = self.cfg.patch_size
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
        P = self.cfg.patch_size
        C = self.cfg.in_channels

        x = x.reshape(B, h, w, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, P, w, P)
        x = x.reshape(B, C, h * P, w * P)
        return x

    def random_masking(
        self,
        x: Tensor,
        mask_ratio: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Randomly mask patches.

        Args:
            x: Patch embeddings, shape (B, N, D).
            mask_ratio: Ratio of patches to mask.

        Returns:
            Tuple of:
                - x_masked: Visible patches only, shape (B, N_vis, D).
                - mask: Binary mask (1 = masked), shape (B, N).
                - ids_restore: Indices to restore original order, shape (B, N).
                - ids_keep: Indices of kept patches, shape (B, N_vis).

        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))
        num_keep = max(1, min(N - 1, num_keep))

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

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(
        self,
        x: Tensor,
        mask_ratio: float,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        """Encode image with ViT encoder.

        Args:
            x: Input image, shape (B, C, H, W).
            mask_ratio: Ratio of patches to mask.

        Returns:
            Tuple of:
                - latent: Encoded patches (all tokens from ViT).
                - mask: Binary mask, shape (B, N).
                - ids_restore: Indices to restore order, shape (B, N).
                - num_patches_h: Number of patches in height.
                - num_patches_w: Number of patches in width.

        """
        B, C, H, W = x.shape
        num_patches_h = H // self.cfg.patch_size
        num_patches_w = W // self.cfg.patch_size
        N = num_patches_h * num_patches_w

        # Encode with ViT (mask patches before encoding)
        latent, h, w = self.encoder.patch_embed(x)  # (B, N, D), (H', W')
        
        # Apply masking to latent using random_masking method
        latent_masked, mask, ids_restore, ids_keep = self.random_masking(latent, mask_ratio)
        
        # Use masked latent for encoder (visible patches only)
        latent = latent_masked
        
        if self.encoder.cfg.use_cls_token:
            cls = self.encoder.cls_token.expand(B, -1, -1)
            latent = torch.cat([cls, latent], dim=1)
        if self.encoder.cfg.num_register_tokens > 0:
            reg = self.encoder.register_tokens.expand(B, -1, -1)
            insert_at = 1 if self.encoder.cfg.use_cls_token else 0
            latent = torch.cat([latent[:, :insert_at], reg, latent[:, insert_at:]], dim=1)

        # Build 2D positions for encoder
        # Get full grid positions from encoder
        positions_full = self.encoder._slice_positions_2d(
            B, h, w, x.device
        )  # (B, N, 2)

        # Gather positions for visible patches
        # ids_keep: (B, N_vis) -> expand to (B, N_vis, 2)
        positions_2d = torch.gather(
            positions_full,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, 2)
        )

        # Verify token count alignment (robustness check)
        expected_vis = latent.shape[1] - (
            int(self.encoder.cfg.use_cls_token) + self.encoder.cfg.num_register_tokens
        )
        if positions_2d.shape[1] != expected_vis:
            raise ValueError(
                f"Positions mismatch: positions_2d has {positions_2d.shape[1]} tokens, "
                f"but latent has {expected_vis} visible patch tokens."
            )

        # Apply encoder blocks with 2D RoPE
        for block in self.encoder.blocks:
            latent = block(latent, positions_2d=positions_2d, grid_hw=(h, w))

        latent = self.encoder.norm(latent)

        return latent, mask, ids_restore, num_patches_h, num_patches_w

    def forward_decoder(
        self,
        x: Tensor,
        ids_restore: Tensor,
        num_patches_h: int,
        num_patches_w: int,
    ) -> Tensor:
        """Decode and reconstruct patches.

        Args:
            x: Encoded patches from encoder (all tokens).
            ids_restore: Indices to restore order.
            num_patches_h: Number of patches in height.
            num_patches_w: Number of patches in width.

        Returns:
            Reconstructed patches, shape (B, N, P*P*C).

        """
        B = x.shape[0]

        # Determine number of special tokens (CLS + register tokens)
        enc_cfg = self.encoder.cfg
        num_special = int(enc_cfg.use_cls_token) + int(enc_cfg.num_register_tokens)

        # Extract patch tokens only (skip CLS and register tokens)
        patch_tokens = x[:, num_special:, :]  # (B, N_vis, D_enc)

        # Project to decoder dimension
        patch_tokens = self.decoder_embed(patch_tokens)  # (B, N_vis, D_dec)

        # Append mask tokens to sequence
        N_total = num_patches_h * num_patches_w
        N_vis = patch_tokens.shape[1]
        
        mask_tokens = self.mask_token.expand(B, N_total - N_vis, -1)
        patch_tokens = torch.cat([patch_tokens, mask_tokens], dim=1)  # (B, N_total, D_dec)

        # Unshuffle to restore original order
        patch_tokens = torch.gather(
            patch_tokens, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        )

        # Build 2D positions for decoder
        positions_2d = self._slice_decoder_positions_2d(
            B, num_patches_h, num_patches_w, x.device
        )

        # Apply decoder blocks with 2D RoPE
        for blk in self.decoder_blocks:
            patch_tokens = blk(
                patch_tokens,
                positions_2d=positions_2d,
                grid_hw=(num_patches_h, num_patches_w),
            )

        patch_tokens = self.decoder_norm(patch_tokens)

        # Predict pixels
        pred = self.decoder_pred(patch_tokens)  # (B, N, P*P*C)

        return pred

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
        denom = mask.sum().clamp_min(1.0)
        loss = (loss * mask).sum() / denom

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
            mask_ratio = self.cfg.mask_ratio

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
            use_cls_token=model_cfg.get("use_cls_token", True),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 1000.0),
            use_moe=model_cfg.get("use_moe", False),
            moe_layer_freq=model_cfg.get("moe_layer_freq", 2),
            pooling="all",
        )

        # Handle MoE config if provided
        moe_config = None
        if model_cfg.get("use_moe", False) and "moe_config" in model_cfg:
            moe_cfg_dict = model_cfg["moe_config"]
            moe_config = MoEConfig(
                dim=encoder_cfg.hidden_dim,
                moe_inter_dim=moe_cfg_dict.get("moe_inter_dim", encoder_cfg.ffn_dim),
                n_routed_experts=moe_cfg_dict.get("n_routed_experts", 8),
                n_shared_experts=moe_cfg_dict.get("n_shared_experts", 1),
                n_activated_experts=moe_cfg_dict.get("n_activated_experts", 2),
            )
            encoder_cfg.moe_config = moe_config

        mae_cfg = MAEConfig(
            encoder_config=encoder_cfg,
            decoder_embed_dim=model_cfg.get("decoder_embed_dim", 512),
            decoder_depth=model_cfg.get("decoder_depth", 8),
            decoder_num_heads=model_cfg.get("decoder_num_heads", 16),
            decoder_ffn_dim=model_cfg.get("decoder_ffn_dim", None),
            decoder_dropout=model_cfg.get("decoder_dropout", 0.0),
            decoder_rope_dim=model_cfg.get("decoder_rope_dim", None),
            decoder_rope_theta=model_cfg.get("decoder_rope_theta", 1000.0),
            mask_ratio=model_cfg.get("mask_ratio", 0.75),
            norm_pix_loss=model_cfg.get("norm_pix_loss", True),
        )

        return cls(mae_cfg)
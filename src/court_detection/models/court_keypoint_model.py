"""Court keypoint detection model.

Detects 20 court keypoints (CourtKP20) from tennis images using a
ViT encoder + lightweight decoder that predicts per-keypoint heatmaps.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.common.models import ViTConfig, ViTEncoder
from src.common.models.components import MoEConfig, RMSNorm, ViTBlock, ViTBlockConfig

NUM_KEYPOINTS = 20


class CourtKeypointModel(nn.Module):
    """Court keypoint detection model using a ViT encoder/decoder.

    Args:
        config: Model configuration dict (encoder/decoder settings).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.config = config
        self.input_size = tuple(config.get("input_size", [256, 256]))
        self.heatmap_size = tuple(config.get("heatmap_size", [64, 64]))
        self.num_keypoints = int(config.get("num_keypoints", NUM_KEYPOINTS))

        encoder_cfg = config.get("encoder", {})
        max_resolution = int(encoder_cfg.get("max_resolution", max(self.input_size)))
        encoder_config = ViTConfig(
            patch_size=encoder_cfg.get("patch_size", 16),
            in_channels=encoder_cfg.get("in_channels", 3),
            max_resolution=max_resolution,
            hidden_dim=encoder_cfg.get("hidden_dim", 768),
            num_layers=encoder_cfg.get("num_layers", 12),
            num_heads=encoder_cfg.get("num_heads", 12),
            ffn_dim=encoder_cfg.get("ffn_dim", None),
            dropout=encoder_cfg.get("dropout", 0.1),
            num_register_tokens=encoder_cfg.get("num_register_tokens", 4),
            use_cls_token=encoder_cfg.get("use_cls_token", True),
            rope_dim=encoder_cfg.get("rope_dim", None),
            rope_theta=encoder_cfg.get("rope_theta", 1000.0),
            use_moe=encoder_cfg.get("use_moe", False),
            moe_layer_freq=encoder_cfg.get("moe_layer_freq", 2),
            pooling="all",
        )

        if encoder_cfg.get("use_moe", False) and "moe_config" in encoder_cfg:
            moe_cfg = encoder_cfg["moe_config"]
            encoder_config.moe_config = MoEConfig(
                dim=encoder_config.hidden_dim,
                moe_inter_dim=moe_cfg.get("moe_inter_dim", encoder_config.ffn_dim),
                n_routed_experts=moe_cfg.get("n_routed_experts", 8),
                n_shared_experts=moe_cfg.get("n_shared_experts", 1),
                n_activated_experts=moe_cfg.get("n_activated_experts", 2),
            )

        self.encoder = ViTEncoder(encoder_config)

        decoder_cfg = config.get("decoder", {})
        decoder_embed_dim = int(decoder_cfg.get("embed_dim", 512))
        decoder_depth = int(decoder_cfg.get("depth", 4))
        decoder_num_heads = int(decoder_cfg.get("num_heads", 8))
        decoder_dropout = float(decoder_cfg.get("dropout", 0.0))
        decoder_rope_theta = float(decoder_cfg.get("rope_theta", 1000.0))

        decoder_ffn_dim = decoder_cfg.get("ffn_dim")
        if decoder_ffn_dim is None:
            decoder_ffn_dim = int((8 * decoder_embed_dim) / 3)
            decoder_ffn_dim = (decoder_ffn_dim + 63) // 64 * 64

        decoder_head_dim = decoder_embed_dim // decoder_num_heads
        decoder_rope_dim = decoder_cfg.get("rope_dim") or decoder_head_dim
        decoder_rope_dim = min(int(decoder_rope_dim), int(decoder_head_dim))
        decoder_rope_dim = (decoder_rope_dim // 4) * 4
        if decoder_rope_dim <= 0:
            decoder_rope_dim = 0

        self.decoder_embed = nn.Linear(
            encoder_config.hidden_dim,
            decoder_embed_dim,
            bias=True,
        )
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(ViTBlockConfig(
                dim=decoder_embed_dim,
                n_heads=decoder_num_heads,
                mlp_inter_dim=decoder_ffn_dim,
                attn_dropout=decoder_dropout,
                mlp_dropout=decoder_dropout,
                use_2d_rope=(decoder_rope_dim > 0),
                rope2d_frequency=decoder_rope_theta,
                rope_dim=decoder_rope_dim if decoder_rope_dim > 0 else None,
                use_moe=False,
            ))
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = RMSNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            encoder_config.patch_size**2 * self.num_keypoints,
            bias=True,
        )

        vis_cfg = config.get("visibility_head", {})
        vis_hidden_dim = int(vis_cfg.get("hidden_dim", 256))
        vis_dropout = float(vis_cfg.get("dropout", 0.2))
        self.visibility_head = nn.Sequential(
            nn.Linear(encoder_config.hidden_dim, vis_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(vis_dropout),
            nn.Linear(vis_hidden_dim, self.num_keypoints),
        )

        max_patches_h = max_resolution // encoder_config.patch_size
        max_patches_w = max_resolution // encoder_config.patch_size
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
        self,
        bsz: int,
        num_patches_h: int,
        num_patches_w: int,
        device: torch.device,
    ) -> Tensor:
        patch_pos = self._decoder_positions_2d_grid[:num_patches_h, :num_patches_w].reshape(1, -1, 2)
        if patch_pos.device != device:
            patch_pos = patch_pos.to(device)
        return patch_pos.expand(bsz, -1, -1).contiguous()

    def _initialize_decoder_weights(self) -> None:
        """Initialize decoder weights."""
        for module in [self.decoder_embed, self.decoder_pred]:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _forward_encoder(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Encode image into patch tokens."""
        bsz, _, h_img, w_img = x.shape
        patch_size = int(self.encoder.cfg.patch_size)
        if h_img % patch_size != 0 or w_img % patch_size != 0:
            raise ValueError("Input size must be divisible by patch_size.")

        tok, h, w = self.encoder.patch_embed(x)

        if self.encoder.cfg.use_cls_token:
            cls = self.encoder.cls_token.expand(bsz, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
        if self.encoder.cfg.num_register_tokens > 0:
            reg = self.encoder.register_tokens.expand(bsz, -1, -1)
            insert_at = 1 if self.encoder.cfg.use_cls_token else 0
            tok = torch.cat([tok[:, :insert_at], reg, tok[:, insert_at:]], dim=1)

        positions_2d = self.encoder._slice_positions_2d(bsz, h, w, x.device)
        for block in self.encoder.blocks:
            tok = block(tok, positions_2d=positions_2d, grid_hw=(h, w))
        tok = self.encoder.norm(tok)

        return tok, h, w

    def _extract_patch_tokens(self, tok: Tensor) -> Tensor:
        num_special = int(self.encoder.cfg.use_cls_token) + int(self.encoder.cfg.num_register_tokens)
        return tok[:, num_special:, :]

    def _pool_visibility_features(self, tok: Tensor) -> Tensor:
        if self.encoder.cfg.use_cls_token:
            return tok[:, 0]
        patch_tokens = self._extract_patch_tokens(tok)
        return patch_tokens.mean(dim=1)

    def _unpatchify(self, x: Tensor, h: int, w: int) -> Tensor:
        """Convert patch predictions to image-like heatmaps."""
        bsz = x.shape[0]
        patch_size = int(self.encoder.cfg.patch_size)
        channels = self.num_keypoints
        x = x.reshape(bsz, h, w, patch_size, patch_size, channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(bsz, channels, h * patch_size, w * patch_size)

    def _forward_decoder(self, tok: Tensor, h: int, w: int) -> Tensor:
        patch_tokens = self._extract_patch_tokens(tok)
        patch_tokens = self.decoder_embed(patch_tokens)

        positions_2d = self._slice_decoder_positions_2d(patch_tokens.shape[0], h, w, tok.device)
        for blk in self.decoder_blocks:
            patch_tokens = blk(
                patch_tokens,
                positions_2d=positions_2d,
                grid_hw=(h, w),
            )

        patch_tokens = self.decoder_norm(patch_tokens)
        pred = self.decoder_pred(patch_tokens)
        heatmaps = self._unpatchify(pred, h, w)

        if (heatmaps.shape[2], heatmaps.shape[3]) != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=self.heatmap_size,
                mode="bilinear",
                align_corners=False,
            )
        return heatmaps

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input images, shape (B, 3, H, W).

        Returns:
            Dictionary with:
                - 'heatmaps': Predicted heatmaps, shape (B, K, Hm, Wm)
                - 'visibility': Visibility logits, shape (B, K)
                - 'keypoints': Predicted keypoint coordinates, shape (B, K, 2)
        """
        tokens, h, w = self._forward_encoder(x)
        heatmaps = self._forward_decoder(tokens, h, w)
        visibility = self.visibility_head(self._pool_visibility_features(tokens))
        keypoints = self._heatmaps_to_coords(heatmaps)

        return {
            "heatmaps": heatmaps,
            "visibility": visibility,
            "keypoints": keypoints,
        }

    def _heatmaps_to_coords(self, heatmaps: Tensor) -> Tensor:
        """Convert heatmaps to keypoint coordinates using soft-argmax.

        Args:
            heatmaps: Heatmaps of shape (B, K, H, W).

        Returns:
            Coordinates of shape (B, K, 2) in normalized [0, 1] range.
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device

        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, K, -1)

        # Apply softmax to get probability distribution
        probs = F.softmax(heatmaps_flat, dim=-1)

        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Flatten coordinate grids
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        # Compute expected coordinates (soft-argmax)
        x = (probs * xx_flat.view(1, 1, -1)).sum(dim=-1)
        y = (probs * yy_flat.view(1, 1, -1)).sum(dim=-1)

        # Stack to (B, K, 2)
        coords = torch.stack([x, y], dim=-1)

        return coords

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict keypoints and visibility.

        Args:
            x: Input images, shape (B, 3, H, W).

        Returns:
            Tuple of:
                - keypoints: Predicted coordinates in pixel space, shape (B, K, 2)
                - visibility: Visibility probabilities, shape (B, K)
        """
        output = self.forward(x)

        keypoints = output["keypoints"].clone()
        keypoints[..., 0] *= self.input_size[1]
        keypoints[..., 1] *= self.input_size[0]

        visibility = torch.sigmoid(output["visibility"])

        return keypoints, visibility


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_cfg = {
        "input_size": [64, 64],
        "heatmap_size": [32, 32],
        "num_keypoints": 20,
        "encoder": {
            "patch_size": 8,
            "in_channels": 3,
            "max_resolution": 64,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.0,
            "num_register_tokens": 0,
            "use_cls_token": True,
            "use_moe": False,
        },
        "decoder": {
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 4,
            "dropout": 0.0,
        },
        "visibility_head": {
            "hidden_dim": 32,
            "dropout": 0.0,
        },
    }
    demo_model = CourtKeypointModel(demo_cfg)
    demo_input = torch.zeros(1, 3, 64, 64)
    demo_output = demo_model(demo_input)
    assert demo_output["heatmaps"].shape == (1, 20, 32, 32)
    assert demo_output["keypoints"].shape == (1, 20, 2)

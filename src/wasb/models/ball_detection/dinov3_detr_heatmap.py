"""DINOv3 DETR-style heatmap model for patch-token inputs (cross-attn per time + temporal self-attn).

Design:
- tokens: [B, T, N, C]
- Query:  [B*T, 1, C]
For each layer:
  1) Cross-attn: Q attends to per-frame patch tokens (K/V) -> [B*T, 1, C]
  2) Temporal self-attn: reshape Q to [B, T, C] and attend over time -> [B, T, C]
Finally:
- Heatmap scores per patch: dot(tokens[b,t,n], q[b,t]) -> [B, T, N]
- Reshape to grid [H, W] and upsample to output_hw by bilinear interpolation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from third_party.dinov3.api.dinov3_loader import get_dinov3_vits16

LOGGER = logging.getLogger(__name__)


class CrossAttentionBlock(nn.Module):
    """Cross-attention (no self-attn) + FFN block (Pre-LN).

    This is closer to the intended design than TransformerDecoderLayer because we
    do NOT include decoder self-attn (which is meaningless when query length is 1).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """Forward.

        Args:
            q:  [B, Lq, C] (here Lq=1)
            kv: [B, Lm, C] (memory tokens)

        Returns:
            Updated q: [B, Lq, C]
        """
        # Pre-LN cross-attn
        q_ln = self.norm_q(q)
        kv_ln = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_ln, kv_ln, kv_ln, need_weights=False)
        q = q + self.drop_attn(attn_out)

        # Pre-LN FFN
        ffn_out = self.ffn(self.norm_ffn(q))
        q = q + self.drop_ffn(ffn_out)
        return q


class DinoV3DETRHeatmap(nn.Module):
    """DETR-style heatmap model using DINOv3 patch tokens as K/V, with temporal query attention."""

    def __init__(
        self,
        cfg: DictConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = {}
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.config = cfg

        # Backbone (optional)
        self.use_backbone = bool(cfg.get("use_backbone", False))
        if self.use_backbone:
            checkpoint_path = cfg.get("backbone_checkpoint", None)
            self.backbone = get_dinov3_vits16(checkpoint_path=checkpoint_path)
        else:
            self.backbone = None

        # Embedding dim
        self.embed_dim = int(cfg.get("embed_dim", 384))
        if self.backbone is not None and hasattr(self.backbone, "embed_dim"):
            self.embed_dim = int(self.backbone.embed_dim)

        # Single learnable query token (expanded across B*T)
        self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        # Layers
        self.num_layers = int(cfg.get("num_layers", 4))
        num_heads = int(cfg.get("num_heads", 6))
        ffn_dim = int(cfg.get("ffn_dim", self.embed_dim * 4))
        dropout = float(cfg.get("dropout", 0.1))

        # Cross-attn blocks (per-frame, no self-attn)
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Temporal self-attn blocks (over time T)
        self.temporal_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,  # Pre-LN (PyTorch 2.x+)
                )
                for _ in range(self.num_layers)
            ]
        )

        # Time positional embedding (learned)
        self.max_frames = int(cfg.get("max_frames", 256))
        self.time_pos = nn.Parameter(torch.randn(1, self.max_frames, self.embed_dim) * 0.02)

        # Optional: causal temporal attention
        self.temporal_causal = bool(cfg.get("temporal_causal", False))

        # Patch/grid & output
        self.patch_size = int(cfg.get("patch_size", 16))
        self.patch_grid_hw = cfg.get("patch_grid_hw", None)
        if self.patch_grid_hw is None:
            raise ValueError("patch_grid_hw must be provided.")
        self.output_hw = cfg.get("heatmap_hw", None)
        if self.output_hw is None:
            self.output_hw = self.patch_grid_hw
        self.frames_out = int(cfg.get("frames_out", 1))

    def forward(self, frames: Tensor) -> Tensor:
        """Forward pass.

        Args:
            frames:
              - If backbone is None: expected tokens [B, T, N, C] (or [B, N, C] / [B,C,H,W] etc. per _resolve_tokens)
              - If backbone exists: image frames [B, C, H, W] or [B, T, C, H, W]

        Returns:
            Heatmaps: [B, frames_out, H_out, W_out]
        """
        tokens = self._resolve_tokens(frames)
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens [B, T, N, C], got {tuple(tokens.shape)}")
        b, t, n, c = tokens.shape
        if c != self.embed_dim:
            raise ValueError(f"Token dim C={c} != embed_dim={self.embed_dim}")

        if t > self.max_frames:
            raise ValueError(f"T={t} exceeds max_frames={self.max_frames}. Set cfg.max_frames larger.")

        # Prepare per-frame memory and queries
        tokens_bt = tokens.reshape(b * t, n, c)             # [B*T, N, C]
        q_bt = self.query.expand(b * t, 1, c).contiguous()  # [B*T, 1, C]

        # Temporal attention mask (optional causal)
        temporal_mask: Tensor | None = None
        if self.temporal_causal:
            # True = masked (disallow attending to future)
            temporal_mask = torch.triu(torch.ones(t, t, device=tokens.device, dtype=torch.bool), diagonal=1)

        # Alternate: cross-attn (per time) -> temporal self-attn (across time)
        for cross_blk, temp_blk in zip(self.cross_blocks, self.temporal_blocks):
            # (1) Cross-attn per-frame: [B*T,1,C] attends to [B*T,N,C]
            q_bt = cross_blk(q_bt, tokens_bt)  # [B*T, 1, C]

            # (2) Temporal self-attn over [B,T,C]
            q = q_bt.squeeze(1).reshape(b, t, c)                 # [B, T, C]
            q = q + self.time_pos[:, :t, :]                      # add time positional encoding
            q = temp_blk(q, src_mask=temporal_mask)              # [B, T, C]

            # Back to [B*T,1,C] for next cross step
            q_bt = q.reshape(b * t, 1, c)

        # Final per-time query after last temporal block
        q_final = q_bt.squeeze(1).reshape(b, t, c)  # [B, T, C]

        # Heatmap scores: [B,T,N]
        scores = torch.einsum("btnc,btc->btn", tokens, q_final) / math.sqrt(c)

        grid_h, grid_w = int(self.patch_grid_hw[0]), int(self.patch_grid_hw[1])
        if n != grid_h * grid_w:
            raise ValueError(f"N={n} must equal grid_h*grid_w={grid_h*grid_w} for view(H,W).")

        score_map = scores.reshape(b * t, 1, grid_h, grid_w)

        output_hw = (int(self.output_hw[0]), int(self.output_hw[1]))
        if output_hw != (grid_h, grid_w):
            score_map = F.interpolate(
                score_map,
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            )

        score_map = score_map.reshape(b, t, 1, output_hw[0], output_hw[1])
        return score_map[:, -self.frames_out :, 0]  # [B, frames_out, H_out, W_out]

    def _resolve_tokens(self, frames: Tensor) -> Tensor:
        """Resolve input to tokens [B, T, N, C]."""
        if self.backbone is None:
            # Expect already tokenized input
            # - [B, T, N, C] passes through
            # - [B, N, C] -> [B, 1, N, C]
            if frames.dim() == 3:
                return frames.unsqueeze(1)
            return frames

        # Backbone path: frames are images
        if frames.dim() == 4:
            # [B, C, H, W] -> tokens [B, 1, N, C]
            tokens = self._encode_frames(frames)
            return tokens.unsqueeze(1)

        if frames.dim() == 5:
            # [B, T, C, H, W] -> tokens [B, T, N, C]
            b, t, c, h, w = frames.shape
            frames_bt = frames.reshape(b * t, c, h, w)
            tokens_bt = self._encode_frames(frames_bt)  # [B*T, N, C]
            return tokens_bt.reshape(b, t, tokens_bt.shape[1], tokens_bt.shape[2])

        return frames

    def _encode_frames(self, frames: Tensor) -> Tensor:
        """Encode images to patch tokens [B, N, C] using DINOv3 backbone."""
        if self.backbone is None:
            raise RuntimeError("Backbone is not initialized for image input.")
        outputs = self.backbone.get_intermediate_layers(
            frames,
            n=1,
            reshape=False,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        tokens = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
        if tokens.dim() != 3:
            raise ValueError(f"Expected patch tokens [B, N, C], got {tuple(tokens.shape)}")
        return tokens

    def load_backbone_checkpoint(
        self,
        checkpoint_path: str | Path,
        map_location: torch.device | str | None = "cpu",
    ) -> None:
        """Load pre-trained DINOv3 weights into the backbone."""
        _ = map_location
        if self.backbone is None:
            raise RuntimeError("Backbone is not initialized.")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        if checkpoint_path.is_dir():
            LOGGER.info(
                "Backbone checkpoint path %s is a directory; skipping weight loading.",
                checkpoint_path,
            )
            return

        loaded_backbone = get_dinov3_vits16(checkpoint_path=str(checkpoint_path))
        self.backbone.load_state_dict(loaded_backbone.state_dict(), strict=True)
        LOGGER.info("Loaded DINOv3 backbone parameters from %s", checkpoint_path)

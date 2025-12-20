"""DINOv3 DETR-style heatmap model for patch-token inputs."""

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


class DinoV3DETRHeatmap(nn.Module):
    """DETR-style heatmap model using DINOv3 patch tokens as K/V."""

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

        self.use_backbone = bool(cfg.get("use_backbone", False))
        if self.use_backbone:
            checkpoint_path = cfg.get("backbone_checkpoint", None)
            self.backbone = get_dinov3_vits16(checkpoint_path=checkpoint_path)
        else:
            self.backbone = None

        self.embed_dim = int(cfg.get("embed_dim", 384))
        if self.backbone is not None and hasattr(self.backbone, "embed_dim"):
            self.embed_dim = int(getattr(self.backbone, "embed_dim"))

        self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        self.num_layers = int(cfg.get("num_layers", 4))
        num_heads = int(cfg.get("num_heads", 6))
        ffn_dim = int(cfg.get("ffn_dim", self.embed_dim * 4))
        dropout = float(cfg.get("dropout", 0.1))
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.patch_size = int(cfg.get("patch_size", 16))
        self.patch_grid_hw = cfg.get("patch_grid_hw", None)
        if self.patch_grid_hw is None:
            raise ValueError("patch_grid_hw must be provided.")
        self.output_hw = cfg.get("heatmap_hw", None)
        if self.output_hw is None:
            self.output_hw = self.patch_grid_hw
        self.frames_out = int(cfg.get("frames_out", 1))

    def forward(self, frames: Tensor) -> Tensor:
        """Forward pass using patch tokens as key/value for cross-attention."""
        tokens = self._resolve_tokens(frames)
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens [B, T, N, C], got {tuple(tokens.shape)}")
        b, t, n, c = tokens.shape

        tokens_bt = tokens.view(b * t, n, c)
        query = self.query.expand(b * t, -1, -1)
        for layer in self.cross_attn_layers:
            query = layer(query, tokens_bt)

        scores = (tokens_bt * query).sum(dim=-1) / math.sqrt(c)
        grid_h, grid_w = int(self.patch_grid_hw[0]), int(self.patch_grid_hw[1])
        score_map = scores.view(b * t, 1, grid_h, grid_w)

        output_hw = (int(self.output_hw[0]), int(self.output_hw[1]))
        if output_hw != (grid_h, grid_w):
            score_map = F.interpolate(
                score_map,
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            )

        score_map = score_map.view(b, t, 1, output_hw[0], output_hw[1])
        return score_map[:, -self.frames_out :, 0]

    def _resolve_tokens(self, frames: Tensor) -> Tensor:
        if self.backbone is None:
            return frames

        if frames.dim() == 4:
            tokens = self._encode_frames(frames)
            return tokens.unsqueeze(1)
        if frames.dim() == 5:
            b, t, c, h, w = frames.shape
            frames_bt = frames.view(b * t, c, h, w)
            tokens = self._encode_frames(frames_bt)
            return tokens.view(b, t, tokens.shape[1], tokens.shape[2])
        return frames

    def _encode_frames(self, frames: Tensor) -> Tensor:
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

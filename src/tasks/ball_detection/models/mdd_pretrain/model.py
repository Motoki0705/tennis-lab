from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.ball_detection.models.mdd_pose.config import MDDPoseConfig
from src.tasks.ball_detection.models.mdd_pose.query import QueryFusionBlock
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.utils.models.components.ffn_layers import SwiGLU
from src.utils.models.dpt import DPTDecoder

from .config import MDDPretrainConfig
from .encoder import DeepMDDEncoder


class MDDDPTDetector(nn.Module):
    """RGB -> fixed FP32 MDD -> deep CNN -> frame-local DPT -> /4 heatmap logits."""
    def __init__(self, config: MDDPretrainConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        super().__init__()
        self.config = config
        self.mdd = RGBToMDD(mdd_a, mdd_b)
        self.encoder = DeepMDDEncoder(config)
        self.decoder = DPTDecoder(encoder_channels=self.encoder.feature_channels,
                                  decoder_channels=config.decoder_channels, reassemble_factors=(1., 1., 1., 1.),
                                  block_style="residual")
        self.head = nn.Sequential(nn.Conv2d(config.decoder_channels, config.decoder_channels // 2, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(config.decoder_channels // 2, 1, 1))

    def forward(self, rgb: Tensor) -> Tensor:
        features = self.encoder(self.mdd(rgb))
        b, t = rgb.shape[:2]
        fused = self.decoder([f.flatten(0, 1) for f in features])
        # Produce logits on the declared /4 target grid; do not upscale the target to RGB size.
        logits = self.head(F.interpolate(fused, size=((rgb.shape[-2] + 3) // 4, (rgb.shape[-1] + 3) // 4),
                                         mode="bilinear", align_corners=False))
        return cast(Tensor, logits.reshape(b, t, *logits.shape[-2:]))


class DeepMDDQueryDetector(nn.Module):
    """Transfer target; initialized query decoder with the exact pretrained CNN topology."""
    def __init__(self, config: MDDPretrainConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        super().__init__()
        self.config = config
        self.mdd = RGBToMDD(mdd_a, mdd_b)
        self.encoder = DeepMDDEncoder(config)
        self.projection = nn.Linear(config.mixed_channels[-1], config.dim)
        self.spatial_position = nn.Linear(2, config.dim)
        self.ball_query = nn.Parameter(torch.randn(config.dim) * .02)
        query_config = MDDPoseConfig("conv2d", None, "query_only", 32, config.stem_channels,
                                    config.mixed_channels, config.dim, config.heads, config.layers,
                                    config.dropout, config.rope_base)
        self.blocks = nn.ModuleList(QueryFusionBlock(query_config, ffn=nn.Sequential(
            SwiGLU(config.dim, config.ffn_dim), nn.Dropout(config.dropout))) for _ in range(config.layers))
        self.head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 2))
        self.encoder_frozen = False

    def freeze_encoder(self, frozen: bool) -> None:
        self.encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)

    def train(self, mode: bool = True) -> DeepMDDQueryDetector:
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, rgb: Tensor, timestamps: Tensor) -> Tensor:
        feature = self.encoder(self.mdd(rgb))[-1]
        b, t, _, h, w = feature.shape
        patches = self.projection(feature.flatten(-2).transpose(-1, -2))
        yy = torch.arange(h, device=feature.device, dtype=feature.dtype) * 64
        xx = torch.arange(w, device=feature.device, dtype=feature.dtype) * 64
        yy = (yy + (yy + 63).clamp_max(rgb.shape[-2] - 1)) / (2 * (rgb.shape[-2] - 1))
        xx = (xx + (xx + 63).clamp_max(rgb.shape[-1] - 1)) / (2 * (rgb.shape[-1] - 1))
        y, x = torch.meshgrid(yy, xx, indexing="ij")
        patches = patches + self.spatial_position(torch.stack((x, y), -1).reshape(h * w, 2))[None, None]
        query = self.ball_query.expand(b, t, -1)
        for block in self.blocks:
            query = block(query, patches, timestamps - timestamps[:, :1])
        return cast(Tensor, self.head(query).float().sigmoid())

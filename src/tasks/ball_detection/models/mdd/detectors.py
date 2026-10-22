from __future__ import annotations

from typing import Self, cast

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd.attention import FusionBlock, QueryFusionBlock
from src.tasks.ball_detection.models.mdd.cnn_encoder import DeepMDDEncoder
from src.tasks.ball_detection.models.mdd.config import MDDPoseConfig, MDDPretrainConfig
from src.tasks.ball_detection.models.mdd.execution import (
    checkpoint_call,
    decode_frame_heatmaps,
)
from src.tasks.ball_detection.models.mdd.pose import PoseTokenizer
from src.tasks.ball_detection.models.mdd.token_encoder import MDDTokenEncoder
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.tasks.ball_detection.preprocessing.rgb_mdd import build_input_transform
from src.utils.models.components.ffn_layers import SwiGLU
from src.utils.models.dpt import DPTDecoder


class MDDPoseDetector(nn.Module):
    """RGB uint8 B,32,3,H,W; pose B,32,N,17,2; output B,32,2.

    Mask/PTS are metadata, not additional visual features. No pretrained modules,
    RGB branch, heatmap fabrication or automatic refiner conversion is provided.
    """

    def __init__(self, config: MDDPoseConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        super().__init__()
        if not config.requires_pose or config.pose_pooling is None:
            raise ValueError("MDDPoseDetector requires pose; use MDDQueryDetector for query_only")
        self.config = config
        self.mdd = RGBToMDD(mdd_a, mdd_b)
        self.encoder = MDDTokenEncoder(config.compression, config.stem_channels, config.mixed_channels, config.dim)
        self.pose = PoseTokenizer(config.pose_pooling, config.dim, config.heads)
        if config.readout == "query":
            self.ball_query = nn.Parameter(torch.randn(config.dim) * .02)
        self.blocks = nn.ModuleList(FusionBlock(config) for _ in range(config.layers))
        self.head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 2))

    def forward(self, rgb: Tensor, coordinates: Tensor, valid: Tensor, timestamps: Tensor) -> Tensor:
        mdd = self.mdd(rgb)
        pose = self.pose(coordinates, valid)
        tokens = pose[:, :, None]
        if self.config.readout == "query":
            tokens = torch.stack((pose, self.ball_query.expand_as(pose)), dim=2)
        patches = self.encoder(mdd)
        times = timestamps - timestamps[:, :1]
        for block in self.blocks:
            tokens, patches = block(tokens, patches, times)
        output: Tensor = self.head(tokens[:, :, -1])
        return output.float().sigmoid()

class MDDQueryDetector(nn.Module):
    """RGB clips become fixed FP32 MDD before learned, pose-free processing."""

    def __init__(self, config: MDDPoseConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        super().__init__()
        if config.readout != "query_only" or config.pose_pooling is not None:
            raise ValueError("MDDQueryDetector requires query_only with no pose pooling")
        self.config = config
        self.mdd = RGBToMDD(mdd_a, mdd_b)
        self.encoder = MDDTokenEncoder(config.compression, config.stem_channels, config.mixed_channels, config.dim)
        self.ball_query = nn.Parameter(torch.randn(config.dim) * .02)
        self.blocks = nn.ModuleList(QueryFusionBlock(config) for _ in range(config.layers))
        self.head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 2))

    def forward(self, rgb: Tensor, timestamps: Tensor) -> Tensor:
        mdd = self.mdd(rgb)
        patches = self.encoder(mdd)
        queries = self.ball_query.expand(mdd.shape[0], mdd.shape[2], -1)
        times = timestamps - timestamps[:, :1]
        for block in self.blocks:
            queries = block(queries, patches, times)
        output: Tensor = self.head(queries)
        return output.float().sigmoid()

class MDDDPTDetector(nn.Module):
    """RGB -> fixed FP32 MDD -> deep CNN -> frame-local DPT -> /4 heatmap logits."""
    def __init__(self, config: MDDPretrainConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        super().__init__()
        self.config = config
        self.mdd = build_input_transform(config.input_channels, a=mdd_a, b=mdd_b, frames=config.frames)
        self.encoder = DeepMDDEncoder(config)
        self.decoder = DPTDecoder(encoder_channels=self.encoder.feature_channels,
                                  decoder_channels=config.decoder_channels, reassemble_factors=(1., 1., 1., 1.),
                                  block_style="residual")
        self.head = nn.Sequential(nn.Conv2d(config.decoder_channels, config.decoder_channels // 2, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(config.decoder_channels // 2, 1, 1))

    def forward(self, rgb: Tensor) -> Tensor:
        features = self.encoder(self.mdd(rgb))
        # Produce logits on the declared /4 target grid; do not upscale the target to RGB size.
        return decode_frame_heatmaps(self.decoder, self.head, features,
            output_size=((rgb.shape[-2] + 3) // 4, (rgb.shape[-1] + 3) // 4),
            frame_chunk_size=self.config.frame_chunk_size, activation_checkpointing=self.config.activation_checkpointing)

class DeepMDDQueryTrunk(nn.Module):
    """Shared query path; registration names and operation order match stage two."""
    def __init__(self, config: MDDPretrainConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        super().__init__()
        self.config = config
        self.mdd = build_input_transform(config.input_channels, a=mdd_a, b=mdd_b, frames=config.frames)
        self.encoder = DeepMDDEncoder(config)
        self.projection = nn.Linear(config.mixed_channels[-1], config.dim)
        self.spatial_position = nn.Linear(2, config.dim)
        self.ball_query = nn.Parameter(torch.randn(config.dim) * .02)
        query_config = MDDPoseConfig("conv2d", None, "query_only", config.frames, config.stem_channels,
                                    config.mixed_channels, config.dim, config.heads, config.layers,
                                    config.dropout, config.rope_base)
        self.blocks = nn.ModuleList(QueryFusionBlock(query_config, ffn=nn.Sequential(
            SwiGLU(config.dim, config.ffn_dim), nn.Dropout(config.dropout))) for _ in range(config.layers))
        self.encoder_frozen = False

    def freeze_encoder(self, frozen: bool) -> None:
        self.encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def encode_queries(self, rgb: Tensor, timestamps: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        features = self.encoder(self.mdd(rgb))
        feature = features[-1]
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
            # Retain block inputs and shared patches; recompute attention/FFN per block.
            query = checkpoint_call(block, query, patches, timestamps - timestamps[:, :1],
                                    enabled=self.config.activation_checkpointing and self.training)
        return features, query


class DeepMDDQueryDetector(DeepMDDQueryTrunk):
    """Stage-two coordinate regression with the exact pretrained CNN topology."""

    def __init__(self, config: MDDPretrainConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> None:
        super().__init__(config, mdd_a=mdd_a, mdd_b=mdd_b)
        self.head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 2))

    def forward(self, rgb: Tensor, timestamps: Tensor) -> Tensor:
        _, query = self.encode_queries(rgb, timestamps)
        return cast(Tensor, self.head(query).float().sigmoid())

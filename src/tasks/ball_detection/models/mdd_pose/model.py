"""Temporal RoPE plus local-frame cross-attention; direct normalized UV output."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MDDPoseConfig
from .encoder import MDDTokenEncoder
from .pooling import PoseTokenizer


class TimeAttention(nn.Module):
    def __init__(self, dim: int, heads: int, base: float, dropout: float) -> None:
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)
        self.register_buffer("frequencies", base ** (-torch.arange(0, dim // heads, 2).float() / (dim // heads)))

    def forward(self, x: Tensor, times: Tensor) -> Tensor:
        b, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(b, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        phase = (times[:, None, :, None] * self.frequencies).to(q.dtype)
        def rotate(z: Tensor) -> Tensor:
            even, odd = z[..., ::2], z[..., 1::2]
            return torch.stack((even * phase.cos() - odd * phase.sin(),
                                even * phase.sin() + odd * phase.cos()), -1).flatten(-2)
        value = F.scaled_dot_product_attention(rotate(q), rotate(k), v,
                                               dropout_p=self.dropout if self.training else 0.)
        return self.output(value.transpose(1, 2).reshape(b, length, dim))


class FusionBlock(nn.Module):
    def __init__(self, config: MDDPoseConfig) -> None:
        super().__init__()
        d = config.dim
        self.temporal = TimeAttention(d, config.heads, config.rope_base, config.dropout)
        self.to_image = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.to_pose = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(4))
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(4 * d, d))

    def forward(self, tokens: Tensor, patches: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
        b, t, count, dim = tokens.shape
        q = self.norms[1](tokens).reshape(b * t, count, dim)
        image = self.norms[2](patches).reshape(b * t, -1, dim)
        image_delta, _ = self.to_image(image, q, q, need_weights=False)
        patches = patches + image_delta.reshape_as(patches)
        delta, _ = self.to_pose(q, patches.reshape(b * t, -1, dim), patches.reshape(b * t, -1, dim), need_weights=False)
        tokens = tokens + delta.reshape_as(tokens)
        tokens = tokens + self.temporal(self.norms[0](tokens).reshape(b, t * count, dim),
                                        times.repeat_interleave(count, dim=1)).reshape(b, t, count, dim)
        return tokens + self.ffn(self.norms[3](tokens)), patches


class MDDPoseDetector(nn.Module):
    """Input MDD B,2,32,H,W; pose B,32,N,17,2; output B,32,2.

    Mask/PTS are metadata, not additional visual features. No pretrained modules,
    RGB branch, heatmap fabrication or automatic refiner conversion is provided.
    """

    def __init__(self, config: MDDPoseConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MDDTokenEncoder(config.compression, config.stem_channels, config.mixed_channels, config.dim)
        self.pose = PoseTokenizer(config.pose_pooling, config.dim, config.heads)
        if config.readout == "query":
            self.ball_query = nn.Parameter(torch.randn(config.dim) * .02)
        self.blocks = nn.ModuleList(FusionBlock(config) for _ in range(config.layers))
        self.head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 2))

    def forward(self, mdd: Tensor, coordinates: Tensor, valid: Tensor, timestamps: Tensor) -> Tensor:
        pose = self.pose(coordinates, valid)
        tokens = pose[:, :, None]
        if self.config.readout == "query":
            tokens = torch.stack((pose, self.ball_query.expand_as(pose)), dim=2)
        patches = self.encoder(mdd)
        times = timestamps - timestamps[:, :1]
        for block in self.blocks:
            tokens, patches = block(tokens, patches, times)
        return self.head(tokens[:, :, -1]).float().sigmoid()

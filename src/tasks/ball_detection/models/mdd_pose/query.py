"""MDD-only coordinate regression with read-only, same-frame cross-attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.preprocessing import RGBToMDD

from .config import MDDPoseConfig
from .encoder import MDDTokenEncoder
from .model import TimeAttention


class QueryFusionBlock(nn.Module):
    """Read image evidence, mix queries over real time, then apply a token FFN."""

    def __init__(self, config: MDDPoseConfig, *, ffn: nn.Module | None = None) -> None:
        super().__init__()
        dim = config.dim
        self.cross = nn.MultiheadAttention(dim, config.heads, dropout=config.dropout, batch_first=True)
        self.temporal = TimeAttention(dim, config.heads, config.rope_base, config.dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(4))
        self.ffn = ffn if ffn is not None else nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(4 * dim, dim))

    def forward(self, queries: Tensor, patches: Tensor, times: Tensor) -> Tensor:
        b, t, dim = queries.shape
        q = self.norms[0](queries).reshape(b * t, 1, dim)
        image = self.norms[1](patches).reshape(b * t, -1, dim)
        delta, _ = self.cross(q, image, image, need_weights=False)
        queries = queries + delta.reshape(b, t, dim)
        queries = queries + self.temporal(self.norms[2](queries), times)
        output: Tensor = queries + self.ffn(self.norms[3](queries))
        return output


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

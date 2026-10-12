from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.ball_detection.models.mdd.config import MDDPoseConfig


class TimeAttention(nn.Module):
    frequencies: Tensor

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
        output: Tensor = self.output(value.transpose(1, 2).reshape(b, length, dim))
        return output

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

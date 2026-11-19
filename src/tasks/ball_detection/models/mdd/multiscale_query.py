"""Few-query spatial MSDA with frame-local image memory and unchanged mHC time path."""
from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd.config import MDDPretrainConfig
from src.tasks.ball_detection.models.mdd.mhc_query import (
    DeepMDDMHCQueryDetector,
    MHCQueryBlock,
)
from src.tasks.ball_detection.models.mdd.multiscale_config import (
    MDDMultiScaleQueryConfig,
)
from src.utils.models.components.ops.msda.api import OfficialMSDeformAttn


class PyramidMemory(NamedTuple):
    values: Tensor
    shapes: Tensor
    starts: Tensor
    mask: Tensor
    valid_ratios: Tensor


def spatial_grid(height: int, width: int, *, stride: int, image_size: tuple[int, int], like: Tensor) -> Tensor:
    """Same source-image endpoint-normalized cell centers as the legacy /64 CA."""
    yy = torch.arange(height, device=like.device, dtype=like.dtype) * stride
    xx = torch.arange(width, device=like.device, dtype=like.dtype) * stride
    yy = (yy + (yy + stride - 1).clamp_max(image_size[0] - 1)) / (2 * (image_size[0] - 1))
    xx = (xx + (xx + stride - 1).clamp_max(image_size[1] - 1)) / (2 * (image_size[1] - 1))
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((x, y), -1).reshape(height * width, 2)


class DeformableMHCQueryBlock(nn.Module):
    def __init__(self, config: MDDMultiScaleQueryConfig) -> None:
        super().__init__()
        self.path = MHCQueryBlock(config)
        del self.path.cross
        self.cross = OfficialMSDeformAttn(config.dim, len(config.msda_levels), config.heads, config.msda_points,
                                        im2col_step=config.msda_im2col_step)
        self.reference_points = nn.Linear(config.dim, 2)
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.zeros_(self.reference_points.bias)
        self.chunk_size = config.msda_chunk_size

    def forward(self, streams: Tensor, memory: PyramidMemory, times: Tensor) -> Tensor:
        b, t, count, dim = streams.shape
        queries = self.path.norms[0](streams).reshape(b*t, count, dim)
        with torch.autocast(queries.device.type, enabled=False):
            references = self.reference_points(queries.float()).sigmoid()
        # Official MSDA references are edge-normalized on each feature grid.
        # A complete fixed image has no invalid tokens: valid ratios are one,
        # including partial CNN border cells. No GT or output-head UV is used.
        references = references[:, :, None] * memory.valid_ratios[:, None]
        values = memory.values.reshape(b*t, -1, dim)
        chunks = []
        for first in range(0, b*t, self.chunk_size):
            last = min(first+self.chunk_size, b*t)
            image = self.path.norms[1](values[first:last])
            chunks.append(self.cross(queries[first:last], references[first:last], image,
                                     memory.shapes, memory.starts, memory.mask[first:last]))
        residual = streams + torch.cat(chunks, dim=0).reshape_as(streams)
        return self.path.temporal_span(residual, times)


class DeepMDDMultiScaleQueryDetector(DeepMDDMHCQueryDetector):
    config: MDDMultiScaleQueryConfig

    def _build_query_modules(self, config: MDDPretrainConfig) -> None:
        if not isinstance(config, MDDMultiScaleQueryConfig):
            raise ValueError("Multi-scale query requires its explicit configuration")
        self.ball_query = nn.Parameter(torch.randn(config.num_streams, config.dim) * .02)
        # Per-kind ordinal seeds pair the initial M/C/temporal/mHC weights across
        # layouts: changing order does not consume a different RNG stream.
        counts: Counter[str] = Counter()
        layers = []
        seed = torch.initial_seed()
        for kind in config.spatial_layers:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + (10000 if kind == "M" else 20000) + counts[kind])
                layers.append(DeformableMHCQueryBlock(config) if kind == "M" else MHCQueryBlock(config))
            counts[kind] += 1
        self.blocks = nn.ModuleList(layers)
        widths = dict(zip((8, 16, 32, 64), (*config.stem_channels[2:], *config.mixed_channels), strict=True))
        self.level_projections = nn.ModuleDict({str(scale): nn.Linear(widths[scale], config.dim)
                                                for scale in config.msda_levels if scale != 64})
        self.level_embedding = nn.Parameter(torch.randn(len(config.msda_levels), config.dim)*.02)

    def prepare_memory(self, features: tuple[Tensor, ...], *, image_size: tuple[int, int]) -> tuple[Tensor, PyramidMemory]:
        b, t = features[-1].shape[:2]
        levels = dict(zip((8, 16, 32, 64), features, strict=True))
        patches = self.projection(features[-1].flatten(-2).transpose(-1, -2))
        h, w = features[-1].shape[-2:]
        patches = patches + self.spatial_position(spatial_grid(h, w, stride=64, image_size=image_size, like=features[-1]))[None, None]
        values, shapes = [], []
        for index, scale in enumerate(self.config.msda_levels):
            feature = levels[scale]
            h, w = feature.shape[-2:]
            if scale == 64:
                projected = patches
            else:
                projected = self.level_projections[str(scale)](feature.flatten(-2).transpose(-1, -2))
                projected = projected + self.spatial_position(spatial_grid(h, w, stride=scale, image_size=image_size, like=feature))[None, None]
            values.append(projected + self.level_embedding[index])
            shapes.append((h, w))
        flattened = torch.cat(values, dim=2)
        shape_tensor = torch.tensor(shapes, dtype=torch.long, device=flattened.device)
        starts = torch.cat((shape_tensor.new_zeros(1), shape_tensor.prod(1).cumsum(0)[:-1]))
        mask = torch.zeros((b*t, flattened.shape[2]), dtype=torch.bool, device=flattened.device)
        ratios = torch.ones((b*t, len(shapes), 2), dtype=torch.float32, device=flattened.device)
        return patches, PyramidMemory(flattened, shape_tensor, starts, mask, ratios)

    def encode_queries(self, rgb: Tensor, timestamps: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        features = self.encoder(self.mdd(rgb))
        patches, memory = self.prepare_memory(features, image_size=(rgb.shape[-2], rgb.shape[-1]))
        b, t = features[-1].shape[:2]
        streams = self.ball_query.expand(b, t, *self.ball_query.shape)
        times = timestamps - timestamps[:, :1]
        for block in self.blocks:
            streams = block(streams, memory if isinstance(block, DeformableMHCQueryBlock) else patches, times)
        return features, streams.mean(dim=-2)

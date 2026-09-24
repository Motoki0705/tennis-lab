"""PLCS track-wise temporal attention followed by clip-level query attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.utils.models import TransformerBlock, TransformerBlockConfig


def keep_mask(valid: Tensor) -> Tensor:
    return valid[:, :, None] & valid[:, None, :]


class TrackQueryStage(nn.Module):
    """Each query reads its own track, exchanges context, then returns next stage."""

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.temporal = TransformerBlock(config)
        self.queries = TransformerBlock(config)

    def forward(
        self, tracks: Tensor, queries: Tensor, observed: Tensor,
        temporal_freqs: Tensor, query_freqs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b, v, p, t, d = tracks.shape
        query_valid = observed.any(-1)
        valid = torch.cat((query_valid[..., None], observed), dim=-1).reshape(b * v * p, t + 1)
        values = torch.cat((queries[..., None, :], tracks), dim=-2).reshape(b * v * p, t + 1, d)
        values = self.temporal(values, freqs_cis=temporal_freqs, attn_mask=keep_mask(valid))
        values = values.masked_fill(~valid[..., None], 0).reshape(b, v, p, t + 1, d)
        tracks = values[..., 1:, :]
        queries = values[..., 0, :].reshape(b, v * p, d)
        flat_valid = query_valid.reshape(b, v * p)
        queries = self.queries(queries, freqs_cis=query_freqs, attn_mask=keep_mask(flat_valid))
        queries = queries.masked_fill(~flat_valid[..., None], 0).reshape(b, v, p, d)
        return tracks, queries

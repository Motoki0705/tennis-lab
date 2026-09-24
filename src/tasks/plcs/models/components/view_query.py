"""PLCS court-side temporal view queries and frame-wise view attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.plcs.models.components.track_query import keep_mask
from src.utils.models import RMSNorm, TransformerBlock, TransformerBlockConfig
from src.utils.models.components.ffn_layers import build_ffn
from src.utils.models.components.mhc import (
    ManifoldConstrainedHyperConnection,
    MHCConfig,
)


class TemporalViewQueryStage(nn.Module):
    """P -> mHC -> (T + one view Q) temporal -> V spatial -> mHC -> P."""

    def __init__(self, attention: TransformerBlockConfig, connection: MHCConfig) -> None:
        super().__init__()
        self.mhc = ManifoldConstrainedHyperConnection(connection)
        self.temporal = TransformerBlock(attention)
        self.spatial = TransformerBlock(attention)
        self.ffn_norm = RMSNorm(attention.dim)
        self.ffn = build_ffn(
            ffn_type=attention.ffn_type, dim=attention.dim, ffn_dim=attention.ffn_dim
        )

    def forward(
        self,
        objects: Tensor,
        queries: Tensor,
        valid: Tensor,
        time_freqs: Tensor,
        spatial_freqs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b, v, t, p, d = objects.shape
        object_valid = valid[..., None].expand(b, v, t, p)
        compressed, state = self.mhc.pre(objects, object_valid)
        time_values = torch.cat(
            (queries[:, :, None], compressed.squeeze(-2)), dim=2
        ).reshape(b * v, t + 1, d)
        time_valid = torch.cat((valid.any(-1, keepdim=True), valid), dim=-1).reshape(
            b * v, t + 1
        )
        time_values = self.temporal(
            time_values, freqs_cis=time_freqs, attn_mask=keep_mask(time_valid)
        )
        time_values = time_values * time_valid[..., None]
        time_values = time_values.reshape(b, v, t + 1, d)
        queries = time_values[:, :, 0]
        spatial_values = time_values[:, :, 1:].permute(0, 2, 1, 3).reshape(b * t, v, d)
        spatial_valid = valid.permute(0, 2, 1).reshape(b * t, v)
        spatial_values = self.spatial(
            spatial_values, freqs_cis=spatial_freqs, attn_mask=keep_mask(spatial_valid)
        )
        spatial_values = spatial_values * spatial_valid[..., None]
        update = spatial_values.reshape(b, t, v, d).permute(0, 2, 1, 3)
        update = update + self.ffn(self.ffn_norm(update))
        queries = queries + self.ffn(self.ffn_norm(queries))
        objects = self.mhc.post(
            update.unsqueeze(-2).to(objects.dtype), residual=objects, state=state
        )
        return objects * object_valid[..., None], queries * valid.any(-1)[..., None]

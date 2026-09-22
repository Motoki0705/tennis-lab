"""Track-query reference trunk with temporal view queries and object ID heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from src.tasks.base.models import validate_reference_context_mask
from src.utils.models import (
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.ffn_layers import build_ffn
from src.utils.models.components.mhc import (
    ManifoldConstrainedHyperConnection,
    MHCConfig,
)


@dataclass(frozen=True)
class AssociationModelConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    ffn_dim: int = 768
    num_stages: int = 4
    rope_dim: int = 32
    num_slots: int = 4
    max_identities: int = 10
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name != "dropout" and (type(value) is not int or value <= 0):
                raise ValueError(
                    f"association.model.{field.name} must be a positive int"
                )
        if (
            self.hidden_dim % self.num_heads
            or self.rope_dim % 2
            or not 6 <= self.rope_dim <= self.hidden_dim // self.num_heads
        ):
            raise ValueError(
                "Invalid attention dimensions for three-axis reference RoPE"
            )
        if not 0 <= self.dropout < 1 or self.max_identities < self.num_slots:
            raise ValueError("Invalid dropout or identity capacity")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> AssociationModelConfig:
        if set(value) != set(asdict(cls())):
            raise ValueError(
                "association.model requires the exact declared configuration fields"
            )
        return cls(**value)


def _keep(valid: Tensor) -> Tensor:
    # SDPA True=keep. Invalid query rows remain finite, then are zeroed explicitly.
    return valid[:, :, None] & valid[:, None, :]


class TemporalViewQueryStage(nn.Module):
    """P -> mHC -> (T + one view Q) temporal -> V spatial -> mHC -> P."""

    def __init__(self, cfg: AssociationModelConfig) -> None:
        super().__init__()
        self.mhc = ManifoldConstrainedHyperConnection(
            MHCConfig(
                dim=cfg.hidden_dim,
                num_streams=cfg.num_slots,
                coefficient_dim=64,
                sinkhorn_iters=20,
                eps=1e-6,
                residual_identity_bias=4.0,
                update_scale_init=0.1,
            )
        )
        block = TransformerBlockConfig(
            dim=cfg.hidden_dim,
            n_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            head_dim=cfg.hidden_dim // cfg.num_heads,
            rope_dim=cfg.rope_dim,
            attn_dropout=cfg.dropout,
            attention_type="mha",
            n_kv_heads=None,
            rope_base=10000.0,
            ffn_type="swiglu",
            ffn_enabled=False,
        )
        self.temporal = TransformerBlock(block)
        self.spatial = TransformerBlock(block)
        self.ffn_norm = RMSNorm(cfg.hidden_dim)
        self.ffn = build_ffn(ffn_type="swiglu", dim=cfg.hidden_dim, ffn_dim=cfg.ffn_dim)

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
            time_values, freqs_cis=time_freqs, attn_mask=_keep(time_valid)
        )
        time_values = time_values * time_valid[..., None]
        time_values = time_values.reshape(b, v, t + 1, d)
        queries = time_values[:, :, 0]
        spatial_values = time_values[:, :, 1:].permute(0, 2, 1, 3).reshape(b * t, v, d)
        spatial_valid = valid.permute(0, 2, 1).reshape(b * t, v)
        spatial_values = self.spatial(
            spatial_values, freqs_cis=spatial_freqs, attn_mask=_keep(spatial_valid)
        )
        spatial_values = spatial_values * spatial_valid[..., None]
        update = spatial_values.reshape(b, t, v, d).permute(0, 2, 1, 3)
        update = update + self.ffn(self.ffn_norm(update))
        queries = queries + self.ffn(self.ffn_norm(queries))
        objects = self.mhc.post(
            update.unsqueeze(-2).to(objects.dtype), residual=objects, state=state
        )
        return objects * object_valid[..., None], queries * valid.any(-1)[..., None]


class ViewAssociationModel(nn.Module):
    """Raw observations plus one reference index; no camera side/pose input.

    Identity classes are arbitrary clip-global labels, not reusable local slots.
    No spatial query tokens or 3D heads exist in this model.
    """

    def __init__(self, cfg: AssociationModelConfig, *, num_keypoints: int) -> None:
        super().__init__()
        if num_keypoints not in (1, 17):
            raise ValueError(
                "Association supports ball (1) or COCO player (17) observations"
            )
        self.config = cfg
        self.num_keypoints = num_keypoints
        self.object_encoder = nn.Sequential(
            nn.Linear(num_keypoints * 3, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.court_encoder = nn.Linear(14 * 3, cfg.hidden_dim)
        self.invisible_object = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.02)
        self.view_query = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.02)
        self.reference_embedding = nn.Embedding(2, cfg.hidden_dim)
        self.stages = nn.ModuleList(
            [TemporalViewQueryStage(cfg) for _ in range(cfg.num_stages)]
        )
        self.time_frequency = RotaryFrequencyComputer(
            dim=cfg.rope_dim, base=10000.0, n_axes=1
        )
        self.spatial_frequency = RotaryFrequencyComputer(
            dim=cfg.rope_dim, base=10000.0, n_axes=3
        )
        self.norm = RMSNorm(cfg.hidden_dim)
        self.side_head = nn.Linear(cfg.hidden_dim, 1)
        self.object_id_head = nn.Linear(cfg.hidden_dim, cfg.max_identities + 1)

    def forward(
        self,
        object_uv: Tensor,
        object_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        reference_view_index: Tensor,
    ) -> dict[str, Tensor]:
        b, v, t, p, j, xy = object_uv.shape
        if (p, j, xy) != (self.config.num_slots, self.num_keypoints, 2):
            raise ValueError("object_uv must match (B,V,T,num_slots,num_keypoints,2)")
        if (
            object_vis.shape != object_uv.shape[:-1]
            or court_kp.shape != (b, v, t, 14, 2)
            or court_vis.shape != (b, v, t, 14)
            or padding_mask.shape != (b, v, t)
        ):
            raise ValueError("Association observation/mask shapes do not agree")
        if any(x.dtype != torch.bool for x in (object_vis, court_vis, padding_mask)):
            raise TypeError("Association visibility and padding must be boolean")
        validate_reference_context_mask(reference_view_index, ~padding_mask)
        visible = object_vis & ~padding_mask[..., None, None]
        court_visible = court_vis & ~padding_mask[..., None]
        features = torch.cat(
            (
                object_uv.masked_fill(~visible[..., None], 0).flatten(-2),
                visible.to(object_uv.dtype),
            ),
            dim=-1,
        )
        objects = self.object_encoder(features)
        objects = torch.where(
            visible.any(-1)[..., None], objects, self.invisible_object
        )
        court_features = torch.cat(
            (
                court_kp.masked_fill(~court_visible[..., None], 0).flatten(-2),
                court_visible.to(court_kp.dtype),
            ),
            dim=-1,
        )
        selector = (
            torch.arange(v, device=object_uv.device)[None, :]
            .ne(reference_view_index[:, None])
            .long()
        )
        objects = (
            objects
            + self.court_encoder(court_features)[..., None, :]
            + self.reference_embedding(selector)[:, :, None, None]
        )
        objects = objects * (~padding_mask)[..., None, None]
        queries = self.view_query.expand(b, v, -1) + self.reference_embedding(selector)
        coordinates = torch.zeros(b, t, v, 3, device=object_uv.device)
        coordinates[..., 0] = torch.arange(t, device=object_uv.device)[None, :, None]
        coordinates[..., 1] = (
            torch.arange(v, device=object_uv.device)[None, None, :] + 1
        )
        coordinates[..., 2] = selector[:, None, :]
        spatial_freqs = self.spatial_frequency(coordinates.flatten(0, 1))
        # Q has a separate position -1 and attends to the whole clip; no CSWA-local interpretation.
        time_freqs = self.time_frequency(
            torch.arange(-1, t, device=object_uv.device)[:, None]
        )
        for stage in self.stages:
            objects, queries = stage(
                objects, queries, ~padding_mask, time_freqs, spatial_freqs
            )
        return {
            "side_logits": self.side_head(self.norm(queries)).squeeze(-1),
            "object_id_logits": self.object_id_head(self.norm(objects)),
        }

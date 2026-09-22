"""Shared court-observation encoder, association heads, and model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from src.utils.models import RMSNorm, RotaryFrequencyComputer, TransformerBlockConfig
from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES, FFNType
from src.utils.models.components.mhc import MHCConfig
from src.utils.models.components.view_query import TemporalViewQueryStage


@dataclass(frozen=True)
class ViewQueryModelConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    ffn_dim: int = 768
    num_stages: int = 4
    rope_dim: int = 32
    num_slots: int = 4
    max_identities: int = 10
    dropout: float = 0.1
    ffn_type: FFNType = "swiglu"

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {"dropout", "ffn_type"}:
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"model.{field.name} must be a positive int"
                )
        if self.ffn_type not in SUPPORTED_FFN_TYPES:
            raise ValueError(f"Unsupported model.ffn_type={self.ffn_type}")
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
    def from_mapping(cls, value: dict[str, Any]) -> ViewQueryModelConfig:
        if set(value) != set(asdict(cls())):
            raise ValueError(
                "model requires the exact declared configuration fields"
            )
        return cls(**value)


class ViewQueryAssociationModel(nn.Module):
    """Raw observations plus one reference index; no camera side/pose input.

    Identity classes are arbitrary clip-global labels, not reusable local slots.
    No spatial query tokens or 3D heads exist in this model.
    """

    def __init__(self, cfg: ViewQueryModelConfig, *, num_keypoints: int) -> None:
        super().__init__()
        if type(num_keypoints) is not int or num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
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
        attention = TransformerBlockConfig(
            dim=cfg.hidden_dim,
            n_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            head_dim=cfg.hidden_dim // cfg.num_heads,
            rope_dim=cfg.rope_dim,
            attn_dropout=cfg.dropout,
            attention_type="mha",
            n_kv_heads=None,
            rope_base=10000.0,
            ffn_type=cfg.ffn_type,
            ffn_enabled=False,
        )
        connection = MHCConfig(
            dim=cfg.hidden_dim,
            num_streams=cfg.num_slots,
            coefficient_dim=64,
            sinkhorn_iters=20,
            eps=1e-6,
            residual_identity_bias=4.0,
            update_scale_init=0.1,
        )
        self.stages = nn.ModuleList(
            [TemporalViewQueryStage(attention, connection) for _ in range(cfg.num_stages)]
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
        b, v, t, _p, _j, _xy = object_uv.shape
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

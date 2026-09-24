"""Independent PLCS camera half-turn estimator, without identity prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.plcs.models.components.view_query import TemporalViewQueryStage
from src.tasks.plcs.models.person_tokens import (
    PersonModelConfig,
    PersonObservationEncoder,
)
from src.utils.models import RMSNorm, RotaryFrequencyComputer
from src.utils.models.components.mhc import MHCConfig


class CourtSideModel(nn.Module):
    def __init__(self, config: PersonModelConfig) -> None:
        super().__init__()
        if config.rope_dim < 6:
            raise ValueError("Court side requires at least six rotary dimensions")
        self.config = config
        self.encoder = PersonObservationEncoder(config.hidden_dim)
        self.view_query = nn.Parameter(torch.randn(config.hidden_dim) * .02)
        self.reference_embedding = nn.Embedding(2, config.hidden_dim)
        attention = config.attention()
        attention.ffn_enabled = False
        connection = MHCConfig(dim=config.hidden_dim, num_streams=config.num_slots,
            coefficient_dim=64, sinkhorn_iters=20, eps=1e-6,
            residual_identity_bias=4., update_scale_init=.1)
        self.stages = nn.ModuleList([TemporalViewQueryStage(attention, connection) for _ in range(config.num_stages)])
        self.time_frequency = RotaryFrequencyComputer(dim=config.rope_dim, base=10000., n_axes=1)
        self.view_frequency = RotaryFrequencyComputer(dim=config.rope_dim, base=10000., n_axes=3)
        self.norm = RMSNorm(config.hidden_dim)
        self.side_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, human_kp: Tensor, human_vis: Tensor, court_kp: Tensor, court_vis: Tensor, padding_mask: Tensor, reference_view_index: Tensor) -> dict[str, Tensor]:
        objects, _, _ = self.encoder(human_kp, human_vis, court_kp, court_vis, padding_mask)
        b, v, t, _, _ = objects.shape
        selector = torch.arange(v, device=objects.device)[None].ne(reference_view_index[:, None]).long()
        reference = self.reference_embedding(selector)
        objects = (objects + reference[:, :, None, None]).masked_fill(padding_mask[..., None, None], 0)
        queries = self.view_query.expand(b, v, -1) + reference
        coordinates = torch.zeros(b, t, v, 3, device=objects.device)
        coordinates[..., 0] = torch.arange(t, device=objects.device)[None, :, None]
        coordinates[..., 2] = selector[:, None]
        # Arbitrary view ordering is not a geometric position.
        spatial = self.view_frequency(coordinates.flatten(0, 1))
        temporal = self.time_frequency(torch.arange(-1, t, device=objects.device)[:, None])
        for stage in self.stages:
            objects, queries = stage(objects, queries, ~padding_mask, temporal, spatial)
        return {"side_logits": self.side_head(self.norm(queries)).squeeze(-1)}

"""Fixed-track PLCS embeddings; identity is decoded using cosine similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.tasks.plcs.models.components.track_query import TrackQueryStage
from src.tasks.plcs.models.person_tokens import (
    PersonModelConfig,
    PersonObservationEncoder,
)
from src.utils.models import RMSNorm, RotaryFrequencyComputer


class PlayerReIDModel(nn.Module):
    def __init__(self, config: PersonModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = PersonObservationEncoder(config.hidden_dim)
        self.track_query = nn.Parameter(torch.randn(config.hidden_dim) * .02)
        self.stages = nn.ModuleList([TrackQueryStage(config.attention()) for _ in range(config.num_stages)])
        self.frequency = RotaryFrequencyComputer(dim=config.rope_dim, base=10000., n_axes=1)
        self.norm = RMSNorm(config.hidden_dim)
        self.projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, human_kp: Tensor, human_vis: Tensor, court_kp: Tensor, court_vis: Tensor, padding_mask: Tensor) -> dict[str, Tensor]:
        tokens, observed, _ = self.encoder(human_kp, human_vis, court_kp, court_vis, padding_mask)
        b, v, t, p, _ = tokens.shape
        tracks = tokens.permute(0, 1, 3, 2, 4)
        observed = observed.permute(0, 1, 3, 2)
        tracks = tracks.masked_fill(~observed[..., None], 0)
        queries = self.track_query.expand(b, v, p, -1)
        temporal_freqs = self.frequency(torch.arange(-1, t, device=tokens.device)[:, None])
        # Queries form an unordered set: camera/slot indices are not positions.
        query_freqs = self.frequency(torch.zeros(v * p, 1, device=tokens.device))
        for stage in self.stages:
            tracks, queries = stage(tracks, queries, observed, temporal_freqs, query_freqs)
        valid = observed.any(-1)
        features = self.norm(queries)
        embedding = F.normalize(self.projection(features).float(), dim=-1).masked_fill(~valid[..., None], 0)
        return {"track_embedding": embedding, "track_valid": valid}

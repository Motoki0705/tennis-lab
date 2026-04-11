"""Token embedding helpers shared across tasks.

These embeddings follow the project's convention:
- Court tokens: (u, v, visibility)
- Ball/trajectory tokens: (u, v, observed_mask)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.schema.court import NUM_COURT_KP


class CourtKPUVTokenEmbedding(nn.Module):
    """Embed court keypoints (uv + visibility) into tokens."""

    def __init__(self, *, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, int(dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, court_kp: Tensor, court_vis: Tensor | None) -> Tensor:
        B = int(court_kp.shape[0])
        if court_kp.dim() == 2:
            court_kp = court_kp.view(B, -1, 2)
        n_kp = court_kp.shape[1]
        vis = (
            torch.ones(B, n_kp, device=court_kp.device, dtype=court_kp.dtype)
            if court_vis is None
            else court_vis.to(court_kp.dtype)
        )
        x = torch.cat([court_kp, vis.unsqueeze(-1)], dim=-1)
        return self.proj(x)


class UVObsTokenEmbedding(nn.Module):
    """Embed ball UV observations (uv + observed_mask) into tokens."""

    def __init__(self, *, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, int(dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, ball_uv: Tensor, ball_obs_mask: Tensor | None) -> Tensor:
        B, T, _ = ball_uv.shape
        obs = (
            torch.ones(B, T, device=ball_uv.device, dtype=ball_uv.dtype)
            if ball_obs_mask is None
            else ball_obs_mask.to(ball_uv.dtype)
        )
        x = torch.cat([ball_uv, obs.unsqueeze(-1)], dim=-1)
        return self.proj(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    emb_court = CourtKPUVTokenEmbedding(dim=32, dropout=0.0)
    emb_ball = UVObsTokenEmbedding(dim=32, dropout=0.0)
    court_kp = torch.rand(2, NUM_COURT_KP, 2)
    court_vis = torch.ones(2, NUM_COURT_KP)
    ball_uv = torch.rand(2, 16, 2)
    ball_obs = torch.randint(0, 2, (2, 16)).float()
    out_c = emb_court(court_kp, court_vis)
    out_b = emb_ball(ball_uv, ball_obs)
    assert out_c.shape == (2, NUM_COURT_KP, 32)
    assert out_b.shape == (2, 16, 32)
    print("token_embeddings smoke ok")


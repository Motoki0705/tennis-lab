"""Linear observation fusion for BLCS track queries."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from src.utils.models.embeddings import CourtBallGroupEmbedding, InvisibleTokenEmbedding


class LinearTrackObservationFusion(nn.Module):
    """Fuse each court/candidate group through the shared linear embedding."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_court_tokens: int,
        invisible_init_std: float,
    ) -> None:
        super().__init__()
        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim,
            init_std=invisible_init_std,
        )
        self.group_embedding = CourtBallGroupEmbedding(
            dim=hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=num_court_tokens,
        )

    def forward(
        self,
        court_kp: Tensor,
        court_vis: Tensor,
        ball_uv: Tensor,
        ball_vis: Tensor,
    ) -> Tensor:
        """Return time-major candidate tokens for every fixed query slot."""
        num_detections = ball_uv.shape[3]
        masked_court = court_kp.masked_fill(~court_vis.unsqueeze(-1), 0.0)
        court_for_candidates = masked_court.unsqueeze(3).expand(
            -1,
            -1,
            -1,
            num_detections,
            -1,
            -1,
        )
        ball_for_candidates = ball_uv.masked_fill(~ball_vis.unsqueeze(-1), 0.0)
        return cast(
            Tensor,
            self.group_embedding(
                court_for_candidates,
                ball_for_candidates,
                ball_vis,
            ),
        ).permute(0, 2, 1, 3, 4)


__all__ = [
    "LinearTrackObservationFusion",
]

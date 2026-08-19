"""Preselected observation-fusion implementations for BLCS track queries."""

from __future__ import annotations

from torch import Tensor, nn

from src.tasks.blcs.configuration import PointFusionConfig
from src.tasks.blcs.models.components.court_ball_point_fusion import (
    CourtBallPointFusion,
)
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
        court_visible: Tensor,
        ball_uv: Tensor,
        ball_visible: Tensor,
        state_valid: Tensor,
        point_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return time-major candidate tokens and their state-valid mask."""
        num_detections = ball_uv.shape[3]
        masked_court = court_kp.masked_fill(~court_visible.unsqueeze(-1), 0.0)
        court_for_candidates = masked_court.unsqueeze(3).expand(
            -1,
            -1,
            -1,
            num_detections,
            -1,
            -1,
        )
        ball_for_candidates = ball_uv.masked_fill(~ball_visible.unsqueeze(-1), 0.0)
        tokens = self.group_embedding(
            court_for_candidates,
            ball_for_candidates,
            ball_visible,
        ).permute(0, 2, 1, 3, 4)
        del point_attention_mask
        return tokens, state_valid.permute(0, 2, 1, 3)


class PointAttentionTrackObservationFusion(nn.Module):
    """Fuse court and candidates with the preconfigured point-attention path."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_court_tokens: int,
        config: PointFusionConfig,
        invisible_init_std: float,
    ) -> None:
        super().__init__()
        self.point_fusion = CourtBallPointFusion(
            output_dim=hidden_dim,
            num_court_points=num_court_tokens,
            config=config,
            invisible_init_std=invisible_init_std,
        )

    def forward(
        self,
        court_kp: Tensor,
        court_visible: Tensor,
        ball_uv: Tensor,
        ball_visible: Tensor,
        state_valid: Tensor,
        point_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return time-major candidate tokens and their state-valid mask."""
        tokens = self.point_fusion(
            court_kp=court_kp,
            court_visible=court_visible,
            ball_uv=ball_uv,
            ball_visible=ball_visible,
            ball_state_valid=state_valid,
            attention_mask=point_attention_mask,
        ).permute(0, 2, 1, 3, 4)
        return tokens, state_valid.permute(0, 2, 1, 3)


__all__ = [
    "LinearTrackObservationFusion",
    "PointAttentionTrackObservationFusion",
]

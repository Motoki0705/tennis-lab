"""Preselected observation-fusion implementations for BLCS track queries."""

from __future__ import annotations

from typing import Literal, Protocol, cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.configuration import PointFusionConfig
from src.tasks.blcs.models.components.court_ball_point_fusion import (
    CourtBallPointFusion,
)
from src.utils.models.embeddings import (
    CourtBallGroupEmbedding,
    CourtObjectSetFusion,
    InvisibleTokenEmbedding,
    SymmetricCourtPeakEncoder,
)


class TrackObservationFusion(Protocol):
    """Uniform already-selected BLCS observation-fusion call contract."""

    def __call__(
        self,
        *,
        ball_uv: Tensor,
        ball_visible: Tensor,
        state_valid: Tensor,
        ball_score: Tensor | None,
        court_kp: Tensor | None,
        court_visible: Tensor | None,
        point_attention_mask: Tensor | None,
        court_peak_uv: Tensor | None,
        court_peak_score: Tensor | None,
        court_peak_covariance: Tensor | None,
        court_peak_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]: ...


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
        *,
        ball_uv: Tensor,
        ball_visible: Tensor,
        state_valid: Tensor,
        ball_score: Tensor | None,
        court_kp: Tensor | None,
        court_visible: Tensor | None,
        point_attention_mask: Tensor | None,
        court_peak_uv: Tensor | None,
        court_peak_score: Tensor | None,
        court_peak_covariance: Tensor | None,
        court_peak_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return time-major candidate tokens and their state-valid mask."""
        court_kp = cast(Tensor, court_kp)
        court_visible = cast(Tensor, court_visible)
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
        del (
            ball_score,
            point_attention_mask,
            court_peak_uv,
            court_peak_score,
            court_peak_covariance,
            court_peak_valid,
        )
        return tokens, state_valid


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
        *,
        ball_uv: Tensor,
        ball_visible: Tensor,
        state_valid: Tensor,
        ball_score: Tensor | None,
        court_kp: Tensor | None,
        court_visible: Tensor | None,
        point_attention_mask: Tensor | None,
        court_peak_uv: Tensor | None,
        court_peak_score: Tensor | None,
        court_peak_covariance: Tensor | None,
        court_peak_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return time-major candidate tokens and their state-valid mask."""
        court_kp = cast(Tensor, court_kp)
        court_visible = cast(Tensor, court_visible)
        point_attention_mask = cast(Tensor, point_attention_mask)
        tokens = self.point_fusion(
            court_kp=court_kp,
            court_visible=court_visible,
            ball_uv=ball_uv,
            ball_visible=ball_visible,
            ball_state_valid=state_valid.permute(0, 2, 1, 3),
            attention_mask=point_attention_mask,
        ).permute(0, 2, 1, 3, 4)
        del ball_score, court_peak_uv, court_peak_score, court_peak_covariance, court_peak_valid
        return tokens, state_valid


class KP7TrackObservationFusion(nn.Module):
    """Create symmetric object-conditioned tokens directly from CourtKP7 peaks."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        invisible_init_std: float,
    ) -> None:
        super().__init__()
        self.peak_encoder = SymmetricCourtPeakEncoder(hidden_dim)
        self.set_fusion = CourtObjectSetFusion(
            hidden_dim,
            object_feature_dim=4,
        )
        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim,
            init_std=invisible_init_std,
        )

    def forward(
        self,
        *,
        ball_uv: Tensor,
        ball_visible: Tensor,
        state_valid: Tensor,
        ball_score: Tensor | None,
        court_kp: Tensor | None,
        court_visible: Tensor | None,
        point_attention_mask: Tensor | None,
        court_peak_uv: Tensor | None,
        court_peak_score: Tensor | None,
        court_peak_covariance: Tensor | None,
        court_peak_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return time-major candidate tokens and their state-valid mask."""
        ball_score = cast(Tensor, ball_score)
        court_peak_uv = cast(Tensor, court_peak_uv)
        court_peak_score = cast(Tensor, court_peak_score)
        court_peak_covariance = cast(Tensor, court_peak_covariance)
        court_peak_valid = cast(Tensor, court_peak_valid)
        encoded, flat_valid = self.peak_encoder(
            court_peak_uv,
            court_peak_score,
            court_peak_covariance,
            court_peak_valid,
        )
        flat_uv = court_peak_uv.flatten(-3, -2)
        object_features = torch.cat(
            (
                ball_uv,
                ball_score.unsqueeze(-1),
                ball_visible.to(ball_uv.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        tokens = self.set_fusion(
            encoded,
            flat_uv,
            flat_valid,
            ball_uv,
            object_features,
        )
        invisible = self.invisible_token().view(1, 1, 1, 1, -1)
        tokens = tokens + (~ball_visible).unsqueeze(-1) * invisible
        del court_kp, court_visible, point_attention_mask
        return tokens.permute(0, 2, 1, 3, 4), state_valid


def build_track_observation_fusion(
    *,
    profile: str,
    observation_fusion: Literal["linear", "point_attention"],
    point_fusion: PointFusionConfig | None,
    hidden_dim: int,
    invisible_init_std: float,
) -> LinearTrackObservationFusion | PointAttentionTrackObservationFusion | KP7TrackObservationFusion:
    """Select one concrete observation path before ``forward`` executes."""
    if profile != "kp14_reference_baseline":
        return KP7TrackObservationFusion(
            hidden_dim=hidden_dim,
            invisible_init_std=invisible_init_std,
        )
    if observation_fusion == "linear":
        return LinearTrackObservationFusion(
            hidden_dim=hidden_dim,
            num_court_tokens=14,
            invisible_init_std=invisible_init_std,
        )
    if observation_fusion == "point_attention" and point_fusion is not None:
        return PointAttentionTrackObservationFusion(
            hidden_dim=hidden_dim,
            num_court_tokens=14,
            config=point_fusion,
            invisible_init_std=invisible_init_std,
        )
    raise ValueError(
        "point_fusion is required when observation_fusion='point_attention'."
    )


__all__ = [
    "LinearTrackObservationFusion",
    "KP7TrackObservationFusion",
    "PointAttentionTrackObservationFusion",
    "TrackObservationFusion",
    "build_track_observation_fusion",
]

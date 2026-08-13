"""Task-specific PLCS object encoding over the shared CourtKP7 set fusion."""

from __future__ import annotations

from typing import Protocol, cast

from torch import Tensor, nn

from src.utils.models.embeddings import (
    CourtObjectSetFusion,
    CourtPlayerGroupEmbedding,
    InvisibleTokenEmbedding,
    SymmetricCourtPeakEncoder,
)


class PlayerObservationFusion(Protocol):
    """Uniform already-selected PLCS observation-fusion call contract."""

    def __call__(
        self,
        *,
        human_kp: Tensor,
        detection_mask: Tensor,
        camera_state_valid: Tensor,
        court_kp: Tensor | None,
        court_vis: Tensor | None,
        court_peak_uv: Tensor | None,
        court_peak_score: Tensor | None,
        court_peak_covariance: Tensor | None,
        court_peak_valid: Tensor | None,
        player_anchor: Tensor | None,
        player_features: Tensor | None,
    ) -> tuple[Tensor, Tensor]: ...


class KP14PlayerObservationFusion(nn.Module):
    """Fuse ordered CourtKP14 and player detections for the named baseline."""

    def __init__(self, *, hidden_dim: int, invisible_init_std: float) -> None:
        super().__init__()
        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim,
            init_std=invisible_init_std,
        )
        self.group_embed = CourtPlayerGroupEmbedding(
            dim=hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=14,
        )

    def forward(
        self,
        *,
        human_kp: Tensor,
        detection_mask: Tensor,
        camera_state_valid: Tensor,
        court_kp: Tensor | None,
        court_vis: Tensor | None,
        court_peak_uv: Tensor | None,
        court_peak_score: Tensor | None,
        court_peak_covariance: Tensor | None,
        court_peak_valid: Tensor | None,
        player_anchor: Tensor | None,
        player_features: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return time-major player tokens and their state-valid mask."""
        court_kp = cast(Tensor, court_kp)
        court_vis = cast(Tensor, court_vis)
        num_detections = human_kp.shape[3]
        masked_court = court_kp.masked_fill(~court_vis.unsqueeze(-1), 0.0)
        court_for_detections = masked_court.unsqueeze(3).expand(
            -1, -1, -1, num_detections, -1, -1
        )
        human_for_detections = human_kp.masked_fill(
            ~detection_mask[..., None, None], 0.0
        )
        tokens = self.group_embed(
            court_for_detections,
            human_for_detections,
            detection_mask,
        ).permute(0, 2, 1, 3, 4)
        del (
            court_peak_uv,
            court_peak_score,
            court_peak_covariance,
            court_peak_valid,
            player_anchor,
            player_features,
        )
        return tokens, camera_state_valid


class KP7PlayerObservationFusion(nn.Module):
    """Fuse adapter-prepared full-pose geometry with unordered CourtKP7 peaks."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        player_feature_dim: int,
        invisible_init_std: float,
    ) -> None:
        super().__init__()
        self.peak_encoder = SymmetricCourtPeakEncoder(hidden_dim)
        self.set_fusion = CourtObjectSetFusion(
            hidden_dim,
            object_feature_dim=player_feature_dim,
        )
        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim,
            init_std=invisible_init_std,
        )

    def forward(
        self,
        *,
        human_kp: Tensor,
        detection_mask: Tensor,
        camera_state_valid: Tensor,
        court_kp: Tensor | None,
        court_vis: Tensor | None,
        court_peak_uv: Tensor | None,
        court_peak_score: Tensor | None,
        court_peak_covariance: Tensor | None,
        court_peak_valid: Tensor | None,
        player_anchor: Tensor | None,
        player_features: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return time-major player tokens and their state-valid mask."""
        court_peak_uv = cast(Tensor, court_peak_uv)
        court_peak_score = cast(Tensor, court_peak_score)
        court_peak_covariance = cast(Tensor, court_peak_covariance)
        court_peak_valid = cast(Tensor, court_peak_valid)
        player_anchor = cast(Tensor, player_anchor)
        player_features = cast(Tensor, player_features)
        encoded, flat_valid = self.peak_encoder(
            court_peak_uv,
            court_peak_score,
            court_peak_covariance,
            court_peak_valid,
        )
        tokens = self.set_fusion(
            encoded,
            court_peak_uv.flatten(-3, -2),
            flat_valid,
            player_anchor,
            player_features,
        )
        invisible = self.invisible_token().view(1, 1, 1, 1, -1)
        tokens = tokens + (~detection_mask).unsqueeze(-1) * invisible
        del human_kp, court_kp, court_vis
        return tokens.permute(0, 2, 1, 3, 4), camera_state_valid


def build_player_observation_fusion(
    *,
    profile: str,
    hidden_dim: int,
    player_feature_dim: int,
    invisible_init_std: float,
) -> KP14PlayerObservationFusion | KP7PlayerObservationFusion:
    """Select one concrete PLCS observation path before ``forward`` executes."""
    if profile == "kp14_reference_baseline":
        return KP14PlayerObservationFusion(
            hidden_dim=hidden_dim,
            invisible_init_std=invisible_init_std,
        )
    return KP7PlayerObservationFusion(
        hidden_dim=hidden_dim,
        player_feature_dim=player_feature_dim,
        invisible_init_std=invisible_init_std,
    )


__all__ = [
    "KP14PlayerObservationFusion",
    "KP7PlayerObservationFusion",
    "PlayerObservationFusion",
    "build_player_observation_fusion",
]

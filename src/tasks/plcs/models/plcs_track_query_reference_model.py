"""Reference-camera-conditioned canonical PLCS track-query model."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.base.models import (
    REFERENCE_SELECTOR_ROPE_CONTRACT,
    ReferenceSelectorMode,
    build_compressed_track_query_spatial_coordinates,
    resolve_reference_selector_mode,
    resolve_track_query_rope_contract,
    validate_reference_context_mask,
    validate_track_query_rope_dimensions,
)
from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.data.tracking_types import PLCSTrackingPrediction
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel


def _architecture_config(config: PLCSModelConfig) -> PLCSModelConfig:
    """Project the reference contract onto the fixed base architecture."""
    values = dict(config.values)
    values["name"] = "plcs_track_query"
    for key in (
        "target_frame_contract",
        "track_query_rope_contract",
        "reference_selector_mode",
    ):
        del values[key]
    return PLCSModelConfig.from_mapping(values)


class PLCSTrackQueryReferenceModel(PLCSTrackQueryModel):
    """Predict PLCS tracks in the selected reference-camera frame."""

    def __init__(self, config: PLCSModelConfig) -> None:
        if config.name != "plcs_track_query_reference":
            raise ValueError(
                "PLCSTrackQueryReferenceModel requires "
                "plcs_track_query_reference config."
            )
        super().__init__(_architecture_config(config))
        self.target_frame_contract = config.string("target_frame_contract")
        self.track_query_rope_contract = resolve_track_query_rope_contract(
            config.string("track_query_rope_contract")
        )
        if self.track_query_rope_contract is not REFERENCE_SELECTOR_ROPE_CONTRACT:
            raise ValueError(
                "PLCSTrackQueryReferenceModel requires the "
                "reference-selector RoPE contract."
            )
        self.reference_selector_mode = resolve_reference_selector_mode(
            config.string("reference_selector_mode")
        )
        if self.reference_selector_mode is not ReferenceSelectorMode.REFERENCE:
            raise ValueError("PLCS reference track-query requires reference mode.")
        validate_track_query_rope_dimensions(
            contract=self.track_query_rope_contract,
            rope_dim=self.rope_dim,
            head_dim=self.hidden_dim // self.num_heads,
        )
        self.register_buffer(
            "_reference_track_query_contract_marker",
            torch.tensor(1, dtype=torch.uint8),
        )
        self.register_buffer(
            "_reference_selector_mode_marker", torch.tensor(1, dtype=torch.uint8)
        )

    @staticmethod
    def build_spatial_coordinates(  # type: ignore[override]
        reference_view_index: Tensor,
        *,
        num_frames: int,
        num_views: int,
        num_detections: int,
        num_queries: int,
    ) -> Tensor:
        """Return exact v2 compressed selector coordinates."""
        if num_detections != num_queries:
            raise ValueError("num_detections must equal num_queries.")
        return build_compressed_track_query_spatial_coordinates(
            reference_view_index,
            num_frames=num_frames,
            num_views=num_views,
            num_queries=num_queries,
        )

    def forward(  # type: ignore[override]
        self,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        reference_view_index: Tensor,
    ) -> PLCSTrackingPrediction:
        """Predict tracks using the required ``int64[B]`` reference selector."""
        _batch_size, num_views, num_frames = human_kp.shape[:3]
        validate_reference_context_mask(
            reference_view_index,
            ~padding_mask,
        )
        spatial_coordinates = self.build_spatial_coordinates(
            reference_view_index,
            num_frames=num_frames,
            num_views=num_views,
            num_detections=self.num_queries,
            num_queries=self.num_queries,
        )
        return self._forward_with_spatial_coordinates(
            human_kp,
            human_vis,
            court_kp,
            court_vis,
            padding_mask,
            spatial_coordinates=spatial_coordinates,
        )


__all__ = ["PLCSTrackQueryReferenceModel"]

"""Reference-conditioned v2 BLCS track-query model."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.base.data.track_query_reference import validate_reference_view_index
from src.tasks.base.models import (
    REFERENCE_SELECTOR_ROPE_CONTRACT,
    ReferenceSelectorMode,
    build_full_track_query_spatial_coordinates,
    resolve_reference_selector_mode,
    validate_reference_context_mask,
    validate_track_query_rope_dimensions,
)
from src.tasks.blcs.configuration import (
    TrackQueryModelConfig,
    TrackQueryReferenceModelConfig,
)
from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.tasks.blcs.models.blcs_track_query_model import BLCSTrackQueryModel


def _legacy_architecture_config(
    config: TrackQueryReferenceModelConfig,
) -> TrackQueryModelConfig:
    """Build only the architecture fields shared with the immutable v1 model."""
    return TrackQueryModelConfig(
        name="blcs_track_query",
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_stages=config.num_stages,
        ffn_dim=config.ffn_dim,
        num_queries=config.num_queries,
        rope_dim=config.rope_dim,
        dropout=config.dropout,
        role_rope_enabled=False,
        invisible_init_std=config.invisible_init_std,
        mhc=config.mhc,
        cswa=config.cswa,
    )


class BLCSTrackQueryReferenceModel(BLCSTrackQueryModel):
    """Predict ball tracks in the frame selected by one camera per sample."""

    def __init__(self, config: TrackQueryReferenceModelConfig) -> None:
        if config.name != "blcs_track_query_reference":
            raise ValueError(
                "BLCSTrackQueryReferenceModel requires "
                "blcs_track_query_reference config."
            )
        super().__init__(_legacy_architecture_config(config))
        self.target_frame_contract = config.target_frame_contract
        self.track_query_rope_contract = config.track_query_rope_contract
        self.reference_selector_mode = resolve_reference_selector_mode(
            config.reference_selector_mode
        )
        if self.reference_selector_mode is not ReferenceSelectorMode.REFERENCE:
            raise ValueError("The normal reference model requires reference mode.")
        validate_track_query_rope_dimensions(
            contract=REFERENCE_SELECTOR_ROPE_CONTRACT,
            rope_dim=self.rope_dim,
            head_dim=self.hidden_dim // self.num_heads,
        )
        self.register_buffer(
            "_track_query_reference_architecture_marker",
            torch.tensor(1, dtype=torch.uint8),
        )
        self.register_forward_pre_hook(
            self._validate_reference_forward_inputs,
            with_kwargs=True,
        )

    def _validate_reference_forward_inputs(
        self,
        _module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        if len(args) <= 5 and "reference_view_index" not in kwargs:
            raise TypeError("reference_view_index is the required sixth tensor.")
        ball_uv = cast(Tensor, args[0] if args else kwargs["ball_uv"])
        padding_mask = cast(
            Tensor,
            args[4] if len(args) > 4 else kwargs["padding_mask"],
        )
        reference_view_index = cast(
            Tensor,
            args[5] if len(args) > 5 else kwargs["reference_view_index"],
        )
        batch_size, num_views = ball_uv.shape[:2]
        validate_reference_view_index(
            reference_view_index,
            batch_size=batch_size,
            num_views=num_views,
            device=ball_uv.device,
        )
        validate_reference_context_mask(reference_view_index, ~padding_mask)

    @staticmethod
    def build_spatial_coordinates(  # type: ignore[override]
        reference_view_index: Tensor,
        *,
        num_frames: int,
        num_views: int,
        num_detections: int,
        num_queries: int,
    ) -> Tensor:
        """Return exact v2 ``(B*T,Q+V*Q,3)`` reference coordinates."""
        if num_detections != num_queries:
            raise ValueError("num_detections must equal num_queries.")
        return build_full_track_query_spatial_coordinates(
            reference_view_index,
            num_frames=num_frames,
            num_views=num_views,
            num_queries=num_queries,
            selector_mode=ReferenceSelectorMode.REFERENCE,
        )

    def forward(  # type: ignore[override]
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        reference_view_index: Tensor,
    ) -> BLCSTrackingPrediction:
        """Predict with the exact required sixth reference-index tensor."""
        _batch_size, num_views, num_frames, _, _ = ball_uv.shape
        coordinates = self.build_spatial_coordinates(
            reference_view_index,
            num_frames=num_frames,
            num_views=num_views,
            num_detections=self.num_queries,
            num_queries=self.num_queries,
        )
        return self._forward_with_spatial_coordinates(
            ball_uv,
            ball_vis,
            court_kp,
            court_vis,
            padding_mask,
            spatial_coordinates=coordinates,
        )


__all__ = ["BLCSTrackQueryReferenceModel"]

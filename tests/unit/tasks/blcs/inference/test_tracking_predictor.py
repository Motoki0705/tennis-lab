from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.base.data.court_peaks import (
    COURT_SEMANTIC_CLASS_NAMES,
    CourtPeakFrame,
)
from src.tasks.base.model_io import bind_model_io
from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.tasks.blcs.inference.tracking_predictor import BLCSTrackingPredictor
from src.tasks.blcs.model_io import TrackQueryBoundModelIO, TrackQueryModelIOAdapter
from src.tasks.blcs.models import BLCSTrackQueryModel


class _FixedTrackingModel(BLCSTrackQueryModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        ball_uv: Tensor,
        ball_visible: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        observation_state_valid: Tensor,
        spatial_attention_mask: Tensor,
        temporal_attention_mask: Tensor,
        point_attention_mask: Tensor,
        reference_view_mask: Tensor,
    ) -> BLCSTrackingPrediction:
        del (
            ball_visible,
            court_kp,
            court_vis,
            frame_mask,
            observation_state_valid,
            spatial_attention_mask,
            temporal_attention_mask,
            point_attention_mask,
            reference_view_mask,
        )
        batch, _, frames = ball_uv.shape[:3]
        return {
            "position": torch.ones(batch, frames, 2, 3, device=ball_uv.device),
            "presence_logits": torch.tensor([-2.0, 2.0], device=ball_uv.device).expand(
                batch, frames, -1
            ),
        }


class _FixedKP7TrackingModel(BLCSTrackQueryModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        ball_uv: Tensor,
        ball_score: Tensor,
        ball_visible: Tensor,
        court_peak_uv: Tensor,
        court_peak_score: Tensor,
        court_peak_covariance: Tensor,
        court_peak_valid: Tensor,
        frame_mask: Tensor,
        observation_state_valid: Tensor,
        spatial_attention_mask: Tensor,
        temporal_attention_mask: Tensor,
        reference_view_mask: Tensor,
    ) -> BLCSTrackingPrediction:
        del (
            ball_score,
            ball_visible,
            court_peak_uv,
            court_peak_score,
            court_peak_covariance,
            court_peak_valid,
            frame_mask,
            observation_state_valid,
            spatial_attention_mask,
            temporal_attention_mask,
            reference_view_mask,
        )
        batch, _, frames = ball_uv.shape[:3]
        return {
            "position": torch.ones(batch, frames, 2, 3, device=ball_uv.device),
            "presence_logits": torch.ones(batch, frames, 2, device=ball_uv.device),
        }


def _court_peak_frames(frames: int) -> list[CourtPeakFrame]:
    return [
        CourtPeakFrame(
            batch_index=0,
            view_index=0,
            frame_index=frame_index,
            keypoints_pixels=torch.full((7, 2, 2), 5.0),
            scores=torch.ones(7, 2),
            covariance_pixels=torch.eye(2)
            .reshape(1, 1, 2, 2)
            .expand(7, 2, 2, 2),
            valid=torch.ones(7, 2, dtype=torch.bool),
            image_size_hw=(11, 11),
            semantic_class_names=COURT_SEMANTIC_CLASS_NAMES,
        )
        for frame_index in range(frames)
    ]


def test_predictor_returns_cpu_query_presence_and_positions() -> None:
    binding = cast(
        "TrackQueryBoundModelIO",
        bind_model_io(
            _FixedTrackingModel(),
            TrackQueryModelIOAdapter(
                num_court_tokens=14,
                num_queries=2,
                presence_threshold=0.5,
                mask_invisible_observations=True,
            ),
        ),
    )
    predictor = BLCSTrackingPredictor(
        model_io=binding,
        device=torch.device("cpu"),
    )
    shape = (1, 2, 3, 2)

    result = predictor.predict(
        ball_uv=torch.zeros(*shape, 2),
        ball_score=torch.ones(*shape),
        ball_visible=torch.ones(*shape, dtype=torch.bool),
        candidate_mask=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        view_mask=torch.ones(1, 2, dtype=torch.bool),
        reference_view_index=torch.zeros(1, dtype=torch.long),
        denormalize=False,
    )

    assert result.position.shape == (1, 3, 2, 3)
    assert not result.presence[..., 0].any()
    assert result.presence[..., 1].all()
    assert result.position.device.type == "cpu"
    assert result.presence_logits.device.type == "cpu"
    assert result.presence_probability.device.type == "cpu"
    assert result.presence.device.type == "cpu"


def test_predictor_accepts_indexed_court_predictor_frames_directly() -> None:
    binding = cast(
        "TrackQueryBoundModelIO",
        bind_model_io(
            _FixedKP7TrackingModel(),
            TrackQueryModelIOAdapter(
                num_court_tokens=14,
                num_queries=2,
                presence_threshold=0.5,
                mask_invisible_observations=True,
                court_observation_profile="kp7_reference",
            ),
        ),
    )
    predictor = BLCSTrackingPredictor(
        model_io=binding,
        device=torch.device("cpu"),
    )
    shape = (1, 1, 2, 1)

    result = predictor.predict(
        ball_uv=torch.zeros(*shape, 2),
        ball_score=torch.ones(*shape),
        ball_visible=torch.ones(*shape, dtype=torch.bool),
        candidate_mask=torch.ones(*shape, dtype=torch.bool),
        court_peak_frames=_court_peak_frames(2),
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
        view_mask=torch.ones(1, 1, dtype=torch.bool),
        reference_view_index=torch.zeros(1, dtype=torch.long),
        denormalize=False,
    )

    assert result.position.shape == (1, 2, 2, 3)
    assert result.presence.all()

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor, nn

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
        candidate_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        camera_state_valid: Tensor,
        spatial_attention_mask: Tensor,
        object_temporal_state_valid: Tensor,
        object_temporal_attention_mask: Tensor,
        query_temporal_state_valid: Tensor,
        query_temporal_attention_mask: Tensor,
        point_attention_mask: Tensor,
    ) -> BLCSTrackingPrediction:
        del (
            ball_visible,
            candidate_mask,
            court_kp,
            court_vis,
            frame_mask,
            camera_state_valid,
            spatial_attention_mask,
            object_temporal_state_valid,
            object_temporal_attention_mask,
            query_temporal_state_valid,
            query_temporal_attention_mask,
            point_attention_mask,
        )
        batch, _, frames = ball_uv.shape[:3]
        return {
            "position": torch.ones(batch, frames, 2, 3, device=ball_uv.device),
            "presence_logits": torch.tensor([-2.0, 2.0], device=ball_uv.device).expand(
                batch, frames, -1
            ),
        }


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
        ball_visible=torch.ones(*shape, dtype=torch.bool),
        candidate_mask=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        view_mask=torch.ones(1, 2, dtype=torch.bool),
        denormalize=False,
    )

    assert result.position.shape == (1, 3, 2, 3)
    assert not result.presence[..., 0].any()
    assert result.presence[..., 1].all()
    assert result.position.device.type == "cpu"
    assert result.presence_logits.device.type == "cpu"
    assert result.presence_probability.device.type == "cpu"
    assert result.presence.device.type == "cpu"


def test_predictor_is_the_only_boundary_that_pads_short_candidates() -> None:
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
    predictor = BLCSTrackingPredictor(binding, torch.device("cpu"))
    common = {
        "court_kp": torch.zeros(1, 1, 3, 14, 2),
        "court_vis": torch.ones(1, 1, 3, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "view_mask": torch.ones(1, 1, dtype=torch.bool),
        "denormalize": False,
    }

    result = predictor.predict(
        ball_uv=torch.zeros(1, 1, 3, 1, 2),
        ball_visible=torch.ones(1, 1, 3, 1, dtype=torch.bool),
        candidate_mask=torch.ones(1, 1, 3, 1, dtype=torch.bool),
        **common,
    )
    assert result.position.shape == (1, 3, 2, 3)

    with pytest.raises(ValueError, match="exceed model.num_queries"):
        predictor.predict(
            ball_uv=torch.zeros(1, 1, 3, 3, 2),
            ball_visible=torch.ones(1, 1, 3, 3, dtype=torch.bool),
            candidate_mask=torch.ones(1, 1, 3, 3, dtype=torch.bool),
            **common,
        )

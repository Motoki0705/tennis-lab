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
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


class _FixedTrackingModel(BLCSTrackQueryModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> BLCSTrackingPrediction:
        del (
            ball_vis,
            court_kp,
            court_vis,
            padding_mask,
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
        ball_vis=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
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
            ),
        ),
    )
    predictor = BLCSTrackingPredictor(binding, torch.device("cpu"))
    common = {
        "court_kp": torch.zeros(1, 1, 3, 14, 2),
        "court_vis": torch.ones(1, 1, 3, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 1, 3, dtype=torch.bool),
        "denormalize": False,
    }

    result = predictor.predict(
        ball_uv=torch.zeros(1, 1, 3, 1, 2),
        ball_vis=torch.ones(1, 1, 3, 1, dtype=torch.bool),
        **common,
    )
    assert result.position.shape == (1, 3, 2, 3)

    with pytest.raises(ValueError, match="exceed model.num_queries"):
        predictor.predict(
            ball_uv=torch.zeros(1, 1, 3, 3, 2),
            ball_vis=torch.ones(1, 1, 3, 3, dtype=torch.bool),
            **common,
        )


def test_default_v1_tracking_predictor_returns_physical_scale() -> None:
    binding = cast(
        "TrackQueryBoundModelIO",
        bind_model_io(
            _FixedTrackingModel(),
            TrackQueryModelIOAdapter(
                num_court_tokens=14,
                num_queries=2,
                presence_threshold=0.5,
            ),
        ),
    )
    predictor = BLCSTrackingPredictor(binding, torch.device("cpu"))

    result = predictor.predict(
        ball_uv=torch.zeros(1, 1, 2, 2, 2),
        ball_vis=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        court_kp=torch.zeros(1, 1, 2, 14, 2),
        court_vis=torch.ones(1, 1, 2, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 1, 2, dtype=torch.bool),
        denormalize=True,
    )

    torch.testing.assert_close(
        result.position,
        torch.tensor([5.485, 11.885, 1.07]).expand(1, 2, 2, 3),
    )


def test_v2_tracking_predictor_returns_meter_valued_positions() -> None:
    binding = cast(
        "TrackQueryBoundModelIO",
        bind_model_io(
            _FixedTrackingModel(),
            TrackQueryModelIOAdapter(
                num_court_tokens=14,
                num_queries=2,
                presence_threshold=0.5,
            ),
        ),
    )
    contract = resolve_court_coordinate_normalization("v2")
    predictor = BLCSTrackingPredictor(
        binding,
        torch.device("cpu"),
        normalization=contract,
    )

    result = predictor.predict(
        ball_uv=torch.zeros(1, 1, 2, 2, 2),
        ball_vis=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        court_kp=torch.zeros(1, 1, 2, 14, 2),
        court_vis=torch.ones(1, 1, 2, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 1, 2, dtype=torch.bool),
        denormalize=True,
    )

    torch.testing.assert_close(
        result.position,
        torch.tensor(contract.scale_xyz).expand(1, 2, 2, 3),
    )

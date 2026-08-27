from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import Tensor, nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.tasks.blcs.inference.tracking_predictor import BLCSTrackingPredictor
from src.tasks.blcs.model_io import (
    TrackQueryBoundModelIO,
    TrackQueryModelIOAdapter,
    compose_blcs_track_query_model_io,
)
from src.tasks.blcs.model_io.adapters import TrackQueryAblationModelIOAdapter
from src.tasks.blcs.models import (
    BLCSTrackQueryAblationModel,
    BLCSTrackQueryModel,
)
from src.utils.configuration import PathResolver


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

    physical = predictor.predict(
        ball_uv=torch.zeros(*shape, 2),
        ball_vis=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
        denormalize=True,
    )
    torch.testing.assert_close(physical.position, torch.full((1, 3, 2, 3), 11.885))


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
    court_kp = torch.zeros(1, 1, 3, 14, 2)
    court_vis = torch.ones(1, 1, 3, 14, dtype=torch.bool)
    padding_mask = torch.zeros(1, 1, 3, dtype=torch.bool)

    result = predictor.predict(
        ball_uv=torch.zeros(1, 1, 3, 1, 2),
        ball_vis=torch.ones(1, 1, 3, 1, dtype=torch.bool),
        court_kp=court_kp,
        court_vis=court_vis,
        padding_mask=padding_mask,
        denormalize=False,
    )
    assert result.position.shape == (1, 3, 2, 3)

    with pytest.raises(ValueError, match="exceed model.num_queries"):
        predictor.predict(
            ball_uv=torch.zeros(1, 1, 3, 3, 2),
            ball_vis=torch.ones(1, 1, 3, 3, dtype=torch.bool),
            court_kp=court_kp,
            court_vis=court_vis,
            padding_mask=padding_mask,
            denormalize=False,
        )


def test_checkpoint_restoration_dispatches_to_exact_ablation_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = Path("src/tasks/blcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=track_query_ablation_d",
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_stages=4",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.backend=reference",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
            ],
        )
    binding = compose_blcs_track_query_model_io(config)
    checkpoint = tmp_path / "ablation.ckpt"
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        BLCSTrackingPredictor,
        "_ensure_checkpoint",
        classmethod(lambda cls, value, *, resolver: [checkpoint]),
    )
    monkeypatch.setattr(
        "src.tasks.blcs.inference.tracking_predictor.load_checkpoint_config",
        lambda path: config,
    )

    def compose_binding(value: object) -> TrackQueryBoundModelIO:
        observed["config"] = value
        return binding

    def load_module(
        cls: type[BLCSTrackingPredictor],
        path: Path,
        module_type: type[nn.Module],
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, torch.device]:
        del cls, module_type
        observed["path"] = path
        observed["strict"] = kwargs["strict"]
        observed["model_io"] = kwargs["model_io"]
        return SimpleNamespace(model_io=binding), torch.device("cpu")

    monkeypatch.setattr(
        "src.tasks.blcs.inference.tracking_predictor.compose_blcs_track_query_model_io",
        compose_binding,
    )
    monkeypatch.setattr(
        BLCSTrackingPredictor,
        "_load_single_lightning_module",
        classmethod(load_module),
    )

    predictor = BLCSTrackingPredictor.load_from_checkpoint(
        checkpoint,
        resolver=cast("PathResolver", object()),
        device="cpu",
    )

    assert observed == {
        "config": config,
        "path": checkpoint,
        "strict": True,
        "model_io": binding,
    }
    assert type(predictor.model) is BLCSTrackQueryAblationModel
    assert type(predictor.model_io.adapter) is TrackQueryAblationModelIOAdapter

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import Tensor, nn

from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.inference.tracking_predictor import PLCSTrackingPredictor
from src.tasks.plcs.model_io import PLCSTrackQueryIOAdapter, build_plcs_model_io
from src.tasks.plcs.models.plcs_track_query_ablation_model import (
    PLCSTrackQueryAblationModel,
)
from src.utils.configuration import PathResolver


class _FixedTrackingModel(nn.Module):
    def forward(
        self,
        *,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Tensor]:
        del (
            human_vis,
            court_kp,
            court_vis,
            padding_mask,
        )
        batch, _, frames = human_kp.shape[:3]
        rotation = torch.tensor([0.0, 1.0], device=human_kp.device)
        return {
            "position": torch.ones(batch, frames, 2, 3, device=human_kp.device),
            "rotation": rotation.expand(batch, frames, 2, -1),
            "presence_logits": torch.tensor([2.0, -2.0], device=human_kp.device).expand(
                batch, frames, -1
            ),
        }


def test_predictor_returns_cpu_lifecycle_and_yaw_outputs() -> None:
    predictor = PLCSTrackingPredictor(
        model=_FixedTrackingModel(),
        adapter=PLCSTrackQueryIOAdapter(
            model_type=_FixedTrackingModel,
            num_queries=2,
            num_court_tokens=14,
            num_joints=17,
        ),
        device=torch.device("cpu"),
    )
    shape = (1, 2, 3, 2)

    result = predictor.predict(
        human_kp=torch.zeros(*shape, 17, 2),
        human_vis=torch.ones(*shape, 17, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
        tracking_metrics=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
        denormalize=True,
    )

    assert result["position_meters"].shape == (1, 3, 2, 3)
    assert result["presence"][..., 0].all()
    assert not result["presence"][..., 1].any()
    torch.testing.assert_close(
        result["yaw_radians"],
        torch.full((1, 3, 2), torch.pi / 2),
    )
    assert all(value.device.type == "cpu" for value in result.values())


def test_checkpoint_restoration_retains_exact_ablation_model_adapter_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = Path("src/tasks/plcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=track_query_ablation_c",
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_stages=4",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
            ],
        )
    binding = build_plcs_model_io(PLCSTrainingConfig.from_config(config))
    assert isinstance(binding.adapter, PLCSTrackQueryIOAdapter)
    checkpoint = tmp_path / "ablation.ckpt"
    observed: dict[str, object] = {}

    def load_module(
        cls: type[PLCSTrackingPredictor],
        path: Path,
        module_type: type[nn.Module],
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, torch.device]:
        del cls, module_type
        observed["path"] = path
        observed["strict"] = kwargs["strict"]
        observed["weights_only"] = kwargs["weights_only"]
        return (
            SimpleNamespace(model=binding.model, io_adapter=binding.adapter),
            torch.device("cpu"),
        )

    monkeypatch.setattr(
        PLCSTrackingPredictor,
        "_load_single_lightning_module",
        classmethod(load_module),
    )

    predictor = PLCSTrackingPredictor.load_from_checkpoint(
        checkpoint,
        resolver=cast("PathResolver", object()),
        device="cpu",
    )

    assert observed == {
        "path": checkpoint,
        "strict": True,
        "weights_only": False,
    }
    assert type(predictor.model) is PLCSTrackQueryAblationModel
    assert predictor.io_adapter is binding.adapter
    assert predictor.io_adapter.model_type is PLCSTrackQueryAblationModel

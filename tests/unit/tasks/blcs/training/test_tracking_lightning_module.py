from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.tasks.base.training.tracking_benchmark import TrackingFusionBenchmarkResult
from src.tasks.blcs.models.components.observation_fusion import (
    KP7TrackObservationFusion,
)
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)


def _module_without_runtime() -> BLCSTrackingLightningModule:
    return object.__new__(BLCSTrackingLightningModule)


def test_checkpoint_rejects_deleted_group_encoder_contract() -> None:
    checkpoint = {
        "state_dict": {
            "model.group_encoder.proj.layers.0.weight": torch.randn(3, 2),
        }
    }

    with pytest.raises(RuntimeError, match="deleted model.group_encoder"):
        _module_without_runtime().on_load_checkpoint(checkpoint)


def test_checkpoint_requires_explicit_state_dict_mapping() -> None:
    with pytest.raises(TypeError, match="state_dict mapping"):
        _module_without_runtime().on_load_checkpoint({})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_blcs_reference_benchmark_uses_active_kp7_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Fusion(KP7TrackObservationFusion):
        def __init__(self) -> None:
            nn.Module.__init__(self)

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
            return ball_uv, state_valid

    module = _module_without_runtime()
    model = nn.Module()
    model.anchor = nn.Parameter(torch.zeros(1, device="cuda"))
    model.court_observation_profile = "kp7_reference"
    model.observation_encoder = _Fusion().cuda()
    object.__setattr__(module, "model", model)
    captured: dict[str, object] = {}

    def _benchmark(call, *, inputs):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        captured["output"] = call()
        return TrackingFusionBenchmarkResult(1.0, 2.0)

    monkeypatch.setattr(
        "src.tasks.blcs.training.tracking_lightning_module.benchmark_tracking_fusion_cuda",
        _benchmark,
    )
    result = module.benchmark_court_peak_fusion()

    assert result == TrackingFusionBenchmarkResult(1.0, 2.0)
    inputs = captured["inputs"]
    assert isinstance(inputs, dict)
    assert inputs["court_peak_uv"].shape == (1, 3, 64, 7, 4, 2)
    assert inputs["ball_uv"].shape == (1, 3, 64, 4, 2)

"""Focused PLCS KP7 fusion benchmark integration tests."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.tasks.base.training.tracking_benchmark import TrackingFusionBenchmarkResult
from src.tasks.plcs.models.components.observation_fusion import (
    KP7PlayerObservationFusion,
)
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_plcs_reference_benchmark_uses_active_kp7_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Fusion(KP7PlayerObservationFusion):
        def __init__(self) -> None:
            super().__init__(
                hidden_dim=4,
                player_feature_dim=52,
                invisible_init_std=0.02,
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
            return human_kp, camera_state_valid

    module = object.__new__(PLCSTrackingLightningModule)
    model = nn.Module()
    model.anchor = nn.Parameter(torch.zeros(1, device="cuda"))
    model.court_observation_profile = "kp7_reference"
    model.kp7_observation_encoder = _Fusion().cuda()
    object.__setattr__(module, "model", model)
    captured: dict[str, object] = {}

    def _benchmark(call, *, inputs):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        captured["output"] = call()
        return TrackingFusionBenchmarkResult(1.0, 2.0)

    monkeypatch.setattr(
        "src.tasks.plcs.training.tracking_lightning_module.benchmark_tracking_fusion_cuda",
        _benchmark,
    )
    result = module.benchmark_court_peak_fusion()

    assert result == TrackingFusionBenchmarkResult(1.0, 2.0)
    inputs = captured["inputs"]
    assert isinstance(inputs, dict)
    assert inputs["court_peak_uv"].shape == (1, 3, 64, 7, 4, 2)
    assert inputs["player_anchor"].shape == (1, 3, 64, 4, 2)
    assert inputs["player_features"].shape == (1, 3, 64, 4, 52)

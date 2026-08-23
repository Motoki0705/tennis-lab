"""Physical-unit metric and uncertainty tests for SLCS."""

from __future__ import annotations

import pytest
import torch

from src.tasks.slcs.model_io import SLCSDecodedOutput, SLCSTrainingTargets
from src.tasks.slcs.normalization import scalar_position_uncertainty_scale_m
from src.tasks.slcs.training.metrics import SLCSMetrics
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


def _outputs(position: torch.Tensor) -> SLCSDecodedOutput:
    return SLCSDecodedOutput(
        player_position=position.view(1, 1, 1, 3),
        player_rotation=torch.tensor([[[[1.0, 0.0]]]]),
        player_position_log_b=torch.zeros(1, 1, 1),
        player_rotation_log_b=torch.zeros(1, 1, 1),
        ball_position=position.view(1, 1, 3),
        ball_position_log_b=torch.zeros(1, 1),
    )


def _targets() -> SLCSTrainingTargets:
    return SLCSTrainingTargets(
        target_player_position=torch.zeros(1, 1, 1, 3),
        target_player_rotation=torch.tensor([[[[1.0, 0.0]]]]),
        target_ball_position=torch.zeros(1, 1, 3),
        player_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        player_weight=torch.ones(1, 1, 1),
        ball_mask=torch.ones(1, 1, dtype=torch.bool),
        ball_weight=torch.ones(1, 1),
        padding_mask=torch.zeros(1, 1, dtype=torch.bool),
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_slcs_metrics_report_identical_meter_error_for_each_version(
    version: str,
) -> None:
    contract = resolve_court_coordinate_normalization(version)
    physical_error = torch.tensor([1.0, 2.0, 2.0])
    normalized_error = physical_error / torch.tensor(contract.scale_xyz)

    metrics = SLCSMetrics(contract)
    batch = metrics.update(_outputs(normalized_error), _targets())
    aggregate = metrics.compute()

    assert batch["player_position_error_m"] == pytest.approx(3.0)
    assert batch["ball_position_error_m"] == pytest.approx(3.0)
    assert aggregate["player_position_error_m"] == pytest.approx(3.0)
    assert aggregate["ball_position_error_m"] == pytest.approx(3.0)
    assert aggregate["player_position_pred_b_m"] == pytest.approx(
        scalar_position_uncertainty_scale_m(contract)
    )
    assert aggregate["ball_position_pred_b_m"] == pytest.approx(
        scalar_position_uncertainty_scale_m(contract)
    )


def test_slcs_scalar_uncertainty_preserves_v1_mean_and_uses_v2_common_scale() -> None:
    v1 = resolve_court_coordinate_normalization("v1")
    v2 = resolve_court_coordinate_normalization("v2")

    assert scalar_position_uncertainty_scale_m(v1) == pytest.approx(
        sum(v1.scale_xyz) / 3.0
    )
    assert scalar_position_uncertainty_scale_m(v2) == pytest.approx(11.885)

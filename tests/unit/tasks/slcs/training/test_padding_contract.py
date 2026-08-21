"""Loss and metric exclusion tests for the SLCS padding contract."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.tasks.slcs.model_io import SLCSDecodedOutput, SLCSTrainingTargets
from src.tasks.slcs.training.losses import (
    SLCSLoss,
    SLCSLossConfig,
    build_slcs_loss_inputs,
)
from src.tasks.slcs.training.metrics import SLCSMetrics


def _output() -> SLCSDecodedOutput:
    return SLCSDecodedOutput(
        player_position=torch.zeros(1, 1, 3, 3),
        player_rotation=torch.tensor([[[[1.0, 0.0]] * 3]]),
        player_position_log_b=torch.zeros(1, 1, 3),
        player_rotation_log_b=torch.zeros(1, 1, 3),
        ball_position=torch.zeros(1, 3, 3),
        ball_position_log_b=torch.zeros(1, 3),
    )


def _targets() -> SLCSTrainingTargets:
    real_frames = torch.tensor([[True, True, False]])
    return SLCSTrainingTargets(
        target_player_position=torch.zeros(1, 1, 3, 3),
        target_player_rotation=torch.tensor([[[[1.0, 0.0]] * 3]]),
        target_ball_position=torch.zeros(1, 3, 3),
        player_mask=real_frames.unsqueeze(1),
        player_weight=real_frames.unsqueeze(1).float(),
        ball_mask=real_frames,
        ball_weight=real_frames.float(),
        padding_mask=~real_frames,
    )


def _loss() -> SLCSLoss:
    return SLCSLoss(
        SLCSLossConfig(
            player_position_weight=1.0,
            player_rotation_weight=1.0,
            player_angle_weight=1.0,
            ball_position_weight=1.0,
            player_position_nll_weight=1.0,
            player_rotation_nll_weight=1.0,
            ball_position_nll_weight=1.0,
            player_position_smoothness_weight=1.0,
            ball_position_smoothness_weight=1.0,
            ground_penetration_weight=1.0,
            smoothness_order=1,
        )
    )


def _mutate_only_padding(output: SLCSDecodedOutput) -> SLCSDecodedOutput:
    player_position = output.player_position.clone()
    player_position[:, :, -1] = torch.tensor([10_000.0, -10_000.0, -10_000.0])
    player_rotation = output.player_rotation.clone()
    player_rotation[:, :, -1] = torch.tensor([-1.0, 0.0])
    player_position_log_b = output.player_position_log_b.clone()
    player_position_log_b[:, :, -1] = 10.0
    player_rotation_log_b = output.player_rotation_log_b.clone()
    player_rotation_log_b[:, :, -1] = 10.0
    ball_position = output.ball_position.clone()
    ball_position[:, -1] = torch.tensor([10_000.0, -10_000.0, -10_000.0])
    ball_position_log_b = output.ball_position_log_b.clone()
    ball_position_log_b[:, -1] = 10.0
    return replace(
        output,
        player_position=player_position,
        player_rotation=player_rotation,
        player_position_log_b=player_position_log_b,
        player_rotation_log_b=player_rotation_log_b,
        ball_position=ball_position,
        ball_position_log_b=ball_position_log_b,
    )


def test_padding_predictions_are_excluded_from_every_loss_term() -> None:
    targets = _targets()
    baseline = _loss()(build_slcs_loss_inputs(_output(), targets))
    modified = _loss()(
        build_slcs_loss_inputs(_mutate_only_padding(_output()), targets)
    )

    assert baseline.keys() == modified.keys()
    for name in baseline:
        torch.testing.assert_close(modified[name], baseline[name])


def test_padding_predictions_are_excluded_from_metrics() -> None:
    targets = _targets()
    baseline_metrics = SLCSMetrics()
    modified_metrics = SLCSMetrics()

    baseline_batch = baseline_metrics.update(_output(), targets)
    modified_batch = modified_metrics.update(_mutate_only_padding(_output()), targets)

    assert modified_batch == pytest.approx(baseline_batch)
    assert modified_metrics.compute() == pytest.approx(baseline_metrics.compute())

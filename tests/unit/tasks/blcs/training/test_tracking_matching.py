from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.blcs.training.tracking_losses import BLCSTrackingLoss
from src.tasks.blcs.training.tracking_matching import match_ball_tracks


def test_matching_uses_same_axis_balance_as_position_loss() -> None:
    pred_position = torch.tensor(
        [[[[0.0, 0.0, 0.0], [10.0, 0.0, 10.0]]]]
    )
    pred_presence = torch.zeros(1, 1, 2)
    target_position = torch.tensor(
        [[[[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]]]
    )
    present = torch.ones(1, 1, 2, dtype=torch.bool)

    assignments = match_ball_tracks(
        pred_position,
        pred_presence,
        target_position,
        present,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        position_cost_weight=1.0,
        presence_cost_weight=0.0,
        presence_inactive_weight=0.25,
        presence_active_weight=1.0,
        presence_transition_weight=2.0,
        transition_radius=2,
        position_axis_weights=torch.tensor((1.0, 1.0, 0.1)),
    )

    query_indices, target_indices = assignments[0]
    torch.testing.assert_close(query_indices, torch.tensor([0, 1]))
    torch.testing.assert_close(target_indices, torch.tensor([0, 1]))


def _tracking_config() -> SimpleNamespace:
    return SimpleNamespace(
        position_weight=1.0,
        presence_weight=1.0,
        presence_inactive_weight=0.25,
        presence_active_weight=1.0,
        presence_transition_weight=2.0,
        transition_radius=2,
        smoothness_weight=0.0,
        gravity_weight=1.0,
        gravity_target=-0.01,
        match_position_weight=1.0,
        match_presence_weight=0.5,
        position_axis_weights=[1.0, 1.0, 0.5],
        position_axis_weights_v2=[1.0, 1.0, 1.0],
        position_huber_beta_v1=1.0,
        position_huber_transition_m_v2=1.0,
    )


def test_tracking_loss_and_hungarian_cost_select_version_specific_defaults() -> None:
    v1 = BLCSTrackingLoss(_tracking_config(), normalization="v1")
    v2 = BLCSTrackingLoss(_tracking_config(), normalization="v2")

    torch.testing.assert_close(
        v1.position_axis_weights,
        torch.tensor([1.0, 1.0, 0.5]),
    )
    torch.testing.assert_close(v2.position_axis_weights, torch.ones(3))
    assert v1.position_beta == 1.0
    assert v2.position_beta == pytest.approx(1.0 / 11.885)
    assert v1.gravity_target == -0.01
    assert v2.gravity_target == pytest.approx(-9.81 * (1.0 / 30.0) ** 2 / 11.885)

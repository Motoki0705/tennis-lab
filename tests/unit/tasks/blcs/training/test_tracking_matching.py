from __future__ import annotations

import torch

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

from __future__ import annotations

import torch

from src.tasks.base.training.tracking_lifecycle import (
    lifecycle_transition_mask,
    weighted_presence_bce_with_logits,
)


def test_transition_mask_marks_birth_and_death_neighborhoods_only_on_valid_frames() -> None:
    presence = torch.tensor(
        [False, False, True, True, False, False, False], dtype=torch.bool
    )
    valid = torch.tensor([True, True, True, True, True, True, False])

    transition = lifecycle_transition_mask(presence, valid, radius=1)

    assert transition.tolist() == [False, True, True, True, True, True, False]


def test_weighted_presence_loss_emphasizes_transition_frames() -> None:
    logits = torch.zeros(1, 5, 1)
    presence = torch.tensor([[[0], [1], [1], [0], [0]]], dtype=torch.bool)
    valid = torch.ones_like(presence)

    loss = weighted_presence_bce_with_logits(
        logits,
        presence,
        valid,
        inactive_weight=0.25,
        active_weight=1.0,
        transition_weight=2.0,
        transition_radius=0,
    )

    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))

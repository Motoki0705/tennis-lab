"""Tests for SLCS masks derived from public padding polarity."""

from __future__ import annotations

import pytest
import torch

from src.tasks.slcs.models.components.padding import build_slcs_padding_masks


def test_builds_raw_validity_and_internal_keep_masks_from_padding() -> None:
    padding_mask = torch.tensor([[False, False, True]])
    dino_padding_mask = torch.tensor([[False, True]])

    masks = build_slcs_padding_masks(
        padding_mask,
        dino_padding_mask,
        num_entities=2,
        dino_tokens_per_sample=3,
    )

    assert torch.equal(masks.frame_state_valid, ~padding_mask)
    assert torch.equal(
        masks.entity_state_valid,
        torch.tensor([[[True, True], [True, True], [False, False]]]),
    )
    entity_keep = masks.entity_attention_keep_mask.reshape(1, 3, 2, 2)
    assert entity_keep[:, :2].all()
    assert entity_keep[:, 2, :, 0].all()
    assert not entity_keep[:, 2, :, 1].any()
    time_keep = masks.time_attention_keep_mask.reshape(1, 2, 3, 3)
    assert time_keep[..., :2].all()
    assert not time_keep[..., 2].any()
    assert torch.equal(masks.dino_sample_valid, ~dino_padding_mask)
    assert masks.dino_batch_has_evidence.tolist() == [True]
    assert masks.dino_attention_keep_mask[..., :3].all()
    assert not masks.dino_attention_keep_mask[..., 3:].any()


def test_all_dino_padding_repairs_attention_only_not_raw_validity() -> None:
    masks = build_slcs_padding_masks(
        torch.zeros(2, 3, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool),
        num_entities=3,
        dino_tokens_per_sample=4,
    )

    assert not masks.dino_sample_valid.any()
    assert not masks.dino_batch_has_evidence.any()
    assert masks.dino_attention_keep_mask[..., 0].all()
    assert not masks.dino_attention_keep_mask[..., 1:].any()


@pytest.mark.parametrize(
    ("padding_mask", "dino_padding_mask", "error", "message"),
    [
        (
            torch.zeros(1, 2),
            torch.zeros(1, 1, dtype=torch.bool),
            TypeError,
            "torch.bool",
        ),
        (
            torch.zeros(1, 2, 3, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
            ValueError,
            r"shape \(B,T\)",
        ),
        (
            torch.zeros(2, 3, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
            ValueError,
            "batch axis",
        ),
    ],
)
def test_rejects_malformed_padding_masks(
    padding_mask: torch.Tensor,
    dino_padding_mask: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        build_slcs_padding_masks(
            padding_mask,
            dino_padding_mask,
            num_entities=2,
            dino_tokens_per_sample=3,
        )

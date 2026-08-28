"""Tests for BLCS model-internal padding-mask derivation."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.models.components.padding import (
    build_axial_padding_masks,
    build_multiview_padding_masks,
    build_single_view_padding_masks,
    mask_trajectory_outputs,
)


def test_single_view_all_padding_repairs_attention_but_keeps_frames_invalid() -> None:
    masks = build_single_view_padding_masks(
        torch.ones(2, 3, dtype=torch.bool),
        num_court_tokens=4,
    )

    assert not masks.frame_valid.any()
    assert masks.attention_keep_mask.shape == (2, 7, 7)
    assert masks.attention_keep_mask.any(dim=-1).all()


def test_multiview_nonrectangular_padding_never_reenables_context_holes() -> None:
    padding_mask = torch.tensor(
        [[[False, True, False], [True, False, True]]],
        dtype=torch.bool,
    )
    masks = build_multiview_padding_masks(
        padding_mask,
        num_court_tokens=2,
    )

    assert torch.equal(masks.context_valid, ~padding_mask)
    assert masks.frame_valid.tolist() == [[True, True, True]]
    per_context = masks.frame_token_valid.reshape(1, 3, 2, 3)
    assert per_context[0, 0, 0].all()
    assert not per_context[0, 0, 1].any()
    assert not per_context[0, 1, 0].any()
    assert per_context[0, 1, 1].all()
    assert masks.cross_attention_keep_mask.any(dim=-1).all()


def test_axial_all_padding_masks_are_finite_repairs_with_raw_frames_invalid() -> None:
    masks = build_axial_padding_masks(
        torch.ones(1, 2, 3, dtype=torch.bool),
        time_window_radius=1,
    )

    assert not masks.context_valid.any()
    assert not masks.frame_valid.any()
    assert masks.camera_attention_keep_mask.any(dim=-1).all()
    assert masks.time_attention_keep_mask.any(dim=-1).all()
    assert masks.sliding_attention_keep_mask.any(dim=-1).all()


def test_axial_sliding_attention_restricts_valid_keys_to_configured_radius() -> None:
    masks = build_axial_padding_masks(
        torch.zeros(1, 1, 5, dtype=torch.bool),
        time_window_radius=1,
    )

    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    )
    assert torch.equal(masks.sliding_attention_keep_mask[0], expected)


def test_mask_trajectory_outputs_zeros_only_padded_frames() -> None:
    value = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    frame_valid = torch.tensor([[True, False, True], [False, False, True]])

    masked = mask_trajectory_outputs({"position": value}, frame_valid)["position"]

    torch.testing.assert_close(masked[frame_valid], value[frame_valid])
    assert not masked[~frame_valid].any()


@pytest.mark.parametrize(
    "padding_mask",
    [torch.zeros(1, 2), torch.zeros(1, 2, 3, dtype=torch.float32)],
)
def test_multiview_padding_builder_rejects_wrong_contract(
    padding_mask: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError), match="padding_mask"):
        build_multiview_padding_masks(padding_mask, num_court_tokens=2)

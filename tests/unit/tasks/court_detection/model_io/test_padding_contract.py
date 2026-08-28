"""Padding-aware valid-region boundaries for Court pose consistency."""

from __future__ import annotations

import torch

from src.tasks.court_detection.model_io.adapters import CourtPoseModelIOAdapter
from src.utils.data.heatmaps import heatmaps_to_soft_argmax


def test_valid_region_mask_excludes_pose_patch_alignment_from_soft_argmax() -> None:
    logits = torch.full((1, 1, 8, 10), -100.0, requires_grad=True)
    # The largest score is deliberately placed in the right/bottom alignment
    # border.  It must not affect either the coordinate or its gradient.
    with torch.no_grad():
        logits[:, :, 7, 9] = 100.0
        logits[:, :, 2, 3] = 10.0

    mask = CourtPoseModelIOAdapter._valid_region_mask(
        image_size=torch.tensor([[8, 10]], dtype=torch.long),
        content_size_hw=torch.tensor([[6, 7]], dtype=torch.long),
        logits=logits,
    )
    assert mask.shape == logits.shape
    assert bool(mask[:, :, :6, :7].all())
    assert not bool(mask[:, :, 6:, :].any())
    assert not bool(mask[:, :, :, 7:].any())

    coordinates = heatmaps_to_soft_argmax(logits, valid_mask=mask)
    coordinates.sum().backward()

    torch.testing.assert_close(
        coordinates[0, 0],
        torch.tensor([3.0 / 9.0, 2.0 / 7.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert logits.grad is not None
    assert not bool(logits.grad[:, :, 6:, :].any())
    assert not bool(logits.grad[:, :, :, 7:].any())

from __future__ import annotations

import torch

from src.tasks.plcs.data.dataset import collate_plcs_batch


def _sample(*, views: int, frames: int) -> dict[str, torch.Tensor]:
    return {
        "human_kp": torch.rand(views, frames, 17, 2),
        "court_kp": torch.rand(views, frames, 20, 2),
        "human_vis": torch.ones(views, frames, 17),
        "court_vis": torch.ones(views, frames, 20),
        "padding_mask": torch.zeros(views, frames, dtype=torch.bool),
        "position": torch.rand(frames, 3),
        "rotation": torch.rand(frames, 2),
    }


def test_collate_uses_true_only_for_added_view_and_time_padding() -> None:
    batch = collate_plcs_batch(
        [_sample(views=1, frames=2), _sample(views=2, frames=3)]
    )
    padding_mask = batch["padding_mask"]

    assert padding_mask.dtype == torch.bool
    assert padding_mask.shape == (2, 2, 3)
    assert not padding_mask[1].any()
    assert not padding_mask[0, 0, :2].any()
    assert padding_mask[0, 0, 2]
    assert padding_mask[0, 1].all()

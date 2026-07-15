from __future__ import annotations

import torch

from src.tasks.plcs.data.dataset import collate_plcs_batch


def _line_sample(*, views: int, frames: int) -> dict[str, torch.Tensor]:
    return {
        "human_kp": torch.rand(views, frames, 17, 2),
        "human_vis": torch.ones(views, frames, 17),
        "human_mask": torch.ones(views, frames),
        "court_line_map": torch.rand(views, frames, 1, 24, 40),
        "position": torch.rand(frames, 3),
        "rotation": torch.rand(frames, 2),
    }


def test_collate_line_samples_preserves_contract_and_zero_padding() -> None:
    batch = collate_plcs_batch(
        [_line_sample(views=2, frames=4), _line_sample(views=1, frames=2)]
    )

    assert batch["court_line_map"].shape == (2, 2, 4, 1, 24, 40)
    assert "court_kp" not in batch
    assert "court_vis" not in batch
    assert torch.all(batch["court_line_map"][1, 1] == 0)
    assert torch.all(batch["court_line_map"][1, 0, 2:] == 0)

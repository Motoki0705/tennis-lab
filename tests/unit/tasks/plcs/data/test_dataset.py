from __future__ import annotations

import torch

from src.tasks.plcs.data.dataset import collate_plcs_batch


def _line_sample(
    *, views: int, frames: int, max_lines: int = 6
) -> dict[str, torch.Tensor]:
    return {
        "human_kp": torch.rand(views, frames, 17, 2),
        "human_vis": torch.ones(views, frames, 17),
        "human_mask": torch.ones(views, frames),
        "court_lines": torch.rand(views, frames, max_lines, 4),
        "position": torch.rand(frames, 3),
        "rotation": torch.rand(frames, 2),
    }


def test_collate_line_samples_preserves_contract_and_zero_padding() -> None:
    batch = collate_plcs_batch(
        [_line_sample(views=2, frames=4), _line_sample(views=1, frames=2)]
    )

    assert batch["court_lines"].shape == (2, 2, 4, 6, 4)
    assert "court_kp" not in batch
    assert "court_vis" not in batch
    assert torch.all(batch["court_lines"][1, 1] == 0)
    assert torch.all(batch["court_lines"][1, 0, 2:] == 0)

from __future__ import annotations

import torch

from src.tasks.blcs.data.dataset import collate_multiview_trajectories
from src.tasks.blcs.data.types import BLCSMultiViewSample


def _line_sample(*, views: int, frames: int) -> BLCSMultiViewSample:
    return {
        "ball_uv": torch.rand(views, frames, 2),
        "ball_vis": torch.ones(views, frames),
        "ball_mask": torch.ones(views, frames),
        "court_line_map": torch.rand(views, frames, 1, 24, 40),
        "position_3d": torch.rand(frames, 3),
        "velocity_3d": torch.rand(frames, 3),
        "seq_len": torch.tensor(frames),
        "camera_R": torch.eye(3).expand(views, -1, -1).clone(),
        "camera_C": torch.zeros(views, 3),
        "camera_f": torch.ones(views),
        "camera_cx": torch.ones(views),
        "camera_cy": torch.ones(views),
        "camera_w": torch.ones(views),
        "camera_h": torch.ones(views),
    }


def test_collate_line_samples_preserves_contract_and_zero_padding() -> None:
    batch = collate_multiview_trajectories(
        [_line_sample(views=2, frames=4), _line_sample(views=1, frames=2)]
    )

    assert batch["court_line_map"].shape == (2, 2, 4, 1, 24, 40)
    assert "court_kp" not in batch
    assert "court_vis" not in batch
    assert torch.all(batch["court_line_map"][1, 1] == 0)
    assert torch.all(batch["court_line_map"][1, 0, 2:] == 0)

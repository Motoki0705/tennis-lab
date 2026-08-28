from __future__ import annotations

import pytest
import torch

from src.tasks.plcs.data.dataset import collate_plcs_batch


def _sample(
    *, views: int, frames: int, reprojection: bool = False
) -> dict[str, torch.Tensor]:
    sample = {
        "human_kp": torch.rand(views, frames, 17, 2),
        "court_kp": torch.rand(views, frames, 20, 2),
        "human_vis": torch.ones(views, frames, 17),
        "court_vis": torch.ones(views, frames, 20),
        "padding_mask": torch.zeros(views, frames, dtype=torch.bool),
        "position": torch.rand(frames, 3),
        "rotation": torch.rand(frames, 2),
    }
    if reprojection:
        sample.update(
            {
                "human_kp_target": sample["human_kp"].clone(),
                "human_vis_target": sample["human_vis"].clone(),
                "camera_R": torch.eye(3).expand(views, -1, -1).clone(),
                "camera_C": torch.zeros(views, 3),
                "camera_f": torch.full((views,), 800.0),
                "camera_cx": torch.full((views,), 640.0),
                "camera_cy": torch.full((views,), 360.0),
                "camera_w": torch.full((views,), 1280.0),
                "camera_h": torch.full((views,), 720.0),
            }
        )
    return sample


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


def test_collate_pads_clean_reprojection_targets_and_cameras() -> None:
    batch = collate_plcs_batch(
        [
            _sample(views=1, frames=2, reprojection=True),
            _sample(views=2, frames=3, reprojection=True),
        ]
    )

    assert batch["human_kp_target"].shape == (2, 2, 3, 17, 2)
    assert batch["human_vis_target"].shape == (2, 2, 3, 17)
    assert batch["camera_R"].shape == (2, 2, 3, 3)
    assert not batch["human_vis_target"][0, 0, 2].any()
    assert not batch["human_vis_target"][0, 1].any()
    torch.testing.assert_close(batch["camera_f"][0, 1], torch.tensor(0.0))
    torch.testing.assert_close(batch["camera_w"][0, 1], torch.tensor(1.0))
    torch.testing.assert_close(batch["camera_h"][0, 1], torch.tensor(1.0))


def test_collate_rejects_partial_or_mixed_reprojection_groups() -> None:
    partial = _sample(views=1, frames=2)
    partial["camera_R"] = torch.eye(3).unsqueeze(0)
    with pytest.raises(ValueError, match="complete group"):
        collate_plcs_batch([partial])

    with pytest.raises(ValueError, match="cannot mix"):
        collate_plcs_batch(
            [
                _sample(views=1, frames=2, reprojection=True),
                _sample(views=1, frames=2),
            ]
        )

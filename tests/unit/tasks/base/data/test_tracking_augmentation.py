from __future__ import annotations

import torch

from src.tasks.base.data.tracking_augmentation import permute_court_keypoint_sets


def test_court_permutation_is_shared_across_time_and_preserves_point_visibility_pairs() -> None:
    torch.manual_seed(13)
    point_ids = torch.arange(14, dtype=torch.float32)
    court_kp = torch.stack([point_ids, point_ids + 100], dim=-1)
    court_kp = court_kp[None, None].expand(2, 3, -1, -1).clone()
    court_vis = (point_ids.long() % 3 != 0)[None, None].expand(2, 3, -1).clone()

    permuted_kp, permuted_vis = permute_court_keypoint_sets(
        court_kp,
        court_vis,
        {"enabled": True, "prob": 1.0},
    )

    assert not torch.equal(permuted_kp, court_kp)
    for view_index in range(2):
        for frame_index in range(1, 3):
            torch.testing.assert_close(
                permuted_kp[view_index, frame_index],
                permuted_kp[view_index, 0],
            )
            assert torch.equal(
                permuted_vis[view_index, frame_index],
                permuted_vis[view_index, 0],
            )
        ids = permuted_kp[view_index, 0, :, 0].long()
        assert sorted(ids.tolist()) == list(range(14))
        assert torch.equal(permuted_vis[view_index, 0], ids % 3 != 0)


"""Independent rotation / pose trunk depths in PLCSMultiViewAxialSplitModel.

Issue #535 follow-up: the split model exposes ``rot_num_task_layers`` /
``pose_num_task_layers`` so the rotation and pose trunks can be sized
independently. ``None`` must preserve the symmetric ``num_task_layers`` default
(backward compatible).
"""

from __future__ import annotations

import pytest

from src.tasks.plcs.models.plcs_multiview_axial_split_model import (
    PLCSMultiViewAxialSplitModel,
)


def _make(**kwargs: object) -> PLCSMultiViewAxialSplitModel:
    return PLCSMultiViewAxialSplitModel(
        hidden_dim=64,
        num_heads=4,
        rope_dim=16,
        num_task_layers=4,
        **kwargs,  # type: ignore[arg-type]
    )


def test_symmetric_default_uses_num_task_layers() -> None:
    m = _make()
    assert m.rot_num_task_layers == 4
    assert m.pose_num_task_layers == 4
    assert len(m.rot_camera_layers) == len(m.rot_time_layers) == 4
    assert len(m.pose_camera_layers) == len(m.pose_time_layers) == 4


def test_independent_rot_and_pose_depths() -> None:
    m = _make(rot_num_task_layers=7, pose_num_task_layers=3)
    assert len(m.rot_camera_layers) == len(m.rot_time_layers) == 7
    assert len(m.pose_camera_layers) == len(m.pose_time_layers) == 3
    # A deeper rotation trunk must add parameters over the symmetric baseline.
    assert sum(p.numel() for p in m.parameters()) > sum(
        p.numel() for p in _make().parameters()
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "bad", [{"rot_num_task_layers": 0}, {"pose_num_task_layers": -1}]
)
def test_nonpositive_depth_raises(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        _make(**bad)

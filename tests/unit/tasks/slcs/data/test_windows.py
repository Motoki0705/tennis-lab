"""Tests for SLCS temporal window planning."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.slcs.data.windows import WindowPlan, plan_windows, select_window_tokens


def test_plan_windows_covers_all_frames_with_tail_anchor() -> None:
    plans = plan_windows(100, window_size=32, stride=16)
    assert plans[0].start == 0
    assert plans[-1].start == 100 - 32
    covered: NDArray[np.bool_] = np.zeros(100, dtype=bool)
    for plan in plans:
        assert plan.pad == 0
        covered[plan.start : plan.start + plan.length] = True
    assert covered.all()


def test_plan_windows_short_clip_pads() -> None:
    plans = plan_windows(10, window_size=16, stride=8)
    assert len(plans) == 1
    plan = plans[0]
    assert (plan.start, plan.length, plan.pad) == (0, 10, 6)
    mask = plan.frame_mask()
    assert mask[:10].all() and not mask[10:].any()
    idx = plan.frame_indices()
    assert idx[9] == 9
    assert (idx[10:] == 9).all()  # padded slots repeat the last real frame


def test_plan_windows_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        plan_windows(0, window_size=8, stride=4)
    with pytest.raises(ValueError):
        plan_windows(10, window_size=8, stride=0)
    with pytest.raises(ValueError):
        WindowPlan(start=0, length=0, window_size=8)
    with pytest.raises(ValueError):
        WindowPlan(start=-1, length=4, window_size=8)


def test_select_window_tokens_matches_only_real_frames() -> None:
    frame_idx = np.array([0, 10, 20, 30], dtype=np.int64)
    plan = WindowPlan(start=8, length=16, window_size=16)  # frames [8, 24)
    sel = select_window_tokens(frame_idx, plan)
    assert frame_idx[sel].tolist() == [10, 20]

    padded = WindowPlan(start=28, length=4, window_size=16)  # frames [28, 32)
    sel_padded = select_window_tokens(frame_idx, padded)
    assert frame_idx[sel_padded].tolist() == [30]

    empty = select_window_tokens(np.array([100], dtype=np.int64), plan)
    assert empty.size == 0

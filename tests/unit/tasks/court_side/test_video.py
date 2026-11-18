"""Clip-local motion filtering and per-frame weighting on independent camera timelines."""
from dataclasses import replace

import numpy as np
import pytest

from src.tasks.court_side.hypothesis import collect_side_evidence, score_hypothesis
from src.tasks.court_side.video import pool_side_evidence
from tests.unit.tasks.court_side.test_hypothesis import (
    CONFIG,
    PHYSICAL,
    local_calibrations,
    observe,
    rally,
)


def test_pool_weights_distinct_source_frames_and_matches_unranked_assignments():
    cameras = PHYSICAL[:3]
    local, _ = local_calibrations(cameras)
    uv, visible = observe(cameras, rally(13))
    # A stationary observation at the end of the first clip is frame zero again
    # in the second. They are different source frames with the same pixels.
    static = np.repeat(uv[:, :1], 80, axis=1)
    a = collect_side_evidence(local, "a", static, np.ones((3, 80), bool), CONFIG)
    # A different physical assignment for part of B changes the cost and support.
    alternate, _ = observe((cameras[0], cameras[1].half_turned(True), cameras[2]), rally(13))
    uv[1, 4:] = alternate[1, 4:]
    visible[2, -3:] = False
    b = collect_side_evidence(local, "a", uv, visible, CONFIG)
    empty = collect_side_evidence(local, "a", uv, np.zeros_like(visible), CONFIG)
    result = pool_side_evidence({"B": b, "A": a, "empty": empty})
    assert result.frames == 14 and result.pair_record() == {"a-b": 14, "a-c": 11, "b-c": 11}
    assert a.hypotheses[0].view_half_turns != b.hypotheses[0].view_half_turns
    points = np.concatenate((static[:, :1], uv), axis=1)
    mask = np.concatenate((np.ones((3, 1), bool), visible), axis=1)
    for h in result.hypotheses:
        expected = score_hypothesis(local, h.view_half_turns, points, mask, CONFIG)
        assert (h.cost, h.support) == pytest.approx((expected.cost, expected.support), abs=1e-12)
    assert pool_side_evidence({"empty": empty}).hypotheses == ()
    with pytest.raises(ValueError, match="reference"):
        pool_side_evidence({"A": a, "B": replace(b, reference_camera="b")})

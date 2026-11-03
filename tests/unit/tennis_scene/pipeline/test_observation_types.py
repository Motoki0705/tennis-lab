"""Camera/carrier provenance and per-frame identity grouping."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.observation_types import (
    ObjectObservations,
    group_observations,
)


def observations() -> ObjectObservations:
    v, t, d, j = np.indices((3, 3, 2, 2))
    code = 100 * v + 10 * t + 3 * d + j
    uv = np.stack((code + 1, code + 2), -1).astype(np.float32)
    return ObjectObservations(
        ("west", "east", "high"), (640, 480), 30., uv,
        (0.4 + code / 1000).astype(np.float32), np.ones((3, 3, 2), bool),
        np.array([[41, -1], [88, 6], [9, 105]], np.int64),
        np.stack((code[..., 0], code[..., 0] + 1, code[..., 0] + 2), -1).astype(np.float32),
    )


def test_select_views_moves_every_array_with_its_camera() -> None:
    raw = observations()
    selected = raw.select_views((2, 0))
    assert selected.camera_ids == ("high", "west")
    np.testing.assert_array_equal(selected.local_track_ids, [[9, 105], [41, -1]])
    for view, original in enumerate((2, 0)):
        for name in ("uv_px", "confidence", "observed", "boxes_xys"):
            np.testing.assert_array_equal(getattr(selected, name)[view], getattr(raw, name)[original])


@pytest.mark.parametrize("indices", [(-1,), (3,), (0, 0)])
def test_select_views_rejects_invalid_or_duplicate_source_indices(indices: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="view indices"):
        observations().select_views(indices)


def test_grouping_follows_frame_identity_switch_and_joint_support() -> None:
    raw = observations()
    ids = np.array([
        [[17, -1], [17, -1], [5, -1]],
        [[5, 17], [17, 5], [17, -1]],
        [[17, 5], [5, 17], [-1, 17]],
    ], np.int64)
    raw.observed[2, 1, 0] = False  # finite pixels without a detector observation
    raw.confidence[1, 2, 0, 0] = .1
    raw.confidence[0, 0, 0, 1] = 1.2  # visible confidence is clipped to 1
    raw.uv_px[2, 2, 1, 1, 0] = 640  # image boundary is outside
    before = ids.copy()
    grouped = group_observations(raw, ids, threshold=.3)
    np.testing.assert_array_equal(grouped.identities, [5, 17])
    # Oracle: explicit carrier rows, including absence, on (identity, camera, frame).
    expected_raw = np.array([
        [[-1, -1, 0], [0, 1, -1], [1, 0, -1]],
        [[0, 0, -1], [1, 0, 0], [0, 1, 1]],
    ])
    np.testing.assert_array_equal(grouped.raw_indices, expected_raw)
    for p, v, t, j in np.ndindex(2, 3, 3, 2):
        carrier = expected_raw[p, v, t]
        visible = carrier >= 0 and (p, v, t, j) not in {(0, 2, 1, 0), (0, 2, 1, 1), (1, 1, 2, 0), (1, 2, 2, 1)}
        assert grouped.visibility[p, v, t, j] == visible
        if visible:
            code = 100 * v + 10 * t + 3 * carrier + j
            np.testing.assert_array_equal(grouped.uv_px[p, v, t, j], [code + 1, code + 2])
            expected_confidence = 1. if (p, v, t, j) == (1, 0, 0, 1) else .4 + code / 1000
            assert grouped.confidence[p, v, t, j] == pytest.approx(expected_confidence)
        else:
            np.testing.assert_array_equal(grouped.uv_px[p, v, t, j], 0)
            assert grouped.confidence[p, v, t, j] == 0
    np.testing.assert_array_equal(ids, before)


def test_grouping_rejects_two_carriers_of_one_identity_in_one_camera_frame() -> None:
    ids: NDArray[np.int64] = np.full((3, 3, 2), -1, np.int64)
    ids[1, 2] = 17
    with pytest.raises(ValueError, match="multiple raw detections"):
        group_observations(observations(), ids, threshold=.3)

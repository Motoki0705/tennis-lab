"""Regression coverage for the shared immutable CourtKP20 contract."""

from __future__ import annotations

import numpy as np

from src.utils.schema.court import (
    COURT_KP_IDX,
    COURT_KP_NAMES,
    GROUND_COURT_KP_NAMES,
    NUM_COURT_KP,
    NUM_GROUND_COURT_KP,
    OPPOSITE_COURT_END_INDEX,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

_EXPECTED_NAMES = (
    "far_doubles_left",
    "far_doubles_right",
    "near_doubles_left",
    "near_doubles_right",
    "far_singles_left",
    "near_singles_left",
    "far_singles_right",
    "near_singles_right",
    "far_service_left",
    "far_service_right",
    "near_service_left",
    "near_service_right",
    "far_service_t",
    "near_service_t",
    "net_center",
    "left_post_base",
    "left_post_top",
    "right_post_base",
    "right_post_top",
    "center_strap_top",
)
_EXPECTED_GROUND_POINTS = np.asarray(
    (
        (-5.485, 11.885, 0.0),
        (5.485, 11.885, 0.0),
        (-5.485, -11.885, 0.0),
        (5.485, -11.885, 0.0),
        (-4.115, 11.885, 0.0),
        (-4.115, -11.885, 0.0),
        (4.115, 11.885, 0.0),
        (4.115, -11.885, 0.0),
        (-4.115, 6.4, 0.0),
        (4.115, 6.4, 0.0),
        (-4.115, -6.4, 0.0),
        (4.115, -6.4, 0.0),
        (0.0, 6.4, 0.0),
        (0.0, -6.4, 0.0),
    ),
    dtype=np.float32,
)


def test_ground_aliases_expose_without_redefining_courtkp20() -> None:
    assert NUM_COURT_KP == 20
    assert COURT_KP_NAMES == _EXPECTED_NAMES
    assert {name: index for index, name in enumerate(_EXPECTED_NAMES)} == COURT_KP_IDX
    assert NUM_GROUND_COURT_KP == 14
    assert COURT_KP_NAMES[:14] == GROUND_COURT_KP_NAMES
    np.testing.assert_array_equal(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].numpy(),
        _EXPECTED_GROUND_POINTS,
    )


def test_opposite_end_mapping_is_a_complete_involution() -> None:
    assert len(OPPOSITE_COURT_END_INDEX) == NUM_GROUND_COURT_KP
    assert set(OPPOSITE_COURT_END_INDEX) == set(range(NUM_GROUND_COURT_KP))
    assert tuple(
        OPPOSITE_COURT_END_INDEX[index] for index in OPPOSITE_COURT_END_INDEX
    ) == tuple(range(NUM_GROUND_COURT_KP))

    points = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].numpy()
    swapped = points[np.asarray(OPPOSITE_COURT_END_INDEX)]
    np.testing.assert_allclose(swapped[:, 0], points[:, 0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(swapped[:, 1], -points[:, 1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(swapped[:, 2], points[:, 2], atol=0.0, rtol=0.0)

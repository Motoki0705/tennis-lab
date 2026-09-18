"""The review court wrappers must reuse the shared schema exactly."""

from __future__ import annotations

import numpy as np
import pytest

from src.tasks.base.visualization.review.court import (
    apron_polygon,
    court_edges,
    court_keypoints,
    net_geometry,
)
from src.utils.schema.court import (
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    court_keypoints_3d,
)


def test_standard_keypoints_match_the_shared_schema() -> None:
    expected = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()
    actual = court_keypoints(None)
    assert actual.shape == (20, 3)
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, atol=1e-6)


def test_net_post_offset_shifts_only_the_post_x_coordinates() -> None:
    standard = court_keypoints(None)
    shifted = court_keypoints(0.49)
    # Keypoints 15 and 17 are the post bases, 16/18 the post tops.
    assert shifted[15, 0] == pytest.approx(-(5.485 + 0.49), abs=1e-5)
    assert shifted[17, 0] == pytest.approx(+(5.485 + 0.49), abs=1e-5)
    assert shifted[16, 2] == pytest.approx(1.07, abs=1e-5)
    assert np.allclose(shifted[15], standard[15] + [-0.49 + 0.914, 0, 0], atol=1e-5)


def test_edges_are_the_shared_skeleton() -> None:
    assert court_edges() == tuple((int(a), int(b)) for a, b in COURT_SKELETON)
    assert len(court_edges()) == 17


def test_apron_is_the_run_off_rectangle() -> None:
    apron = apron_polygon()
    assert apron.shape == (4, 2)
    assert tuple(apron[0]) == pytest.approx((X_MIN, Y_MIN))
    assert tuple(apron[1]) == pytest.approx((X_MAX, Y_MIN))
    assert tuple(apron[2]) == pytest.approx((X_MAX, Y_MAX))
    assert tuple(apron[3]) == pytest.approx((X_MIN, Y_MAX))


def test_net_geometry_samples_the_top_cable_and_reuses_the_posts() -> None:
    geometry = net_geometry(0.49)
    top = geometry["top"]
    assert isinstance(top, dict)
    assert len(top["x"]) == len(top["z"]) == 33
    assert top["x"][0] == pytest.approx(-(5.485 + 0.49), abs=1e-5)
    assert top["x"][-1] == pytest.approx(+(5.485 + 0.49), abs=1e-5)
    # The cable sags from the post height (1.07 m) to the centre strap (0.914 m).
    assert top["z"][0] == pytest.approx(1.07, abs=1e-5)
    assert top["z"][16] == pytest.approx(0.914, abs=1e-5)
    posts = np.asarray(geometry["posts"])
    assert posts.shape == (4, 3)
    keypoints = court_keypoints(0.49)
    assert np.allclose(posts, keypoints[[15, 16, 17, 18]], atol=1e-6)

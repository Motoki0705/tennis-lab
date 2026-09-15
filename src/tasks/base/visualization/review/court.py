"""Thin wrappers over the shared court schema for the review UI.

Court geometry, keypoint names, and skeleton connectivity stay owned by
:mod:`src.utils.schema.court`; this module only reshapes them for the browser
payload and never redefines a dimension or an index.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.utils.schema.court import (
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    CourtConfig,
    court_keypoints_3d,
    net_height_at_x,
)

NET_TOP_SAMPLES = 33
NET_POST_INDICES = (15, 16, 17, 18)


def court_keypoints(net_post_offset_x: float | None) -> NDArray[np.float32]:
    """Return ``(20, 3)`` physical court keypoints in metres.

    ``None`` selects :data:`STANDARD_COURT_CONFIG`; PLCS omits ``court_config``
    and therefore always uses the standard net post offset.
    """
    if net_post_offset_x is None:
        config = STANDARD_COURT_CONFIG
    else:
        offset = float(net_post_offset_x)
        if not np.isfinite(offset):
            raise ValueError("net_post_offset_x must be finite.")
        config = CourtConfig(net_post_offset_x=offset, net_post_offset_x_range=None)
    return np.asarray(court_keypoints_3d(config).numpy(), dtype=np.float32)


def court_edges() -> tuple[tuple[int, int], ...]:
    """Return the shared court skeleton as index pairs."""
    return tuple((int(a), int(b)) for a, b in COURT_SKELETON)


def apron_polygon() -> NDArray[np.float32]:
    """Return the run-off apron rectangle on the court plane ``z = 0``."""
    return np.asarray(
        (
            (X_MIN, Y_MIN),
            (X_MAX, Y_MIN),
            (X_MAX, Y_MAX),
            (X_MIN, Y_MAX),
        ),
        dtype=np.float32,
    )


def net_geometry(net_post_offset_x: float | None) -> dict[str, object]:
    """Return the net's top cable samples and the four post base/top points."""
    keypoints = court_keypoints(net_post_offset_x)
    left_top = float(keypoints[16, 0])
    right_top = float(keypoints[18, 0])
    xs = np.linspace(left_top, right_top, NET_TOP_SAMPLES)
    zs = np.asarray([net_height_at_x(float(x)) for x in xs], dtype=np.float32)
    posts = keypoints[list(NET_POST_INDICES)]
    return {
        "top": {"x": [float(x) for x in xs], "z": [float(z) for z in zs]},
        "posts": posts.tolist(),
    }


__all__ = [
    "NET_POST_INDICES",
    "NET_TOP_SAMPLES",
    "apron_polygon",
    "court_edges",
    "court_keypoints",
    "net_geometry",
]

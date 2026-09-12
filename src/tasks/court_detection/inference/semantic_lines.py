"""Camera-invariant probabilities from the trained categorical line head."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.target_schemas import SEMANTIC_LINE_CHANNEL_NAMES

INVARIANT_LINE_SCHEMA = "court_line_type_probabilities_v1"
INVARIANT_LINE_NAMES = (
    "background",
    "baseline",
    "doubles_sideline",
    "singles_sideline",
    "service_line",
    "center_service_line",
    "center_mark",
)
_GROUPS = ((0,), (1, 2), (3, 4), (5, 6), (7, 8), (9,), (10, 11))


def merge_camera_line_probabilities(
    probabilities: NDArray[np.float32],
    *,
    channel_names: Sequence[str],
) -> NDArray[np.float32]:
    """Sum mutually exclusive softmax classes, retaining background and six types.

    Merge before projection; neither argmax nor a maximum preserves probability
    mass. Left/right sidelines also exchange under a reversed camera viewpoint.
    The checkpoint head and its supervision remain unchanged.
    """
    values = np.asarray(probabilities, dtype=np.float32)
    if tuple(channel_names) != SEMANTIC_LINE_CHANNEL_NAMES:
        raise ValueError("Expected the exact camera-view semantic line channel order.")
    if (
        values.ndim != 3
        or values.shape[0] != len(channel_names)
        or min(values.shape[1:]) < 2
    ):
        raise ValueError("Semantic probabilities must have shape (12,H,W), H,W >= 2.")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Semantic probabilities must be finite and in [0,1].")
    if not np.allclose(values.sum(axis=0), 1.0, atol=2e-5, rtol=0):
        raise ValueError("Expected categorical softmax probabilities summing to one.")
    return np.stack([values[list(group)].sum(axis=0) for group in _GROUPS]).astype(
        np.float32
    )

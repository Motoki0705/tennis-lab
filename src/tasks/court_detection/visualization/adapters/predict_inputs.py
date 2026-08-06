"""Model-input adapters for court-detection visualization.

The court predictors accept a single RGB image and run their own short-side
resize + ImageNet normalization internally, so the adapter only standardizes the
loaded frame into the contiguous ``uint8`` RGB array the predictors expect.
"""

from __future__ import annotations

import numpy as np

from src.tasks.court_detection.model_io.contracts import CourtModelIOError
from src.tasks.court_detection.visualization.io.frames import CourtFrame


def to_predictor_input(frame: CourtFrame) -> np.ndarray:
    """Convert a loaded frame into the predictor input array."""
    rgb = frame.rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise CourtModelIOError(
            "Visualization frames must have shape (H, W, 3) and dtype uint8."
        )
    contiguous: np.ndarray = np.ascontiguousarray(rgb)
    return contiguous

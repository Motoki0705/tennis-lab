"""Combine the NHT render and procedural layers into one output frame."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.visualization.reveal.timeline import RevealFrameState


def gaussian_layer(
    state: RevealFrameState,
    nht_rgb: NDArray[np.float32],
    nht_alpha: NDArray[np.float32],
    point_rgb: tuple[float, float, float],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return the premultiplied Gaussian layer for one frame.

    The colour moves from ``point_rgb`` to the decoded NHT colour by
    ``color_progress``. The matte (NHT alpha over the background) moves to the
    raw NHT RGB by ``matte_release``; at the end of the reveal the layer equals
    the unmodified ``nht-render`` RGB with full coverage.
    """
    color = (
        np.asarray(point_rgb, dtype=np.float32) * (1.0 - state.color_progress)
        + nht_rgb * state.color_progress
    )
    release = state.matte_release
    premultiplied = color * nht_alpha * (1.0 - release) + nht_rgb * release
    alpha = nht_alpha * (1.0 - release) + release
    return premultiplied.astype(np.float32), alpha.astype(np.float32)


def over(
    base: NDArray[np.float32],
    layer_rgb: NDArray[np.float32],
    layer_alpha: NDArray[np.float32],
    opacity: float,
) -> NDArray[np.float32]:
    """Composite a premultiplied layer, faded by ``opacity``, over ``base``."""
    if opacity <= 0.0:
        return base
    return layer_rgb * opacity + base * (1.0 - layer_alpha * opacity)


def to_uint8(frame: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Quantize a linear ``[0, 1]`` RGB frame for encoding."""
    return (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

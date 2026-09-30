"""Source-coordinate GMM component ellipses; these are not mixture HDRs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating]
Image: TypeAlias = NDArray[np.uint8]
CYAN = (255, 230, 0)
MAGENTA = (255, 60, 255)
GREEN = (60, 255, 60)
ORANGE = (0, 190, 255)
RED = (60, 60, 255)


@dataclass(frozen=True)
class ComponentEllipse:
    center: tuple[float, float]
    semiaxes: tuple[float, float]
    angle_degrees: float
    weight: float


def component_geometry(
    means: FloatArray, scale_tril: FloatArray, logits: FloatArray, scale_xy: tuple[float, float],
) -> tuple[NDArray[np.float64], tuple[ComponentEllipse, ...]]:
    """Map each full covariance with DΣDᵀ; semiaxes are exactly 2σ.

    ``scale_xy`` maps normalized source UV to the displayed pixels, including
    the store's width-ratio resize. It is not necessarily (display W-1,H-1).
    No uncertainty from the separation of different components is folded into
    their ellipses. Only the point marker uses the weighted mixture mean.
    """
    n = len(means)
    if n < 1 or means.shape != (n, 2) or scale_tril.shape != (n, 2, 2) or logits.shape != (n,):
        raise ValueError('Expected one frame of K full-covariance components')
    if not all(np.isfinite(x).all() for x in (means, scale_tril, logits)):
        raise ValueError('Nonfinite mixture')
    scale = np.asarray(scale_xy, np.float64)
    if scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Pixel scale must be positive and finite')
    if (scale_tril[:, (0, 1), (0, 1)] <= 0).any() or (scale_tril[:, 0, 1] != 0).any():
        raise ValueError('Expected a nonsingular lower Cholesky factor')
    weights = np.exp(logits.astype(np.float64) - logits.max())
    weights /= weights.sum()
    pixel_means = means.astype(np.float64) * scale
    factors = scale_tril.astype(np.float64) * scale[None, :, None]
    covariance = factors @ factors.transpose(0, 2, 1)
    ellipses = []
    for mean, cov, weight in zip(pixel_means, covariance, weights, strict=True):
        values, vectors = np.linalg.eigh(cov)
        if values[0] <= 0:
            raise ValueError('Pixel covariance is not positive definite')
        axes = 2 * np.sqrt(values[::-1])
        direction = vectors[:, -1]
        ellipses.append(ComponentEllipse((float(mean[0]), float(mean[1])), (float(axes[0]), float(axes[1])),
                                         float(np.degrees(np.arctan2(direction[1], direction[0]))), float(weight)))
    return np.asarray((pixel_means * weights[:, None]).sum(0), np.float64), tuple(ellipses)


def mark(image: Image, point: FloatArray, color: tuple[int, int, int], symbol: int) -> None:
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('Cannot draw an undefined position')
    xy = (int(round(float(point[0]))), int(round(float(point[1]))))
    cv2.drawMarker(image, xy, (0, 0, 0), symbol, 16, 4, cv2.LINE_AA)
    cv2.drawMarker(image, xy, color, symbol, 14, 2, cv2.LINE_AA)


def draw_mixture(image: Image, means: FloatArray, factors: FloatArray, logits: FloatArray,
                 scale_xy: tuple[float, float], *,
                 point_summary: Literal["mixture", "top_component"] = "mixture") -> None:
    """Blend component outlines with alpha=weight and draw an explicit summary."""
    mean, ellipses = component_geometry(means, factors, logits, scale_xy)
    if point_summary == "top_component":
        mean = np.asarray(ellipses[int(logits.argmax())].center, np.float64)
    elif point_summary != "mixture":
        raise ValueError("Unknown point summary")
    # Stable component order; blending is visualization only, not a density union.
    for ellipse in ellipses:
        layer = image.copy()
        cv2.ellipse(layer, (ellipse.center, (2 * ellipse.semiaxes[0], 2 * ellipse.semiaxes[1]), ellipse.angle_degrees),
                    CYAN, 2, cv2.LINE_AA)
        cv2.addWeighted(layer, ellipse.weight, image, 1 - ellipse.weight, 0, dst=image)
    mark(image, mean, CYAN, cv2.MARKER_CROSS)


def text_line(image: Image, text: str, x: int, y: int, color: tuple[int, int, int] = (235, 235, 235),
              size: float = .55) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

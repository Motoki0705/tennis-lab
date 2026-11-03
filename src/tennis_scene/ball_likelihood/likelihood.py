"""Per-frame continuous interpolation and bounded-background log likelihood."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline
from scipy.special import expit, logsumexp

from src.utils.geometry.triangulation import PinholeCamera

from .config import DenseLikelihoodConfig

# Exact sigmoid 0/1 has lost information. Symmetric FP32 half-epsilon clipping
# supplies finite +/-16.64 logits; it does not recover the lost peak curvature.
SIGMOID_CLIP = float(np.finfo(np.float32).eps / 2)
_Y, _X = np.mgrid[-1:2, -1:2]
_DESIGN = np.stack((np.ones(9), _X.ravel(), _Y.ravel(), _X.ravel() ** 2,
                    _X.ravel() * _Y.ravel(), _Y.ravel() ** 2), -1)
_QUADRATIC = np.linalg.pinv(_DESIGN)


class HeatmapLikelihood:
    """Own only one frame's log-domain map; source heatmaps stay shared/read-only."""

    def __init__(self, heatmap: NDArray[np.float32], camera: PinholeCamera,
                 source_size_wh: tuple[int, int], config: DenseLikelihoodConfig) -> None:
        if heatmap.ndim != 2 or min(heatmap.shape) < 4 or heatmap.dtype != np.float32:
            raise ValueError("Dense interpolation requires a float32 map at least 4x4")
        if not np.isfinite(heatmap).all() or np.any((heatmap < 0) | (heatmap > 1)):
            raise ValueError("Dense map must be finite in [0,1]")
        self.camera, self.config = camera, config
        self.temperature, self.background = config.for_view(camera.camera_id)
        self.shape = heatmap.shape
        self.scale = (np.array(self.shape[::-1]) - 1) / (np.array(source_size_wh) - 1)
        p = np.clip(heatmap.astype(np.float64), SIGMOID_CLIP, 1 - SIGMOID_CLIP)
        self.log_map = np.log(p) if config.interpolation == "quadratic_log" else np.log(p) - np.log1p(-p)
        self.peak = float(self.log_map.max())
        self.informative = bool(heatmap.max() >= config.min_peak_probability and np.ptp(self.log_map) > 1e-3)
        self.saturated = int(np.count_nonzero(heatmap == 1))
        self.normalizer = float(logsumexp(self.log_map / self.temperature) - np.log(heatmap.size))
        self.spline = (RectBivariateSpline(np.arange(self.shape[0]), np.arange(self.shape[1]), self.log_map)
                       if config.interpolation == "bicubic_logit" else None)

    def interpolate(self, xy: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Native-grid coordinates. Quadratic uses a real 3x3 (one-sided at edge)."""
        x, y = xy[:, 0], xy[:, 1]
        if self.spline is not None:
            value = self.spline.ev(y, x)
            gradient = np.column_stack((self.spline.ev(y, x, dy=1), self.spline.ev(y, x, dx=1)))
            return value, gradient
        cx = np.clip(np.rint(x).astype(int), 1, self.shape[1] - 2)
        cy = np.clip(np.rint(y).astype(int), 1, self.shape[0] - 2)
        samples = self.log_map[cy[:, None] + _Y.ravel(), cx[:, None] + _X.ravel()]
        coef = samples @ _QUADRATIC.T
        dx, dy = x - cx, y - cy
        value = coef[:, 0] + coef[:, 1] * dx + coef[:, 2] * dy + coef[:, 3] * dx ** 2 + coef[:, 4] * dx * dy + coef[:, 5] * dy ** 2
        gradient = np.column_stack((coef[:, 1] + 2 * coef[:, 3] * dx + coef[:, 4] * dy,
                                    coef[:, 2] + coef[:, 4] * dx + 2 * coef[:, 5] * dy))
        return value, gradient

    def evaluate(self, xyz: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Log mixture relative to uniform, gradient per metre, signal support."""
        xyz = np.atleast_2d(xyz)
        count = len(xyz)
        log_likelihood = np.full(count, np.log(self.background))
        gradient: NDArray[np.float64] = np.zeros((count, 3), np.float64)
        support: NDArray[np.bool_] = np.zeros(count, bool)
        if not self.informative:
            return log_likelihood, gradient, support
        uv, front = self.camera.project(xyz)
        xy = uv * self.scale
        inside = front & (xy >= 0).all(-1) & (xy <= np.array(self.shape[::-1]) - 1).all(-1)
        if not inside.any():
            return log_likelihood, gradient, support
        value, deriv = self.interpolate(xy[inside])
        signal = np.log1p(-self.background) + value / self.temperature - self.normalizer
        log_likelihood[inside] = np.logaddexp(np.log(self.background), signal)
        responsibility = expit(signal - np.log(self.background))
        points = xyz[inside]
        projected = points @ self.camera.matrix[:, :3].T + self.camera.matrix[:, 3]
        jac = (self.camera.matrix[None, :2, :3] - uv[inside, :, None] * self.camera.matrix[None, 2, :3]) / projected[:, 2, None, None]
        gradient[inside] = np.einsum("ni,nij->nj", deriv * self.scale, jac) * (responsibility / self.temperature)[:, None]
        support[inside] = value >= self.peak - self.config.support_logit_drop * self.temperature
        return log_likelihood, gradient, support


def objective(maps: tuple[HeatmapLikelihood, ...], xyz: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    samples = [m.evaluate(xyz) for m in maps]
    return (-np.sum([r[0] for r in samples], axis=0),
            -np.sum([r[1] for r in samples], axis=0), np.stack([r[2] for r in samples]))

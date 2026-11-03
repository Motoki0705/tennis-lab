"""Selectable single-peak CPU decoders for native sigmoid heatmaps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import least_squares
from torch import Tensor

from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic

METHODS = ("legacy", "axis_logit", "quadratic_log", "quadratic_logit", "centroid", "softargmax", "gaussian")


class FitStatus(IntEnum):
    LEGACY = 0
    REFINED = 1
    PARTIAL_AXIS = 2
    DEGENERATE = 3
    NON_CONCAVE = 4
    OUTSIDE_LOCAL_CELL = 5
    OPTIMIZER_FAILED = 6
    INSUFFICIENT_PATCH = 7


@dataclass(frozen=True, slots=True)
class HeatmapDecodingConfig:
    method: str
    patch_size: int
    temperature: float
    coordinate_correction: str
    model_stride: int
    scale_xy: tuple[float, float]
    offset_xy: tuple[float, float]

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown heatmap decoding method: {self.method}")
        if type(self.patch_size) is not int or self.patch_size not in (3, 5):
            raise ValueError("patch_size must be 3 or 5")
        if self.method == "gaussian" and self.patch_size != 5:
            raise ValueError("Nonlinear Gaussian fitting requires patch_size=5")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.coordinate_correction not in ("none", "stride", "affine"):
            raise ValueError("Unknown coordinate_correction")
        if type(self.model_stride) is not int or self.model_stride < 1:
            raise ValueError("model_stride must be a positive integer")
        if (len(self.scale_xy) != 2 or len(self.offset_xy) != 2
                or not all(math.isfinite(v) for v in (*self.scale_xy, *self.offset_xy))
                or min(self.scale_xy) <= 0):
            raise ValueError("Affine correction requires finite XY values and positive scales")
        if self.coordinate_correction != "affine" and (self.scale_xy != (1.0, 1.0) or self.offset_xy != (0.0, 0.0)):
            raise ValueError("Non-identity scale/offset require coordinate_correction=affine")

    @property
    def preserves_prediction(self) -> bool:
        return self.method == "legacy" and self.coordinate_correction == "none"


@dataclass(frozen=True, slots=True)
class DecodedHeatmaps:
    coords: Tensor
    confidence: Tensor
    fit_status: Tensor


def correct_coordinates(coords: Tensor, size_hw: tuple[int, int], config: HeatmapDecodingConfig) -> Tensor:
    """Return normalized endpoint coordinates; stride mode is a half-pixel hypothesis."""
    if config.coordinate_correction == "none":
        return coords
    if config.coordinate_correction == "affine":
        corrected = coords * coords.new_tensor(config.scale_xy) + coords.new_tensor(config.offset_xy)
    else:
        h, w = size_hw
        cells = coords.new_tensor([w, h])
        corrected = ((coords * (cells - 1) + 0.5) * config.model_stride - 0.5) / (cells * config.model_stride - 1)
    return corrected.clamp(0, 1)


@lru_cache(maxsize=64)
def _design(x0: int, x1: int, y0: int, y1: int) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    xy = np.stack((xx.ravel(), yy.ravel()), axis=1)
    a = np.column_stack((np.ones(len(xy)), xy, xx.ravel() ** 2, xx.ravel() * yy.ravel(), yy.ravel() ** 2))
    return xy, np.linalg.pinv(a), int(np.linalg.matrix_rank(a))


def _domain(values: NDArray[np.float64], logit: bool) -> NDArray[np.float64]:
    # Float32 sigmoid saturation has already lost information; clipping only makes
    # the fit defined and is not claimed to recover the original logits.
    if logit:
        p = np.clip(values, 1e-7, 1 - 1e-7)
        return np.log(p) - np.log1p(-p)
    return np.log(np.maximum(values, 1e-12))


def _axis_logit(image: NDArray[np.float64], x: int, y: int) -> tuple[NDArray[np.float64], FitStatus]:
    h, w = image.shape
    offset = np.zeros(2)
    count = 0
    for axis, index, limit in ((0, x, w), (1, y, h)):
        if not 0 < index < limit - 1:
            continue
        values = image[y, x - 1:x + 2] if axis == 0 else image[y - 1:y + 2, x]
        minus, centre, plus = _domain(values, True)
        curvature = 2 * centre - minus - plus
        if curvature > 1e-12:
            offset[axis] = np.clip(0.5 * (plus - minus) / curvature, -0.5, 0.5)
            count += 1
    status = FitStatus.REFINED if count == 2 else FitStatus.PARTIAL_AXIS if count else FitStatus.DEGENERATE
    return offset, status


def _gaussian(patch: NDArray[np.float64], xy: NDArray[np.float64]) -> tuple[NDArray[np.float64], FitStatus]:
    """Fit background + amplitude*exp(-0.5*d^T LL^T d), including XY covariance."""
    low, high = float(patch.min()), float(patch.max())
    if high - low < 1e-8:
        return np.zeros(2), FitStatus.DEGENERATE
    values = patch.ravel()

    def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        dx, dy = xy[:, 0] - p[0], xy[:, 1] - p[1]
        exponent = (np.exp(p[2]) * dx + p[3] * dy) ** 2 + (np.exp(p[4]) * dy) ** 2
        return p[5] + p[6] * np.exp(-0.5 * exponent) - values

    fit = least_squares(
        residual, np.array([0., 0., 0., 0., 0., low, high - low]),
        bounds=([-1., -1., -2.3, -8., -2.3, 0., 1e-8], [1., 1., 2.3, 8., 2.3, 1., 2.]),
        max_nfev=100, ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )
    if not fit.success:
        return np.zeros(2), FitStatus.OPTIMIZER_FAILED
    if np.any(np.abs(fit.x[:2]) > 0.999):
        return np.zeros(2), FitStatus.OUTSIDE_LOCAL_CELL
    return fit.x[:2], FitStatus.REFINED


def _local_offset(image: NDArray[np.float64], x: int, y: int, config: HeatmapDecodingConfig) -> tuple[NDArray[np.float64], FitStatus]:
    if config.method == "axis_logit":
        return _axis_logit(image, x, y)
    h, w = image.shape
    radius = config.patch_size // 2
    x0, x1, y0, y1 = max(-radius, -x), min(radius + 1, w - x), max(-radius, -y), min(radius + 1, h - y)
    patch = image[y + y0:y + y1, x + x0:x + x1]
    xy, inverse, rank = _design(x0, x1, y0, y1)
    if float(np.ptp(patch)) < 1e-8:
        return np.zeros(2), FitStatus.DEGENERATE
    if config.method in ("centroid", "softargmax"):
        weights = np.maximum(patch.ravel() - patch.min(), 0)
        if config.method == "softargmax":
            # Spatial softmax of log background-subtracted probability / T.
            positive = weights > 0
            logweights = np.log(weights[positive]) / config.temperature
            weights[positive] = np.exp(logweights - logweights.max())
        return (weights[:, None] * xy).sum(0) / weights.sum(), FitStatus.REFINED
    if rank < 6:
        return np.zeros(2), FitStatus.INSUFFICIENT_PATCH
    if config.method == "gaussian":
        return _gaussian(patch, xy)
    coeff = inverse @ _domain(patch.ravel(), config.method == "quadratic_logit")
    hessian = np.array([[2 * coeff[3], coeff[4]], [coeff[4], 2 * coeff[5]]])
    if np.max(np.linalg.eigvalsh(hessian)) >= -1e-10:
        return np.zeros(2), FitStatus.NON_CONCAVE
    offset = np.linalg.solve(hessian, -coeff[1:3])
    if np.any(np.abs(offset) > 1.0):
        return np.zeros(2), FitStatus.OUTSIDE_LOCAL_CELL
    return offset, FitStatus.REFINED


def decode_heatmaps(heatmaps: Tensor, config: HeatmapDecodingConfig, *, subpixel_refine: bool) -> DecodedHeatmaps:
    """Decode (...,H,W) native probabilities without modifying them or peak scores.

    New fits run on CPU, inspect only a 3x3/5x5 patch and explicitly report all
    rejected fits. The legacy branch delegates to the unchanged shared decoder.
    """
    if heatmaps.ndim < 2 or min(heatmaps.shape[-2:]) < 2 or not heatmaps.is_floating_point():
        raise ValueError("Expected floating native heatmaps with H,W >= 2")
    if not torch.isfinite(heatmaps).all() or bool(((heatmaps < 0) | (heatmaps > 1)).any()):
        raise ValueError("Native sigmoid heatmaps must be finite in [0,1]")
    coords, confidence = heatmaps_to_argmax(heatmaps)
    h, w = heatmaps.shape[-2:]
    status = torch.full_like(confidence, FitStatus.LEGACY, dtype=torch.int64)
    if config.method == "legacy":
        if subpixel_refine:
            coords = refine_peaks_log_parabolic(heatmaps, coords)
    else:
        if not subpixel_refine:
            raise ValueError("Alternative decoding requires subpixel_refine=true")
        if heatmaps.device.type != "cpu":
            raise ValueError("Alternative decoding requires CPU native heatmaps")
        maps = heatmaps.detach().reshape(-1, h, w).numpy()
        flat = coords.reshape(-1, 2).clone()
        for i, image in enumerate(maps):
            x, y = (flat[i] * flat.new_tensor([w - 1, h - 1])).round().to(torch.int64).tolist()
            offset, code = _local_offset(image.astype(np.float64), x, y, config)
            flat[i] = flat.new_tensor([x + offset[0], y + offset[1]]) / flat.new_tensor([w - 1, h - 1])
            status.reshape(-1)[i] = int(code)
        coords = flat.reshape(coords.shape).clamp(0, 1)
    return DecodedHeatmaps(correct_coordinates(coords, (h, w), config), confidence, status)


def decode_native_sequence(heatmaps: NDArray[np.float32], config: HeatmapDecodingConfig, *, subpixel_refine: bool) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """Replay a read-only T,H,W sequence with at most 64 maps copied at once."""
    if heatmaps.ndim != 3 or not len(heatmaps):
        raise ValueError("Expected a nonempty T,H,W native sequence")
    coords, scores, codes = [], [], []
    for start in range(0, len(heatmaps), 64):
        maps = torch.from_numpy(np.array(heatmaps[start:start + 64], copy=True))
        result = decode_heatmaps(maps, config, subpixel_refine=subpixel_refine)
        coords.append(result.coords.numpy())
        scores.append(result.confidence.numpy())
        codes.append(result.fit_status.numpy())
    return np.concatenate(coords).astype(np.float32), np.concatenate(scores).astype(np.float32), np.concatenate(codes)

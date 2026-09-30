"""CPU diagnostics for a maximum-weight component, distinct from full GMM HDR."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def precision_rows(
    means: NDArray[np.floating], scale_tril: NDArray[np.floating],
    logits: NDArray[np.floating], target: NDArray[np.floating],
    detector: NDArray[np.floating], candidates: NDArray[np.floating],
    valid: NDArray[np.bool_], size_wh: tuple[int, int],
) -> dict[str, NDArray[np.float64]]:
    """All rows must have an observed target; invalid candidates never anchor a row.

    ``z_radius`` is Mahalanobis radius in the chosen component, whose nominal
    radial CDF is chi-square(2). It is not calibration of the entire mixture.
    """
    n = len(target)
    if means.ndim != 3 or means.shape[:2] != logits.shape or means.shape[-1] != 2:
        raise ValueError('Invalid component axes')
    if scale_tril.shape != (*means.shape, 2) or target.shape != (n, 2) or detector.shape != (n, 2):
        raise ValueError('Invalid target/covariance axes')
    if candidates.shape != (*valid.shape, 2) or len(candidates) != n or len(means) != n:
        raise ValueError('Invalid candidate axes')
    if min(size_wh) < 2 or not all(np.isfinite(x).all() for x in (means, scale_tril, logits, target, detector, candidates)):
        raise ValueError('Nonfinite diagnostic inputs or invalid source size')
    if (np.diagonal(scale_tril, axis1=-2, axis2=-1) <= 0).any() or (scale_tril[..., 0, 1] != 0).any():
        raise ValueError('Expected positive lower triangular scales')
    factor = np.asarray(size_wh, np.float64) - 1
    top = logits.argmax(-1)
    mean = means[np.arange(n), top].astype(np.float64)
    chol = scale_tril[np.arange(n), top].astype(np.float64) * factor[:, None]
    covariance = chol @ chol.swapaxes(-2, -1)
    error = (target - mean) * factor
    whitened = np.linalg.solve(chol, error[..., None])[..., 0]
    distance = np.linalg.norm((candidates - mean[:, None]) * factor, axis=-1)
    distance[~valid] = np.inf
    nearest = distance.min(-1)
    nearest[~valid.any(-1)] = np.nan
    detector_delta = (mean - detector) * factor
    return {
        'error_px': np.linalg.norm(error, axis=-1),
        'detector_error_px': np.linalg.norm((target - detector) * factor, axis=-1),
        'detector_distance_px': np.linalg.norm(detector_delta, axis=-1),
        'detector_dx_px': detector_delta[:, 0], 'detector_dy_px': detector_delta[:, 1],
        'nearest_candidate_px': nearest,
        'sigma_major_px': np.sqrt(np.linalg.eigvalsh(covariance)[:, -1]),
        'z_radius': np.linalg.norm(whitened, axis=-1),
        'z_x': whitened[:, 0], 'z_y': whitened[:, 1],
    }


def detector_groups(error: NDArray[np.floating]) -> dict[str, NDArray[np.bool_]]:
    if not np.isfinite(error).all() or (error < 0).any():
        raise ValueError('Detector errors must be finite and nonnegative')
    return {'all': np.ones(len(error), np.bool_), 'within8': error <= 8,
            '8to20': (error > 8) & (error <= 20), 'wrong': error > 20}

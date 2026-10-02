"""Weighted Rayleigh/logistic scales, with explicit failure instead of fallback."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.tasks.player_association.association.associate import AssociationConfig
from src.tasks.player_association.calibration.samples import (
    PairWindow,
    hierarchical_weights,
)


class CalibrationRejected(RuntimeError):
    """Insufficient support, unstable fit or no admissible fixed candidate."""


@dataclass(frozen=True)
class Scales:
    sigma_m: float
    slope: float
    center: float
    logistic_loss: float
    iterations: int
    appearance_positive: int
    appearance_negative: int


def fit_scales(pairs: list[PairWindow]) -> Scales:
    weights = hierarchical_weights(pairs)
    positive = np.array([p.positive for p in pairs], bool)
    available = np.array([p.cosine is not None for p in pairs], bool)
    support = [int((available & (positive == label)).sum()) for label in (True, False)]
    if min(support) < 10:
        raise CalibrationRejected(f'Appearance fit requires >=10 positive/negative pair-windows: {support}')
    distances = np.array([p.distance_m for p in pairs])
    sigma = float(np.sqrt(np.sum(weights[positive] * distances[positive] ** 2) / (2 * weights[positive].sum())))
    if not np.isfinite(sigma) or sigma <= 0:
        raise CalibrationRejected('Nonpositive/nonfinite Rayleigh scale')
    x = np.array([p.cosine for p in pairs if p.cosine is not None], np.float64)
    y = positive[available].astype(np.float64)
    w = weights[available].copy()
    for label in (0, 1):
        at = y == label
        w[at] *= .5 / w[at].sum()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = parameters
        logits = a * x + b
        loss = float(np.sum(w * (np.logaddexp(0, logits) - y * logits)) + 1e-4 * a * a / 2)
        residual = w * (expit(logits) - y)
        return loss, np.array([np.sum(residual * x) + 1e-4 * a, residual.sum()])

    result = minimize(objective, np.array([62.7, -62.7 * .847]), method='BFGS',
                      jac=True, tol=1e-9, options={'maxiter': 1000})
    a, b = map(float, result.x)
    if not result.success or not np.isfinite([a, b, result.fun]).all() or a <= 0:
        raise CalibrationRejected(f'Appearance fit failed: {result.message}; slope={a}')
    if not np.isfinite(-b / a):
        raise CalibrationRejected('Nonfinite appearance center')
    return Scales(sigma, a, -b / a, float(result.fun), int(result.nit), *support)


def fitted_config(base: AssociationConfig, scales: Scales, margin: float, runner_up: float) -> AssociationConfig:
    if base.appearance is None:
        raise ValueError('The fixed protocol requires CLIP appearance')
    try:
        return replace(base, geometry=replace(base.geometry, sigma_m=scales.sigma_m),
                       appearance=replace(base.appearance, slope=scales.slope, center=scales.center),
                       min_margin=margin, max_runner_up_ratio=runner_up)
    except ValueError as error:
        raise CalibrationRejected(f'Fitted values violate the association config contract: {error}') from error

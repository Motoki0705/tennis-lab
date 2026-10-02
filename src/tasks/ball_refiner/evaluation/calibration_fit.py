"""Fit one covariance multiplier with clip/condition-balanced conditional NLL."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from torch import Tensor

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D


@dataclass(frozen=True)
class FitBlock:
    group: str
    condition: str
    mahalanobis: NDArray[np.float64]
    log_coefficient: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (not self.group or self.condition not in ("observed", "evidence_gap")
                or self.mahalanobis.ndim != 2 or min(self.mahalanobis.shape) < 1
                or self.log_coefficient.shape != self.mahalanobis.shape
                or not np.isfinite(self.mahalanobis).all()
                or not np.isfinite(self.log_coefficient).all()
                or (self.mahalanobis < 0).any()):
            raise ValueError("Invalid located-position calibration block")


@dataclass(frozen=True)
class ScaleFit:
    covariance_multiplier: float
    nll_uv_before: float
    nll_uv_after: float
    groups: tuple[str, ...]
    frames: int


def density_terms(prediction: BallGMM2D, target: Tensor, *, group: str, condition: str) -> FitBlock:
    """Precompute mixture terms once in float64 for exact scalar NLL evaluation."""
    if (target.shape != (*prediction.presence_logits.shape, 2)
            or target.device != prediction.means.device or not torch.isfinite(target).all()):
        raise ValueError("Calibration targets must be finite and match GMM axes/device")
    means, tril, logits = (x.detach().double() for x in
                          (prediction.means, prediction.scale_tril, prediction.mixture_logits))
    white = torch.linalg.solve_triangular(tril, (target.double().unsqueeze(-2) - means).unsqueeze(-1), upper=False)
    q = white.square().sum((-2, -1))
    coefficient = logits.log_softmax(-1) - math.log(2 * math.pi) - tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return FitBlock(group, condition, q.reshape(-1, q.shape[-1]).cpu().numpy(),
                    coefficient.reshape(-1, q.shape[-1]).cpu().numpy())


def fit_covariance_multiplier(
    blocks: list[FitBlock], *, bounds: tuple[float, float], grid_points: int,
) -> ScaleFit:
    """Equal clip weights, equal observed/gap weights; cameras pooled within each.

    GMM scalar NLL can be nonconvex. Refine every grid-local minimum, retain
    endpoints as candidates, and reject a boundary solution instead of expanding
    the declared family. The finite search is not a proof of a global optimum.
    """
    if (not blocks or not all(math.isfinite(x) and x > 0 for x in bounds)
            or not bounds[0] < bounds[1] or grid_points < 3):
        raise ValueError("Require calibration blocks and explicit positive bounds/grid")
    groups = tuple(sorted({b.group for b in blocks}))
    counts = {(g, c): sum(len(b.mahalanobis) for b in blocks if b.group == g and b.condition == c)
              for g in groups for c in ("observed", "evidence_gap")}
    if any(n == 0 for n in counts.values()):
        raise ValueError("Every fit clip requires both evidence conditions")
    q = np.concatenate([b.mahalanobis for b in blocks])
    coefficient = np.concatenate([b.log_coefficient for b in blocks])
    weights = np.concatenate([np.full(len(b.mahalanobis), 1 / (len(groups) * 2 * counts[b.group, b.condition]))
                              for b in blocks])

    def objective(log_scale: float) -> float:
        return float(weights @ -logsumexp(coefficient - log_scale - .5 * q / math.exp(log_scale), axis=-1))

    grid = np.linspace(math.log(bounds[0]), math.log(bounds[1]), grid_points)
    values = [objective(float(x)) for x in grid]
    candidates = [(values[i], float(grid[i])) for i in (0, len(grid) - 1)]
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            result = minimize_scalar(objective, bounds=(grid[i - 1], grid[i + 1]),
                                     method="bounded", options={"xatol": 1e-10})
            if not result.success or not math.isfinite(result.fun):
                raise ValueError("Covariance scalar optimization failed")
            candidates.append((float(result.fun), float(result.x)))
    nll, log_scale = min(candidates)
    if log_scale in (float(grid[0]), float(grid[-1])):
        raise ValueError("Covariance optimum is at declared boundary; report before changing bounds")
    return ScaleFit(math.exp(log_scale), objective(0), nll, groups, len(q))


def cross_validate_clips(
    blocks: list[FitBlock], *, bounds: tuple[float, float], grid_points: int,
) -> tuple[ScaleFit, dict[str, ScaleFit]]:
    """Hold out every camera/condition of a temporal clip together."""
    groups = sorted({b.group for b in blocks})
    if len(groups) < 3:
        raise ValueError("Clip cross-validation requires at least three groups")
    folds = {held: fit_covariance_multiplier([b for b in blocks if b.group != held],
                                            bounds=bounds, grid_points=grid_points) for held in groups}
    full = fit_covariance_multiplier(blocks, bounds=bounds, grid_points=grid_points)
    return full, folds

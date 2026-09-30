"""Positive-weight adaptive cubature on a tangent-mapped ray chart.

The sum of embedded 5/7-point rule differences is an empirical error estimate,
not a rigorous bound. The map covers the entire chart (no fixed world box).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

from .distributions import FloatArray
from .ray import Moments, RayConfig, RayProposal


@dataclass(frozen=True)
class AdaptiveRayConfig(RayConfig):
    relative_error: float
    max_cells: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.relative_error) or self.relative_error <= 0:
            raise ValueError("Adaptive error target must be positive")
        if type(self.max_cells) is not int or self.max_cells < 8:
            raise ValueError("Adaptive cubature needs >=8 cells")


@lru_cache(maxsize=2)
def _rule(order: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = roots_legendre(order)
    indices = np.indices((order,) * 3).reshape(3, -1).T
    return nodes[indices], weights[indices].prod(-1)


class AdaptiveRayProposal:
    """Cache a chart and refine all ten mass/first/second moment integrals.

    Prior-whitened world moments prevent large world origins from dominating the
    error control. Every component gets its own cell tree, including tiny weights.
    Cell budgets increase independently of the fixed outer convergence tolerances.
    """

    def __init__(self, ray: RayProposal) -> None:
        self.ray = ray
        self.world_chol = np.linalg.cholesky(ray.prior.covariance)
        self.world_inverse = np.linalg.inv(self.world_chol)
        self.log_scale = float(ray.log_target(ray.mode[None])[0][0]) + ray.log_det
        bits = np.indices((2, 2, 2)).reshape(3, -1).T
        self.lower = bits.astype(float) - 1
        self.upper = bits.astype(float)
        self.estimates, self.errors = self.evaluate(self.lower, self.upper)
        self.evaluations = 8 * (7**3 + 5**3)
        self.diagnostic: dict[str, float | int | bool] = {}

    def integrand(self, points: FloatArray) -> FloatArray:
        angles = np.pi / 2 * points
        nodes = np.tan(angles)
        coordinates = self.ray.mode + nodes @ self.ray.proposal_chol.T
        log_value, world = self.ray.log_target(coordinates)
        log_jacobian = (np.log(np.pi / 2) - 2 * np.log(np.cos(angles))).sum(-1)
        mass = np.exp(log_value + self.ray.log_det + log_jacobian - self.log_scale)
        centered = (world - self.ray.prior.mean) @ self.world_inverse.T
        xx = (centered[:, :, None] * centered[:, None, :])[:, np.triu_indices(3)[0], np.triu_indices(3)[1]]
        values = mass[:, None] * np.column_stack((np.ones(len(points)), centered, xx))
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite adaptive ray integrand")
        return values

    def evaluate(self, lower: FloatArray, upper: FloatArray) -> tuple[FloatArray, FloatArray]:
        midpoint, radius = (lower + upper) / 2, (upper - lower) / 2
        values = []
        for order in (5, 7):
            nodes, weights = _rule(order)
            points = midpoint[:, None] + radius[:, None] * nodes
            evaluated = self.integrand(points.reshape(-1, 3)).reshape(len(lower), len(nodes), 10)
            values.append(np.einsum("cnm,n->cm", evaluated, weights) * radius.prod(-1)[:, None])
        return values[1], np.abs(values[1] - values[0])

    def integrate(self, config: AdaptiveRayConfig) -> Moments:
        # Every outer level performs additional quadrature, even if its tighter
        # error target was already met at the previous level.
        minimum_cells = len(self.lower) + 7 if self.diagnostic else len(self.lower)
        while True:
            total = self.estimates.sum(0)
            error = self.errors.sum(0)
            if not np.isfinite(total).all() or total[0] <= 0:
                raise RuntimeError("Adaptive ray quadrature has no finite mass")
            # An absolute moment tolerance scaled by evidence, rather than by
            # nearly-zero signed first/off-diagonal moments.
            relative = float(error.max() / total[0])
            if (relative <= config.relative_error and len(self.lower) >= minimum_cells) or len(self.lower) + 7 > config.max_cells:
                break
            count = min(8, (config.max_cells - len(self.lower)) // 7, len(self.lower))
            selected = np.argsort(self.errors.max(-1))[-count:]
            keep: NDArray[np.bool_] = np.ones(len(self.lower), dtype=bool)
            keep[selected] = False
            lower, upper = self.lower[selected], self.upper[selected]
            midpoint = (lower + upper) / 2
            bits = np.indices((2, 2, 2)).reshape(3, -1).T.astype(bool)
            child_lower = np.where(bits[None], midpoint[:, None], lower[:, None]).reshape(-1, 3)
            child_upper = np.where(bits[None], upper[:, None], midpoint[:, None]).reshape(-1, 3)
            estimate, errors = self.evaluate(child_lower, child_upper)
            self.evaluations += len(child_lower) * (7**3 + 5**3)
            self.lower = np.concatenate((self.lower[keep], child_lower))
            self.upper = np.concatenate((self.upper[keep], child_upper))
            self.estimates = np.concatenate((self.estimates[keep], estimate))
            self.errors = np.concatenate((self.errors[keep], errors))
        mean = total[1:4] / total[0]
        second = np.empty((3, 3))
        ii, jj = np.triu_indices(3)
        second[ii, jj] = second[jj, ii] = total[4:] / total[0]
        covariance = self.world_chol @ (second - np.outer(mean, mean)) @ self.world_chol.T
        covariance = (covariance + covariance.T) / 2
        np.linalg.cholesky(covariance)
        self.diagnostic = {
            "embedded_relative_error": relative, "requested_error": config.relative_error,
            "embedded_converged": relative <= config.relative_error,
            "cells": len(self.lower), "evaluations": self.evaluations,
            "metric_is_local_hessian": self.ray.metric_is_local_hessian,
        }
        return self.ray.prior.mean + self.world_chol @ mean, covariance, float(np.log(total[0]) + self.log_scale)

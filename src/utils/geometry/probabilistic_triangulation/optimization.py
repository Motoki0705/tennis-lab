"""Feasible Gauss-Newton steps for the positive-depth camera half spaces."""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .distributions import FloatArray


class NonregularComponentError(RuntimeError):
    """A regular interior Laplace approximation is unavailable, with a reason."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Triangulation optimization failed: {reason}")


def feasible_start(mean: FloatArray, matrices: FloatArray, minimum_depth: float) -> FloatArray:
    normals, offsets = matrices[:, 2, :3], matrices[:, 2, 3]
    if (normals @ mean + offsets > minimum_depth).all():
        return mean.copy()
    # A convex initialization problem only; never evaluate perspective projection
    # at the infeasible prior. The one-unit margin is not a posterior constraint.
    constraint = LinearConstraint(normals, minimum_depth + 1 - offsets, np.inf)
    fit = minimize(
        lambda x: .5 * float((x - mean) @ (x - mean)), mean,
        jac=lambda x: x - mean, method="SLSQP", constraints=constraint,
        options={"maxiter": 100, "ftol": 1e-12},
    )
    if not fit.success or not (normals @ fit.x + offsets > minimum_depth).all():
        raise NonregularComponentError("no_feasible_initial_point")
    return np.asarray(fit.x, dtype=np.float64)

"""Reproduce the fixed positive-depth failure; no random search or data access."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, least_squares, minimize

from src.utils.geometry.probabilistic_triangulation.solver import project_with_jacobian

ROOT = Path(__file__).parent
case = np.load(ROOT / "behind-mode-case.npz", allow_pickle=False)
matrices, means, covariance = case["P"], case["means"], case["cov"]
prior_mean = np.array([0., 0., 2.])
prior_white = np.diag(1 / np.array([6., 12., 3.]))
white = np.linalg.inv(np.linalg.cholesky(covariance))


def residual(point):
    uv, _, _ = project_with_jacobian(point, matrices)
    return np.concatenate((prior_white @ (point - prior_mean), np.einsum("vij,vj->vi", white, uv - means).ravel()))


def jacobian(point):
    _, jac, _ = project_with_jacobian(point, matrices)
    return np.concatenate((prior_white, (white @ jac).reshape(-1, 3)))


starts = [prior_mean]
for camera, mean in zip(matrices, means, strict=True):
    center = -np.linalg.solve(camera[:, :3], camera[:, 3])
    ray = np.linalg.solve(camera[:, :3], np.r_[mean, 1.])
    scale = np.dot(prior_white @ ray, prior_white @ (prior_mean - center)) / np.linalg.norm(prior_white @ ray) ** 2
    starts.append(center + ray * scale)
starts.append(np.mean(starts[1:], axis=0))
rows = []
for method in ("trf", "lm", "dogbox"):
    for start in starts:
        fit = least_squares(residual, start, jac=jacobian, method=method, max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        rows.append({"method": method, "start": start.tolist(), "success": bool(fit.success), "cost": float(fit.cost), "point": fit.x.tolist(), "depth": project_with_jacobian(fit.x, matrices)[2].tolist()})
fit = minimize(
    lambda point: 0.5 * float(residual(point) @ residual(point)),
    prior_mean, jac=lambda point: jacobian(point).T @ residual(point),
    method="SLSQP", constraints=LinearConstraint(matrices[:, 2, :3], 1e-4 - matrices[:, 2, 3], np.inf),
    options={"maxiter": 100, "ftol": 1e-10},
)
rows.append({"method": "SLSQP_positive_depth", "start": prior_mean.tolist(), "success": bool(fit.success), "message": fit.message, "iterations": int(fit.nit), "cost": float(fit.fun), "point": fit.x.tolist(), "depth": project_with_jacobian(fit.x, matrices)[2].tolist()})
(ROOT / "mode-diagnostic.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
print(json.dumps({"cases": len(rows), "positive_converged": sum(row["success"] and min(row["depth"]) > 0 for row in rows)}))

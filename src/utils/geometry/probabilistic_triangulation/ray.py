"""Positive-depth ray quadrature, with analytic single-view radial moments.

Quadrature order differences are empirical error estimates, not rigorous bounds.
Multi-view proposals use a local mode: common missed modes remain a limitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, ndtr, roots_hermitenorm

from src.utils.geometry.triangulation import PinholeCamera

from .distributions import FloatArray, GaussianPrior3D

Moments = tuple[FloatArray, FloatArray, float]


@dataclass(frozen=True)
class RayConfig:
    order: int

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order < 4:
            raise ValueError("Ray quadrature order must be an integer >= 4")


@lru_cache(maxsize=12)
def hermite_rule(order: int, dimensions: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = roots_hermitenorm(order)
    indices = np.indices((order,) * dimensions).reshape(dimensions, -1).T
    return nodes[indices], np.log(weights[indices] / np.sqrt(2 * np.pi)).sum(-1)


def _moments(points: FloatArray, log_mass: FloatArray) -> Moments:
    log_z = float(logsumexp(log_mass))
    if not np.isfinite(log_z):
        raise RuntimeError("Ray quadrature has no finite mass")
    weights = np.exp(log_mass - log_z)
    mean = weights @ points
    delta = points - mean
    covariance = np.einsum("n,ni,nj->ij", weights, delta, delta)
    return mean, covariance, log_z


def single_view_moments(
    camera: PinholeCamera, mean_px: FloatArray, covariance_px: FloatArray,
    prior: GaussianPrior3D, order: int,
) -> Moments:
    """Integrate d^2/d^3/d^4 times the Gaussian prior for d>0 analytically.

    x=C+d*r(u,v), dx=d^2 |det K|^-1 du dv dd. The angular rule is
    centred/scaled by the *entire* selected 2D Gaussian, not a spatial grid.
    No finite world box or radial quantile discretization is used.
    """
    nodes, log_weights = hermite_rule(order, 2)
    uv = mean_px + nodes @ np.linalg.cholesky(covariance_px).T
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(camera.intrinsic).T @ camera.rotation
    offset = camera.center - prior.mean
    precision = np.linalg.inv(prior.covariance)
    a = np.einsum("ni,ij,nj->n", rays, precision, rays)
    b = rays @ precision @ offset
    mu, variance = -b / a, 1 / a
    i0 = np.sqrt(2 * np.pi * variance) * ndtr(mu / np.sqrt(variance))
    i1 = mu * i0 + variance * np.exp(-mu**2 / (2 * variance))
    i2 = mu * i1 + variance * i0
    i3 = mu * i2 + 2 * variance * i1
    i4 = mu * i3 + 3 * variance * i2
    # The recurrence loses precision for a far-behind prior. Reject explicitly;
    # never clip radial variances or substitute a spatial prior.
    if (mu / np.sqrt(variance) < -5).any() or (i2 <= 0).any():
        raise RuntimeError("Single-view radial recurrence outside stable domain")
    log_mass = log_weights - .5 * (offset @ precision @ offset - b * b / a) + np.log(i2)
    log_mass -= .5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(prior.covariance)[1]) + np.linalg.slogdet(camera.intrinsic)[1]
    depth = i3 / i2
    radial_variance = i4 / i2 - depth * depth
    if (radial_variance <= 0).any():
        raise RuntimeError("Nonpositive analytic radial variance")
    points = camera.center + depth[:, None] * rays
    mean, covariance, log_z = _moments(points, log_mass)
    covariance += np.einsum("n,ni,nj->ij", np.exp(log_mass - log_z) * radial_variance, rays, rays)
    np.linalg.cholesky(covariance)
    return mean, covariance, log_z


class RayProposal:
    """Gaussian proposal in whitened image coordinates and transformed depth.

    Every depth is in the intersection of the active camera half-spaces. Infinite
    intervals use log(depth-lower), bounded ones a logit. The full Jacobian is
    included in the target. Mode/metric only place quadrature nodes; the returned
    evidence and moments integrate the nonlinear target, not its Laplace fit.
    """

    def __init__(
        self, cameras: tuple[PinholeCamera, ...], means: FloatArray,
        covariance: FloatArray, prior: GaussianPrior3D, *, anchor_index: int | None = None, _fit_only: bool = False,
    ) -> None:
        if anchor_index is None:
            # All active cameras supply a deterministic optimization start. The
            # best physical target value locates the pilot mode; its nearest
            # camera defines the chart. No observed component is removed.
            pilots = [RayProposal(cameras, means, covariance, prior, anchor_index=i, _fit_only=True) for i in range(len(cameras))]
            pilot = min(pilots, key=lambda p: p.physical_cost(p.log_target(p.mode[None])[1][0]))
            point = pilot.log_target(pilot.mode[None])[1][0]
            nearest = int(np.argmin([np.linalg.norm(point - c.center) for c in cameras]))
            self.__dict__.update(pilots[nearest].__dict__)
            self.set_metric()
            return
        anchor = anchor_index
        indices = [anchor] + [i for i in range(len(cameras)) if i != anchor]
        self.cameras = tuple(cameras[i] for i in indices)
        self.means, self.covariance, self.prior = means[indices], covariance[indices], prior
        camera = self.cameras[0]
        self.center = camera.center
        self.ray_matrix = np.linalg.inv(camera.intrinsic).T @ camera.rotation
        self.chol = np.linalg.cholesky(self.covariance[0])
        self.precision = np.linalg.inv(self.covariance)
        self.prior_precision = np.linalg.inv(prior.covariance)
        self.normalizer = .5 * ((3 + 2 * len(cameras)) * np.log(2 * np.pi) + np.linalg.slogdet(prior.covariance)[1] + np.linalg.slogdet(self.covariance)[1].sum())
        self.rotations = np.stack([c.rotation for c in self.cameras])
        self.translations = np.stack([c.translation for c in self.cameras])
        self.matrices = np.stack([c.matrix for c in self.cameras])
        ray = np.r_[self.means[0], 1.] @ self.ray_matrix
        a = ray @ self.prior_precision @ ray
        b = ray @ self.prior_precision @ (self.center - prior.mean)
        depth = (-b + np.sqrt(b * b + 12 * a)) / (2 * a)
        lower, upper = self.bounds(ray[None])
        if upper[0] <= lower[0]:
            raise RuntimeError("Anchor mean ray has no positive-depth interval")
        if np.isfinite(upper[0]):
            depth = np.clip(depth, lower[0] + .01 * (upper[0] - lower[0]), upper[0] - .01 * (upper[0] - lower[0]))
            value = np.log((depth - lower[0]) / (upper[0] - depth))
        else:
            depth = max(depth, lower[0] + 1.)
            value = np.log(depth - lower[0])
        fit = minimize(self.cost_gradient, np.array([0., 0., value]), method="BFGS", jac=True, options={"gtol": 1e-5, "maxiter": 150})
        # A stationary point is not required for a change of integration variable.
        # A finite centre and strictly positive full Hessian are required; no jitter.
        self.mode: FloatArray = np.asarray(fit.x, np.float64)
        if not np.isfinite(self.cost(self.mode)):
            raise RuntimeError("Nonfinite ray quadrature centre")
        if not _fit_only:
            self.set_metric()

    def physical_cost(self, point: FloatArray) -> float:
        q = self.matrices[:, :, :3] @ point + self.matrices[:, :, 3]
        if (q[:, 2] <= 1e-12).any():
            return float("inf")
        residual = q[:, :2] / q[:, 2, None] - self.means
        delta = point - self.prior.mean
        return .5 * float(delta @ self.prior_precision @ delta + np.einsum("vi,vij,vj->", residual, self.precision, residual))

    def cost_gradient(self, coordinates: FloatArray) -> tuple[float, FloatArray]:
        uv = self.means[0] + self.chol @ coordinates[:2]
        ray = np.r_[uv, 1.] @ self.ray_matrix
        ray_jacobian = self.ray_matrix[:2].T @ self.chol
        offsets = self.rotations[:, 2, :] @ self.center + self.translations[:, 2]
        slopes = self.rotations[:, 2, :] @ ray
        with np.errstate(divide="ignore", invalid="ignore"):
            crossings = -offsets / slopes
        low_values = np.where(slopes > 0, crossings, -np.inf)
        high_values = np.where(slopes < 0, crossings, np.inf)
        li, hi = int(np.argmax(low_values)), int(np.argmin(high_values))
        lower, upper = max(0., low_values[li]), high_values[hi]
        if upper <= lower:
            return float("inf"), np.zeros(3)
        lower_gradient = offsets[li] / slopes[li]**2 * (self.rotations[li, 2] @ ray_jacobian) if lower > 0 else np.zeros(2)
        if np.isfinite(upper):
            upper_gradient = offsets[hi] / slopes[hi]**2 * (self.rotations[hi, 2] @ ray_jacobian)
            probability = float(expit(coordinates[2]))
            depth = lower + (upper - lower) * probability
            depth_gradient = np.r_[(1 - probability) * lower_gradient + probability * upper_gradient, (upper - lower) * probability * (1 - probability)]
            log_jacobian_gradient = np.r_[(upper_gradient - lower_gradient) / (upper - lower), 1 - 2 * probability]
        else:
            increment = float(np.exp(coordinates[2]))
            depth = lower + increment
            depth_gradient = np.r_[lower_gradient, increment]
            log_jacobian_gradient = np.array([0., 0., 1.])
        point = self.center + depth * ray
        point_jacobian = np.outer(ray, depth_gradient)
        point_jacobian[:, :2] += depth * ray_jacobian
        q = self.matrices[:, :, :3] @ point + self.matrices[:, :, 3]
        if (q[:, 2] <= 0).any():
            return float("inf"), np.zeros(3)
        pixel = q[:, :2] / q[:, 2, None]
        pixel_jacobian = (self.matrices[:, :2, :3] - pixel[:, :, None] * self.matrices[:, 2, None, :3]) / q[:, 2, None, None]
        gradient_world = self.prior_precision @ (point - self.prior.mean) + np.einsum("vij,vjk,vk->i", pixel_jacobian.transpose(0, 2, 1), self.precision, pixel - self.means)
        gradient = point_jacobian.T @ gradient_world - 2 * depth_gradient / depth - log_jacobian_gradient
        return self.cost(coordinates), gradient

    def set_metric(self) -> None:
        # Differentiate the analytic gradient, avoiding subtraction of enormous
        # nearly equal log densities for incompatible products.
        epsilon = 1e-4
        axes = np.eye(3) * epsilon
        hessian = np.column_stack([(self.cost_gradient(self.mode + axis)[1] - self.cost_gradient(self.mode - axis)[1]) / (2 * epsilon) for axis in axes])
        hessian = .5 * (hessian + hessian.T)
        if not np.isfinite(hessian).all():
            raise RuntimeError("Nonfinite ray proposal Hessian")
        self.proposal_chol = np.linalg.cholesky(np.linalg.inv(hessian))
        self.log_det = float(np.linalg.slogdet(self.proposal_chol)[1])

    def bounds(self, rays: FloatArray) -> tuple[FloatArray, FloatArray]:
        offsets = self.rotations[:, 2, :] @ self.center + self.translations[:, 2]
        slopes = rays @ self.rotations[:, 2, :].T
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = -offsets / slopes
        lower = np.maximum(0, np.max(np.where(slopes > 0, crossing, -np.inf), axis=-1))
        upper = np.min(np.where(slopes < 0, crossing, np.inf), axis=-1)
        impossible = ((slopes == 0) & (offsets <= 0)).any(-1)
        return lower, np.where(impossible, -np.inf, upper)

    def cost(self, coordinates: FloatArray) -> float:
        return -float(self.log_target(coordinates[None])[0][0])

    def log_target(self, coordinates: FloatArray) -> tuple[FloatArray, FloatArray]:
        uv = self.means[0] + coordinates[:, :2] @ self.chol.T
        rays = np.column_stack((uv, np.ones(len(uv)))) @ self.ray_matrix
        lower, upper = self.bounds(rays)
        bounded = np.isfinite(upper)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            depth = lower + np.exp(coordinates[:, 2])
            log_jacobian = coordinates[:, 2].copy()
            probability = expit(coordinates[bounded, 2])
            depth[bounded] = lower[bounded] + (upper[bounded] - lower[bounded]) * probability
            log_jacobian[bounded] = np.log(upper[bounded] - lower[bounded]) - np.logaddexp(0, coordinates[bounded, 2]) - np.logaddexp(0, -coordinates[bounded, 2])
            points = self.center + depth[:, None] * rays
            difference = points - self.prior.mean
            energy = np.einsum("ni,ij,nj->n", difference, self.prior_precision, difference)
            projected = np.einsum("vij,nj->nvi", self.matrices[:, :, :3], points) + self.matrices[:, :, 3][None]
            residual = projected[:, :, :2] / projected[:, :, 2, None] - self.means
            energy += np.einsum("nvi,vij,nvj->n", residual, self.precision, residual)
            value = -.5 * energy - self.normalizer + 2 * np.log(depth) + log_jacobian + np.linalg.slogdet(self.chol)[1] + np.linalg.slogdet(self.ray_matrix)[1]
        valid = (upper > lower) & (projected[:, :, 2] > 0).all(-1) & np.isfinite(value)
        # Invalid quadrature nodes have zero target mass, never a fake observation.
        return np.where(valid, value, -np.inf), np.where(valid[:, None], points, self.center)

    def integrate(self, order: int) -> Moments:
        nodes, log_weights = hermite_rule(order, 3)
        coordinates = self.mode + nodes @ self.proposal_chol.T
        value, points = self.log_target(coordinates)
        log_mass = log_weights + value + .5 * (nodes * nodes).sum(-1) + 1.5 * np.log(2 * np.pi) + self.log_det
        result = _moments(points, log_mass)
        np.linalg.cholesky(result[1])
        return result

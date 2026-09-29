"""Positive-domain volume moments in a camera ray chart, without a spatial box."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr, logsumexp
from scipy.stats import truncnorm

from src.utils.geometry.triangulation import PinholeCamera

from src.utils.geometry.probabilistic_triangulation.distributions import FloatArray, GaussianPrior3D


@dataclass(frozen=True)
class RayVolumeConfig:
    angular_nodes: int
    depth_nodes: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 4 for n in (self.angular_nodes, self.depth_nodes)):
            raise ValueError("Ray quadrature needs integer orders >=4")


@lru_cache(maxsize=16)
def _nodes(angular: int, depth: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    h, hw = np.polynomial.hermite_e.hermegauss(angular)
    uv = np.stack(np.meshgrid(h, h, indexing="ij"), axis=-1).reshape(-1, 2)
    log_uv_mass = np.log((hw[:, None] * hw[None, :] / (2 * np.pi)).ravel())
    d, dw = np.polynomial.legendre.leggauss(depth)
    return uv, log_uv_mass, (d + 1) / 2, np.log(dw / 2)


def _log_interval(a: FloatArray, b: FloatArray) -> FloatArray:
    # Subtract survival probabilities in the positive tail to avoid 1 - 1.
    left = np.where(a >= 0, log_ndtr(-b), log_ndtr(a))
    right = np.where(a >= 0, log_ndtr(-a), log_ndtr(b))
    return right + np.log(-np.expm1(left - right))


def integrate_ray_component(
    cameras: tuple[PinholeCamera, ...], means: FloatArray, covariance: FloatArray,
    prior: GaussianPrior3D, config: RayVolumeConfig,
) -> tuple[FloatArray, FloatArray, float]:
    """Integrate one product; the camera-plane endpoint has zero volume.

    For the reference camera, X=C+d*r(u,v), d>0 and
    dX=d²/|det(K)| du dv dd. Integrate its Gaussian pixels by Gauss-Hermite.
    Along each ray, the Gaussian prior is a 1D Gaussian times an analytic
    line-evidence factor. All active camera half spaces give an exact depth
    interval; Gauss-Legendre integrates its truncated-normal quantile.
    Other camera likelihoods and the d² Jacobian remain in the integrand.
    """
    if not cameras:
        return prior.mean.copy(), prior.covariance.copy(), 0.
    if any(not np.allclose(c.intrinsic[2], [0., 0., 1.], atol=1e-12, rtol=0) for c in cameras):
        raise ValueError("Ray integration requires canonical pinhole intrinsics")
    # Choose the narrowest angular likelihood, independent of image resize/order.
    reference = min(
        range(len(cameras)),
        key=lambda i: (float(np.linalg.slogdet(covariance[i])[1] - 2 * np.log(abs(np.linalg.det(cameras[i].intrinsic)))), cameras[i].camera_id),
    )
    camera = cameras[reference]
    normals, log_uv_mass, quantiles, log_depth_mass = _nodes(config.angular_nodes, config.depth_nodes)
    pixels = means[reference] + normals @ np.linalg.cholesky(covariance[reference]).T
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(camera.intrinsic).T @ camera.rotation
    center = camera.center
    precision = np.linalg.inv(prior.covariance)
    delta = center - prior.mean
    a = np.einsum("ni,ij,nj->n", rays, precision, rays)
    b = rays @ precision @ delta
    mean_depth, sigma_depth = -b / a, 1 / np.sqrt(a)
    lower, upper = np.zeros(len(rays)), np.full(len(rays), np.inf)
    valid: NDArray[np.bool_] = np.ones(len(rays), dtype=bool)
    for view in cameras:
        slope = rays @ view.rotation[2]
        offset = float(view.rotation[2] @ center + view.translation[2]) - 1e-6
        positive, negative = slope > 0, slope < 0
        lower[positive] = np.maximum(lower[positive], -offset / slope[positive])
        upper[negative] = np.minimum(upper[negative], -offset / slope[negative])
        valid &= (slope != 0) | (offset > 0)
    valid &= lower < upper
    if not valid.any():
        raise RuntimeError("Ray integral contains no finite likelihood mass")
    # Exclude empty ray-domain intervals, never mixture components.
    rays, mean_depth, sigma_depth = rays[valid], mean_depth[valid], sigma_depth[valid]
    alpha = (lower[valid] - mean_depth) / sigma_depth
    beta = (upper[valid] - mean_depth) / sigma_depth
    interval = _log_interval(alpha, beta)
    depth = truncnorm.ppf(quantiles[None], alpha[:, None], beta[:, None], loc=mean_depth[:, None], scale=sigma_depth[:, None])
    points = center + rays[:, None] * depth[..., None]
    log_line = (
        -.5 * (float(delta @ precision @ delta) - b[valid] ** 2 / a[valid])
        - math.log(2 * math.pi) - .5 * float(np.linalg.slogdet(prior.covariance)[1])
        - .5 * np.log(a[valid]) + interval
    )
    log_mass = (
        log_uv_mass[valid, None] + log_depth_mass[None] + log_line[:, None]
        + 2 * np.log(depth) - np.log(abs(np.linalg.det(camera.intrinsic)))
    )
    flat = points.reshape(-1, 3)
    for i, view in enumerate(cameras):
        if i == reference:
            continue
        projected, front = view.project(flat)
        if not front.all():
            raise RuntimeError("Ray quadrature left the positive-depth interval")
        error = projected - means[i]
        quadratic = np.einsum("ni,ij,nj->n", error, np.linalg.inv(covariance[i]), error)
        log_mass -= .5 * (quadratic.reshape(depth.shape) + np.linalg.slogdet(covariance[i])[1] + 2 * np.log(2 * np.pi))
    if not np.isfinite(points).all() or not np.isfinite(log_mass).all():
        raise RuntimeError("Nonfinite ray quadrature")
    log_evidence = float(logsumexp(log_mass))
    weights = np.exp(log_mass.ravel() - log_evidence)
    mean = weights @ flat
    centered = flat - mean
    covariance3d = np.einsum("n,ni,nj->ij", weights, centered, centered)
    np.linalg.cholesky(covariance3d)
    return mean, covariance3d, log_evidence

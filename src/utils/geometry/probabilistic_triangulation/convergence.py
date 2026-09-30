"""Budgeted quadrature checks; reaching the cap returns a flagged full mixture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.triangulation import PinholeCamera

from .adaptive import AdaptiveRayConfig, AdaptiveRayProposal
from .distributions import CameraGMM, FloatArray, GaussianPrior3D
from .ray import RayConfig, RayProposal
from .solver import (
    ComponentCache,
    LaplaceConfig,
    ProbabilisticTriangulation,
    _triangulate,
)
from .volume import VoxelConfig


@dataclass(frozen=True)
class ConvergenceConfig:
    initial_cells: tuple[int, ...]
    levels: tuple[int, ...]
    refine_cells: tuple[int, ...]
    prior_sigmas: float
    nll_tolerance_nat: float
    log_evidence_tolerance_nat: float
    mean_tolerance: float  # caller's world units (metres for ball refiner)
    covariance_relative_tolerance: float

    def __post_init__(self) -> None:
        lengths = {len(self.initial_cells), len(self.levels), len(self.refine_cells)}
        if len(lengths) != 1 or len(self.initial_cells) < 2:
            raise ValueError("Need at least two equally sized refinement budgets")
        for values in (self.initial_cells, self.levels, self.refine_cells):
            if any(type(v) is not int or v < 1 for v in values) or any(b <= a for a, b in zip(values[:-1], values[1:], strict=True)):
                raise ValueError("Refinement budgets must strictly increase")
        for value in (self.nll_tolerance_nat, self.log_evidence_tolerance_nat, self.mean_tolerance, self.covariance_relative_tolerance):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Convergence tolerances must be positive and finite")
        for i in range(len(self.initial_cells)):
            self.budget(i)

    def budget(self, index: int) -> VoxelConfig:
        return VoxelConfig(self.initial_cells[index], self.levels[index], self.refine_cells[index], self.prior_sigmas)


    @property
    def round_limit(self) -> int:
        return len(self.initial_cells)

    def budget_record(self, index: int) -> dict[str, int]:
        return {"initial_cells": self.initial_cells[index], "levels": self.levels[index], "refine_cells": self.refine_cells[index]}


@dataclass(frozen=True)
class RayConvergenceConfig:
    orders: tuple[int, ...]
    nll_tolerance_nat: float
    log_evidence_tolerance_nat: float
    mean_tolerance: float
    covariance_relative_tolerance: float

    def __post_init__(self) -> None:
        if len(self.orders) < 3 or any(b <= a for a, b in zip(self.orders[:-1], self.orders[1:], strict=True)):
            raise ValueError("Ray quadrature needs >=3 strictly increasing orders")
        for order in self.orders:
            RayConfig(order)
        for value in (self.nll_tolerance_nat, self.log_evidence_tolerance_nat, self.mean_tolerance, self.covariance_relative_tolerance):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Convergence tolerances must be positive and finite")

    @property
    def round_limit(self) -> int:
        return len(self.orders)

    def budget(self, index: int) -> RayConfig:
        return RayConfig(self.orders[index])

    def budget_record(self, index: int) -> dict[str, int]:
        return {"quadrature_order": self.orders[index]}


@dataclass(frozen=True)
class AdaptiveRayConvergenceConfig(RayConvergenceConfig):
    relative_errors: tuple[float, ...]
    max_cells: tuple[int, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.relative_errors) != len(self.orders) or len(self.max_cells) != len(self.orders):
            raise ValueError("Adaptive budgets must match orders")
        if any(b >= a for a, b in zip(self.relative_errors[:-1], self.relative_errors[1:], strict=True)) or any(b <= a for a, b in zip(self.max_cells[:-1], self.max_cells[1:], strict=True)):
            raise ValueError("Adaptive error must decrease and cell budget increase")
        for i in range(self.round_limit):
            self.budget(i)

    def budget(self, index: int) -> AdaptiveRayConfig:
        return AdaptiveRayConfig(self.orders[index], self.relative_errors[index], self.max_cells[index])

    def budget_record(self, index: int) -> dict[str, int]:
        return {"quadrature_order": self.orders[index], "max_cells": self.max_cells[index]}


def convergence_config(values: dict[str, Any]) -> ConvergenceConfig | RayConvergenceConfig:
    settings = dict(values)
    method = settings.pop("method", "voxel")  # Historical schema explicitly defines voxel.
    if method == "ray":
        return RayConvergenceConfig(**settings)
    if method == "adaptive_ray":
        return AdaptiveRayConvergenceConfig(**settings)
    if method == "voxel":
        return ConvergenceConfig(**settings)
    raise ValueError(f"Unknown integration method: {method}")


@dataclass(frozen=True)
class CheckedTriangulation:
    posterior: ProbabilisticTriangulation
    converged: bool
    rounds: int
    # Per component, INCLUDING zero-weight components. No probability threshold.
    component_converged: NDArray[np.bool_]
    component_changes: FloatArray  # M,3: log evidence, mean norm, relative covariance
    nll_delta_nat: float
    history: tuple[dict[str, float | int | bool], ...]


def triangulate_converged(
    observations: CameraGMM, cameras: tuple[PinholeCamera, ...], *,
    prior: GaussianPrior3D, laplace: LaplaceConfig, config: ConvergenceConfig | RayConvergenceConfig,
) -> CheckedTriangulation:
    """Check independent, increasingly fine grids, always retain the last one.

    NLL probes include the first, previous and current component means and their
    +/- one marginal standard-deviation axis offsets. No ground truth is used.
    Evidence and moments must also stabilize for EVERY volume component.
    These finite-grid checks do not bound box truncation or Laplace model error.
    """
    cache: ComponentCache = {}
    ray_cache: dict[tuple[int, ...], RayProposal | AdaptiveRayProposal] = {}
    previous: ProbabilisticTriangulation | None = None
    initial_probes = np.empty((0, 3))
    history: list[dict[str, float | int | bool]] = []
    for index in range(config.round_limit):
        current = _triangulate(observations, cameras, prior=prior, config=laplace, volume=config.budget(index), regular_cache=cache, ray_cache=ray_cache)
        if previous is None:
            initial_probes = _probes(current)
        else:
            a, b = previous.distribution, current.distribution
            changes = np.stack((
                np.abs(current.component_log_evidence - previous.component_log_evidence),
                np.linalg.norm(b.means - a.means, axis=-1),
                np.linalg.norm(b.covariance - a.covariance, axis=(-2, -1)) / np.maximum(np.linalg.norm(a.covariance, axis=(-2, -1)), np.linalg.norm(b.covariance, axis=(-2, -1))),
            ), axis=-1)
            passed = (changes <= np.asarray([config.log_evidence_tolerance_nat, config.mean_tolerance, config.covariance_relative_tolerance])).all(-1)
            probes = np.concatenate((initial_probes, _probes(previous), _probes(current)))
            nll_delta = float(np.max(np.abs(b.log_prob(probes) - a.log_prob(probes))))
            converged = bool(passed.all() and nll_delta <= config.nll_tolerance_nat and (not isinstance(config, RayConvergenceConfig) or index >= 2))
            history.append({
                "round": index + 1, **config.budget_record(index),
                "nll_delta_nat": nll_delta,
                "max_log_evidence_delta_nat": float(changes[:, 0].max()),
                "max_mean_delta": float(changes[:, 1].max()),
                "max_covariance_relative_delta": float(changes[:, 2].max()),
                "nonconverged_components": int((~passed).sum()), "converged": converged,
            })
            if converged or index == config.round_limit - 1:
                return CheckedTriangulation(current, converged, index + 1, passed, changes, nll_delta, tuple(history))
        previous = current
    raise AssertionError("At least two refinement rounds are required")


def _probes(posterior: ProbabilisticTriangulation) -> FloatArray:
    distribution = posterior.distribution
    sigma = np.sqrt(distribution.covariance.diagonal(axis1=-2, axis2=-1))
    offsets = np.concatenate((np.zeros((1, 3)), np.eye(3), -np.eye(3)))
    return (distribution.means[:, None] + sigma[:, None] * offsets).reshape(-1, 3)

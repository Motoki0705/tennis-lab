"""Deterministic metric Gaussian geometry for procedural tennis-ball assets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_MIN_GAUSSIAN_COUNT = 32


@dataclass(frozen=True)
class ProceduralBallGeometry:
    """Ball-centred, metre-valued Gaussian source geometry."""

    means: NDArray[np.float32]
    quats: NDArray[np.float32]
    log_scales: NDArray[np.float32]
    opacity_logits: NDArray[np.float32]
    nominal_diameter_m: float

    def __post_init__(self) -> None:
        count = int(self.means.shape[0]) if self.means.ndim == 2 else -1
        expected_shapes = {
            "means": (count, 3),
            "quats": (count, 4),
            "log_scales": (count, 3),
            "opacity_logits": (count,),
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(self, name)
            if value.dtype != np.float32 or value.shape != expected_shape:
                raise ValueError(
                    f"{name} must be float32 with shape {expected_shape}, "
                    f"got dtype={value.dtype}, shape={value.shape}."
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
            value.setflags(write=False)
        if count < _MIN_GAUSSIAN_COUNT or count % 2 != 0:
            raise ValueError(
                f"Procedural geometry requires an even count >= {_MIN_GAUSSIAN_COUNT}."
            )
        if not math.isfinite(self.nominal_diameter_m) or not (
            0.05 <= self.nominal_diameter_m <= 0.09
        ):
            raise ValueError("nominal_diameter_m must lie in [0.05, 0.09].")
        quaternion_norms = np.linalg.norm(self.quats, axis=1)
        if not np.allclose(quaternion_norms, 1.0, atol=1.0e-6, rtol=0.0):
            raise ValueError("Procedural quaternions must be normalized.")

    @property
    def gaussian_count(self) -> int:
        """Return the number of Gaussian primitives."""
        return int(self.means.shape[0])

    def three_sigma_radii_m(self) -> NDArray[np.float32]:
        """Return centre radius plus the largest three-sigma extent."""
        standard_deviations = np.exp(self.log_scales)
        radii = np.linalg.norm(self.means, axis=1) + 3.0 * standard_deviations.max(
            axis=1
        )
        result = np.asarray(radii, dtype=np.float32)
        result.setflags(write=False)
        return result

    def metric_summary(self) -> dict[str, object]:
        """Return geometry evidence used by procedural-asset manifests."""
        radii = self.three_sigma_radii_m()
        lower = self.means.min(axis=0)
        upper = self.means.max(axis=0)
        return {
            "gaussian_count": self.gaussian_count,
            "nominal_diameter_m": self.nominal_diameter_m,
            "p99_three_sigma_radius_m": float(np.quantile(radii, 0.99)),
            "maximum_three_sigma_radius_m": float(radii.max()),
            "mean_offset_m": float(np.linalg.norm(self.means.mean(axis=0))),
            "aabb_midpoint_m": [float(value) for value in ((lower + upper) * 0.5)],
        }


def build_procedural_ball_geometry(
    *,
    nominal_diameter_m: float = 0.067,
    gaussian_count: int = 512,
    gaussian_sigma_fraction: float = 0.055,
    opacity: float = 0.98,
) -> ProceduralBallGeometry:
    """Build an antipodally symmetric Gaussian shell with a 3σ ball envelope."""
    if isinstance(gaussian_count, bool) or (
        gaussian_count < _MIN_GAUSSIAN_COUNT or gaussian_count % 2 != 0
    ):
        raise ValueError(
            f"gaussian_count must be an even integer >= {_MIN_GAUSSIAN_COUNT}."
        )
    if not math.isfinite(nominal_diameter_m) or not (
        0.05 <= nominal_diameter_m <= 0.09
    ):
        raise ValueError("nominal_diameter_m must lie in [0.05, 0.09].")
    if not math.isfinite(gaussian_sigma_fraction) or not (
        0.0 < gaussian_sigma_fraction < 1.0 / 3.0
    ):
        raise ValueError("gaussian_sigma_fraction must lie in (0, 1/3).")
    if not math.isfinite(opacity) or not 0.0 < opacity < 1.0:
        raise ValueError("opacity must lie in (0, 1).")

    half_count = gaussian_count // 2
    indices = np.arange(half_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / half_count
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * indices
    directions = np.stack(
        (radial * np.cos(theta), radial * np.sin(theta), z),
        axis=1,
    )
    directions = np.concatenate((directions, -directions), axis=0)

    nominal_radius = nominal_diameter_m * 0.5
    standard_deviation = nominal_radius * gaussian_sigma_fraction
    centre_radius = nominal_radius - 3.0 * standard_deviation
    means = np.asarray(directions * centre_radius, dtype=np.float32)
    quats: NDArray[np.float32] = np.zeros((gaussian_count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    log_scales: NDArray[np.float32] = np.full(
        (gaussian_count, 3),
        math.log(standard_deviation),
        dtype=np.float32,
    )
    opacity_logits: NDArray[np.float32] = np.full(
        (gaussian_count,),
        math.log(opacity / (1.0 - opacity)),
        dtype=np.float32,
    )
    return ProceduralBallGeometry(
        means=means,
        quats=quats,
        log_scales=log_scales,
        opacity_logits=opacity_logits,
        nominal_diameter_m=float(nominal_diameter_m),
    )

"""Explicit closed orbit/open curve descriptors and frozen arc-length maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_sampling.shapes import (
    unit_shape_points,
)
from src.synthetic_data_generation.dataset.court.contracts import OrbitShape
from src.synthetic_data_generation.scene_contract import RigidTransform

from .core import GeometryRecipe, require_integer, stable_id


@dataclass(frozen=True)
class Curve:
    """Coordinates are local to one pinned metric reference transform."""

    kind: str
    scene_from_local: RigidTransform
    parameters: dict[str, Any]
    center_court_id: str | None

    def __post_init__(self) -> None:
        fields = {
            "orbit": {
                "shape",
                "radius_x_m",
                "radius_y_m",
                "orientation",
                "height",
                "amplitude",
                "phase",
                "cycles",
            },
            "line": {"controls"},
            "bezier": {"controls"},
        }
        if self.kind not in fields or set(self.parameters) != fields[self.kind]:
            raise ValueError("Unknown curve kind or invalid curve parameters.")
        if self.kind == "orbit":
            p = self.parameters
            OrbitShape(p["shape"])
            if any(not math.isfinite(p[k]) for k in fields["orbit"] - {"shape"}):
                raise ValueError("Orbit parameters must be finite.")
            if (
                min(p["radius_x_m"], p["radius_y_m"], p["height"] - abs(p["amplitude"]))
                <= 0
            ):
                raise ValueError("Orbit radius and complete height must be positive.")
        else:
            controls = np.asarray(self.parameters["controls"], dtype=float)
            if (
                controls.shape != ((2 if self.kind == "line" else 4), 3)
                or not np.isfinite(controls).all()
                or np.any(controls[:, 2] <= 0)
            ):
                raise ValueError(
                    "Open curves require finite positive-height control points."
                )
            # Positive progress along the endpoint chord proves a regular tangent everywhere.
            chord = controls[-1] - controls[0]
            if np.any(np.diff(controls, axis=0) @ chord <= 0):
                raise ValueError("Open curve has a singular or reversing tangent.")

    @property
    def closed(self) -> bool:
        return self.kind == "orbit"

    def physical_key(self) -> dict[str, Any]:
        """Remove equivalent path parameterizations, including open-path reversal."""
        descriptor = self.to_dict()
        parameters = dict(self.parameters)
        if self.closed:
            if parameters["shape"] == "circle":
                period = 0.0
            elif parameters["radius_x_m"] == parameters["radius_y_m"] and parameters[
                "shape"
            ] in {"rectangle", "superellipse"}:
                period = math.pi / 2
            else:
                period = math.pi
            angle = parameters["orientation"]
            normalized = angle % period if period else 0.0
            parameters["orientation"] = normalized
            parameters["phase"] = (
                (
                    (parameters["phase"] - parameters["cycles"] * (angle - normalized))
                    % (2 * math.pi)
                )
                if parameters["amplitude"]
                else 0.0
            )
        else:
            controls = parameters["controls"]
            parameters["controls"] = min(controls, controls[::-1])
        descriptor["parameters"] = parameters

        def rounded(value: Any) -> Any:
            if isinstance(value, float):
                return round(value, 12)
            if isinstance(value, list):
                return [rounded(v) for v in value]
            if isinstance(value, dict):
                return {k: rounded(v) for k, v in value.items()}
            return value

        return dict(rounded(descriptor))

    def at(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        p = self.parameters
        if self.kind == "orbit":
            theta = t * (2 * math.pi)
            xy = unit_shape_points(OrbitShape(p["shape"]), theta) * [
                p["radius_x_m"],
                p["radius_y_m"],
            ]
            c, s = math.cos(p["orientation"]), math.sin(p["orientation"])
            xy = xy @ np.array([[c, s], [-s, c]])
            local = np.column_stack(
                [
                    xy,
                    p["height"]
                    + p["amplitude"] * np.sin(p["cycles"] * theta + p["phase"]),
                ]
            )
        else:
            controls = np.asarray(p["controls"], dtype=np.float64)
            u = t[:, None]
            if self.kind == "line":
                local = (1 - u) * controls[0] + u * controls[1]
            else:
                local = (
                    (1 - u) ** 3 * controls[0]
                    + 3 * (1 - u) ** 2 * u * controls[1]
                    + 3 * (1 - u) * u**2 * controls[2]
                    + u**3 * controls[3]
                )
        return self.scene_from_local.apply(local)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scene_from_local": self.scene_from_local.to_list(),
            "parameters": self.parameters,
            "center_court_id": self.center_court_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Curve:
        if set(value) != {"kind", "scene_from_local", "parameters", "center_court_id"}:
            raise ValueError("Invalid curve descriptor keys.")
        return cls(
            value["kind"],
            RigidTransform(tuple(value["scene_from_local"])),
            value["parameters"],
            value["center_court_id"],
        )


@dataclass(frozen=True)
class ArcMap:
    curve: Curve
    t: NDArray[np.float64]
    cumulative_m: NDArray[np.float64]
    base_count: int
    convergence_error_m: float

    @classmethod
    def freeze(cls, curve: Curve, recipe: GeometryRecipe) -> ArcMap:
        t = np.linspace(0, 1, recipe.arc_intervals + 1, dtype=np.float64)
        points = curve.at(t)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if not np.isfinite(steps).all() or np.any(steps <= 0):
            raise ValueError("Curve has a non-positive or non-finite arc interval.")
        cumulative = np.r_[0.0, np.cumsum(steps)]
        coarse_length = np.linalg.norm(np.diff(points[::2], axis=0), axis=1).sum()
        error = abs(float(cumulative[-1] - coarse_length))
        if error > recipe.arc_tolerance_m:
            raise ValueError(f"Frozen arc map has not converged: {error} m.")
        n = max(
            recipe.minimum_points, math.ceil(float(cumulative[-1]) / recipe.base_step_m)
        )
        multiple = recipe.count_multiple
        n = ((n + multiple - 1) // multiple) * multiple
        t.flags.writeable = cumulative.flags.writeable = False
        return cls(curve, t, cumulative, n, error)

    @property
    def length_m(self) -> float:
        return float(self.cumulative_m[-1])

    def at(self, q: Fraction) -> NDArray[np.float64]:
        if not 0 <= q <= 1:
            raise ValueError("Arc fraction lies outside [0,1].")
        t = np.interp(float(q) * self.length_m, self.cumulative_m, self.t)
        return self.curve.at(np.array([t]))[0]

    def fractions(self, density_level: int) -> list[Fraction]:
        require_integer(density_level, "density_level", minimum=0)
        n = self.base_count * 2**density_level
        return [Fraction(j, n) for j in range(n)]

    def point_id(self, trajectory_id: str, view_id: str, q: Fraction) -> str:
        return str(
            stable_id("point", trajectory_id, view_id, [q.numerator, q.denominator])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "court_scale_frozen_arc_map_v1",
            "algorithm": "fixed_polyline_binary64_v1",
            "base_count": self.base_count,
            "length_m": self.length_m,
            "convergence_error_m": self.convergence_error_m,
            "t": self.t.tolist(),
            "cumulative_m": self.cumulative_m.tolist(),
        }

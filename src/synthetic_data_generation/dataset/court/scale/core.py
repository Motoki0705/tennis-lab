"""Versioned geometry recipes, identity and field-local counter randomness."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

IDENTITY_SCHEMA = "court_scale_identity_v1"


class TrajectoryMethod(StrEnum):
    GRID_BUDGET_REPLAY = "grid_budget_replay"
    STRATIFIED_COUNTER = "stratified_counter"
    SOBOL = "sobol"
    CAPTURE_LOCAL = "capture_local"
    OPEN_BEZIER = "open_bezier"


def canonical_bytes(value: Any) -> bytes:
    """JSON binary64 round trips, sorted UTF-8 keys, no non-finite numbers."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_id(namespace: str, *parts: Any) -> str:
    return digest([IDENTITY_SCHEMA, namespace, *parts])


def field_rng(seed: int, identity: Any, field: str) -> np.random.Generator:
    """Adding another field never consumes this field's random stream."""
    require_integer(seed, "generation_seed", minimum=0)
    entropy = np.frombuffer(
        bytes.fromhex(stable_id("rng", seed, identity, field)), dtype="<u4"
    ).tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def disk_offset(seed: int, identity: str, radius_m: float) -> NDArray[np.float64]:
    radius = radius_m * math.sqrt(
        float(field_rng(seed, identity, "look_at.radius").random())
    )
    angle = 2 * math.pi * float(field_rng(seed, identity, "look_at.angle").random())
    return np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])


def require_integer(value: int, name: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True)
class GeometryRecipe:
    """D2/D6 and arc precision are explicit and independent of method and QA."""

    schema: str = "court_scale_geometry_recipe_v1"
    hull_expansion_percent: float = 0.0
    hull_margin_m: float = 0.5
    hfov_degrees: tuple[float, ...] = (75.0, 92.5, 110.0)
    look_at_heights_m: tuple[float, ...] = (0.0, 1.5)
    jitter_radius_m: float = 1.0
    base_step_m: float = 4.0
    minimum_points: int = 8
    count_multiple: int = 1
    arc_intervals: int = 8192
    arc_tolerance_m: float = 0.01
    near_plane_m: float = 0.01
    width: int = 640
    height: int = 360

    def __post_init__(self) -> None:
        if self.schema != "court_scale_geometry_recipe_v1":
            raise ValueError("Unknown geometry recipe schema.")
        for name in ("hull_expansion_percent", "hull_margin_m", "jitter_radius_m"):
            v = getattr(self, name)
            if isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("base_step_m", "arc_tolerance_m", "near_plane_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in (
            "minimum_points",
            "count_multiple",
            "arc_intervals",
            "width",
            "height",
        ):
            require_integer(getattr(self, name), name, minimum=1)
        if self.arc_intervals < 16 or self.arc_intervals % 8:
            raise ValueError("arc_intervals must be a multiple of eight >= 16.")
        if len(self.hfov_degrees) != 3 or any(
            not math.isfinite(x) or not 0 < x < 180 for x in self.hfov_degrees
        ):
            raise ValueError(
                "Exactly three finite HFOV layers in (0,180) are required."
            )
        if len(self.look_at_heights_m) != 2 or any(
            not math.isfinite(x) or x < 0 for x in self.look_at_heights_m
        ):
            raise ValueError("Exactly two non-negative look-at heights are required.")

    def to_dict(self) -> dict[str, Any]:
        result = json.loads(canonical_bytes(asdict(self)))
        assert isinstance(result, dict)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GeometryRecipe:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Geometry recipe keys do not match the declared schema.")
        return cls(
            **{
                **value,
                "hfov_degrees": tuple(value["hfov_degrees"]),
                "look_at_heights_m": tuple(value["look_at_heights_m"]),
            }
        )


@dataclass(frozen=True)
class QualityPolicy:
    """Existing visibility gates by default; new hard gates require explicit values."""

    schema: str = "court_scale_quality_policy_v1"
    minimum_target_visible_kp: int | None = None
    maximum_roi_hole_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.schema != "court_scale_quality_policy_v1":
            raise ValueError("Unknown quality policy schema.")
        if self.minimum_target_visible_kp is not None:
            require_integer(
                self.minimum_target_visible_kp, "minimum_target_visible_kp", minimum=0
            )
            if self.minimum_target_visible_kp > 14:
                raise ValueError("Target visibility cannot exceed 14 KP.")
        if (
            self.maximum_roi_hole_fraction is not None
            and not 0 <= self.maximum_roi_hole_fraction <= 1
        ):
            raise ValueError("ROI hole threshold must lie in [0,1].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

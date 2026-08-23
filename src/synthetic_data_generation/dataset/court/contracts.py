"""Strict semantic contracts for the canonical Court dataset stage."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_DATASET_SCHEMA_V1,
    COURT_DATASET_SCHEMA_V2,
    COURT_DATASET_SCHEMA_V3,
    COURT_PLAN_SCHEMA_V1,
    COURT_PLAN_SCHEMA_V2,
    COURT_PLAN_SCHEMA_V3,
    COURT_SAMPLE_SCHEMA_V1,
    COURT_SAMPLE_SCHEMA_V2,
    COURT_SAMPLE_SCHEMA_V3,
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera

if TYPE_CHECKING:
    from src.synthetic_data_generation.configuration import CourtSamplingPolicy

# Legacy public names remain exact v1 authorities.  Version-aware boundaries
# use the suffixed constants rather than changing these values in place.
COURT_DATASET_SCHEMA = COURT_DATASET_SCHEMA_V1
COURT_PLAN_SCHEMA = COURT_PLAN_SCHEMA_V1
COURT_SAMPLE_SCHEMA = COURT_SAMPLE_SCHEMA_V1


class OrbitShape(StrEnum):
    """Supported camera-centre curve shapes."""

    CIRCLE = "circle"
    ELLIPSE = "ellipse"


class OrbitCenterKind(StrEnum):
    """Typed centre authority for a camera-centre curve."""

    COMPLEX = "complex"
    COURT = "court"


class OrbitCurveMode(StrEnum):
    """Supported vertical trajectory modes."""

    PLANAR = "planar"
    SINUSOIDAL_HEIGHT = "sinusoidal_height"


class OrbitTargetKind(StrEnum):
    """Typed look-at target authority."""

    COMPLEX = "complex"
    COURT = "court"


class OrbitTargetMode(StrEnum):
    """Finite configured look-at targets, including their owning frame."""

    COMPLEX_CENTER = "complex_center"
    COURT_CENTER = "court_center"
    NEAR_BASELINE = "near_baseline"
    FAR_BASELINE = "far_baseline"

    @property
    def target_kind(self) -> OrbitTargetKind:
        """Return the coordinate-frame authority for this target."""
        if self is OrbitTargetMode.COMPLEX_CENTER:
            return OrbitTargetKind.COMPLEX
        return OrbitTargetKind.COURT


class OrbitCoverageMode(StrEnum):
    """Requested framing diversity carried independently of trajectory geometry."""

    FULL = "full"
    NEAR_FULL = "near_full"
    PARTIAL = "partial"


class DatasetSplit(StrEnum):
    """Group-disjoint dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class OrbitSamplingMode(StrEnum):
    """The sole production sampling algorithm."""

    UNIFORM_ARC_LENGTH = "uniform_arc_length"


class OrbitStableField(StrEnum):
    """Finite typed fields used for canonical candidate ordering."""

    SHAPE = "shape"
    CENTER_KIND = "center_kind"
    RADIUS_SCALE = "radius_scale"
    AXIS_RATIO = "axis_ratio"
    ORIENTATION_DEGREES = "orientation_degrees"
    BASE_HEIGHT_M = "base_height_m"
    VERTICAL_MODULATION_M = "vertical_modulation_m"
    CURVE_MODE = "curve_mode"


class OrbitCoverageObjective(StrEnum):
    """Ordered token families available to the greedy coverage selector."""

    COVERAGE_MODE = "coverage_mode"
    SEMANTIC_VISIBILITY = "semantic_visibility"
    TRAJECTORY_GROUP = "trajectory_group"


class TargetCourtResolutionPolicy(StrEnum):
    """Discriminant for the v2 group policy and resolved sample target."""

    TRAJECTORY_CENTER_COURT = "trajectory_center_court"
    NEAREST_CAMERA = "nearest_camera"


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name=name)


def _strict(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys do not match; missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}."
        )
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return value


def _finite_vector(
    value: Sequence[object], *, size: int, name: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} numeric values.")
    return tuple(_finite(item, name=name) for item in value)


def _required_sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a non-string sequence.")
    return tuple(value)


def _enum_text_sequence(
    value: object,
    *,
    enum_type: type[StrEnum],
    name: str,
) -> tuple[str, ...]:
    values = _required_sequence(value, name=name)
    texts = tuple(_text(item, name=name) for item in values)
    if not texts or len(texts) != len(set(texts)):
        raise ValueError(f"{name} must be non-empty and unique.")
    for text in texts:
        enum_type(text)
    return texts


@dataclass(frozen=True, slots=True)
class OrbitTrajectorySpec:
    """One typed camera-centre path, independent of view and sample policy."""

    trajectory_id: str
    trajectory_group_id: str
    shape: OrbitShape
    center_kind: OrbitCenterKind
    center_court_instance_id: str | None
    base_radius_m: float
    radius_scale: float
    axis_ratio: float
    orientation_radians: float
    base_height_m: float
    vertical_amplitude_m: float
    vertical_cycles: int
    vertical_phase_radians: float
    curve_mode: OrbitCurveMode

    def __post_init__(self) -> None:
        _text(self.trajectory_id, name="trajectory_id")
        _text(self.trajectory_group_id, name="trajectory_group_id")
        if not isinstance(self.shape, OrbitShape):
            raise TypeError("shape must be an OrbitShape.")
        if not isinstance(self.center_kind, OrbitCenterKind):
            raise TypeError("center_kind must be an OrbitCenterKind.")
        if not isinstance(self.curve_mode, OrbitCurveMode):
            raise TypeError("curve_mode must be an OrbitCurveMode.")
        center_id = _optional_text(
            self.center_court_instance_id,
            name="center_court_instance_id",
        )
        if (self.center_kind is OrbitCenterKind.COURT) != (center_id is not None):
            raise ValueError(
                "center_court_instance_id is required exactly for court-centred trajectories."
            )
        base_radius = _finite(self.base_radius_m, name="base_radius_m")
        radius_scale = _finite(self.radius_scale, name="radius_scale")
        axis_ratio = _finite(self.axis_ratio, name="axis_ratio")
        orientation = _finite(self.orientation_radians, name="orientation_radians")
        base_height = _finite(self.base_height_m, name="base_height_m")
        amplitude = _finite(self.vertical_amplitude_m, name="vertical_amplitude_m")
        cycles = _integer(self.vertical_cycles, name="vertical_cycles", minimum=0)
        phase = _finite(self.vertical_phase_radians, name="vertical_phase_radians")
        if base_radius <= 0.0 or radius_scale <= 0.0:
            raise ValueError("base_radius_m and radius_scale must be positive.")
        if not 0.0 < axis_ratio <= 1.0:
            raise ValueError("axis_ratio must lie in (0, 1].")
        if self.shape is OrbitShape.CIRCLE and axis_ratio != 1.0:
            raise ValueError("A circle must have axis_ratio exactly 1.0.")
        if self.shape is OrbitShape.ELLIPSE and axis_ratio > 0.8:
            raise ValueError("A production ellipse must have axis_ratio <= 0.8.")
        if base_height <= 0.0 or amplitude < 0.0:
            raise ValueError(
                "base_height_m must be positive and amplitude non-negative."
            )
        if self.curve_mode is OrbitCurveMode.PLANAR and (
            amplitude != 0.0 or cycles != 0
        ):
            raise ValueError("Planar trajectories require zero amplitude and cycles.")
        if self.curve_mode is OrbitCurveMode.SINUSOIDAL_HEIGHT and (
            amplitude <= 0.0 or cycles <= 0
        ):
            raise ValueError(
                "Non-planar trajectories require positive amplitude and cycles."
            )
        object.__setattr__(self, "base_radius_m", base_radius)
        object.__setattr__(self, "radius_scale", radius_scale)
        object.__setattr__(self, "axis_ratio", axis_ratio)
        object.__setattr__(self, "orientation_radians", orientation)
        object.__setattr__(self, "base_height_m", base_height)
        object.__setattr__(self, "vertical_amplitude_m", amplitude)
        object.__setattr__(self, "vertical_cycles", cycles)
        object.__setattr__(self, "vertical_phase_radians", phase)

    @property
    def radius_x_m(self) -> float:
        """Return the major radius in metres."""
        return self.base_radius_m * self.radius_scale

    @property
    def radius_y_m(self) -> float:
        """Return the minor radius in metres."""
        return self.radius_x_m * self.axis_ratio

    def semantic_key(self) -> tuple[object, ...]:
        """Return typed semantics without interpreting either opaque ID."""
        return (
            self.shape,
            self.center_kind,
            self.center_court_instance_id,
            self.base_radius_m,
            self.radius_scale,
            self.axis_ratio,
            self.orientation_radians,
            self.base_height_m,
            self.vertical_amplitude_m,
            self.vertical_cycles,
            self.vertical_phase_radians,
            self.curve_mode,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict semantic representation."""
        return {
            "trajectory_id": self.trajectory_id,
            "trajectory_group_id": self.trajectory_group_id,
            "shape": self.shape.value,
            "center_kind": self.center_kind.value,
            "center_court_instance_id": self.center_court_instance_id,
            "base_radius_m": self.base_radius_m,
            "radius_scale": self.radius_scale,
            "axis_ratio": self.axis_ratio,
            "orientation_radians": self.orientation_radians,
            "base_height_m": self.base_height_m,
            "vertical_amplitude_m": self.vertical_amplitude_m,
            "vertical_cycles": self.vertical_cycles,
            "vertical_phase_radians": self.vertical_phase_radians,
            "curve_mode": self.curve_mode.value,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse a trajectory and reject unknown fields and modes."""
        keys = {
            "trajectory_id",
            "trajectory_group_id",
            "shape",
            "center_kind",
            "center_court_instance_id",
            "base_radius_m",
            "radius_scale",
            "axis_ratio",
            "orientation_radians",
            "base_height_m",
            "vertical_amplitude_m",
            "vertical_cycles",
            "vertical_phase_radians",
            "curve_mode",
        }
        raw = _strict(value, keys=keys, name="orbit trajectory")
        return cls(
            trajectory_id=_text(raw["trajectory_id"], name="trajectory_id"),
            trajectory_group_id=_text(
                raw["trajectory_group_id"], name="trajectory_group_id"
            ),
            shape=OrbitShape(_text(raw["shape"], name="shape")),
            center_kind=OrbitCenterKind(_text(raw["center_kind"], name="center_kind")),
            center_court_instance_id=_optional_text(
                raw["center_court_instance_id"],
                name="center_court_instance_id",
            ),
            base_radius_m=_finite(raw["base_radius_m"], name="base_radius_m"),
            radius_scale=_finite(raw["radius_scale"], name="radius_scale"),
            axis_ratio=_finite(raw["axis_ratio"], name="axis_ratio"),
            orientation_radians=_finite(
                raw["orientation_radians"], name="orientation_radians"
            ),
            base_height_m=_finite(raw["base_height_m"], name="base_height_m"),
            vertical_amplitude_m=_finite(
                raw["vertical_amplitude_m"], name="vertical_amplitude_m"
            ),
            vertical_cycles=_integer(
                raw["vertical_cycles"], name="vertical_cycles", minimum=0
            ),
            vertical_phase_radians=_finite(
                raw["vertical_phase_radians"], name="vertical_phase_radians"
            ),
            curve_mode=OrbitCurveMode(_text(raw["curve_mode"], name="curve_mode")),
        )


@dataclass(frozen=True, slots=True)
class OrbitViewSpec:
    """One typed look-at/framing variant, independent of the camera-centre path."""

    view_id: str
    target_kind: OrbitTargetKind
    target_court_instance_id: str | None
    target_mode: OrbitTargetMode
    coverage_mode: OrbitCoverageMode
    look_at_height_m: float
    hfov_degrees: float

    def __post_init__(self) -> None:
        _text(self.view_id, name="view_id")
        if not isinstance(self.target_kind, OrbitTargetKind):
            raise TypeError("target_kind must be an OrbitTargetKind.")
        target_id = _optional_text(
            self.target_court_instance_id,
            name="target_court_instance_id",
        )
        if (self.target_kind is OrbitTargetKind.COURT) != (target_id is not None):
            raise ValueError(
                "target_court_instance_id is required exactly for court targets."
            )
        if not isinstance(self.target_mode, OrbitTargetMode):
            raise TypeError("target_mode must be an OrbitTargetMode.")
        if self.target_mode.target_kind is not self.target_kind:
            raise ValueError(
                f"Target mode {self.target_mode.value!r} requires "
                f"target_kind={self.target_mode.target_kind.value!r}."
            )
        if not isinstance(self.coverage_mode, OrbitCoverageMode):
            raise TypeError("coverage_mode must be an OrbitCoverageMode.")
        height = _finite(self.look_at_height_m, name="look_at_height_m")
        hfov = _finite(self.hfov_degrees, name="hfov_degrees")
        if height < 0.0:
            raise ValueError("look_at_height_m must be non-negative.")
        if not 0.0 < hfov < 180.0:
            raise ValueError("hfov_degrees must lie in (0, 180).")
        object.__setattr__(self, "look_at_height_m", height)
        object.__setattr__(self, "hfov_degrees", hfov)

    def semantic_key(self) -> tuple[object, ...]:
        """Return typed view semantics without interpreting ``view_id``."""
        return (
            self.target_kind,
            self.target_court_instance_id,
            self.target_mode,
            self.coverage_mode,
            self.look_at_height_m,
            self.hfov_degrees,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict semantic representation."""
        return {
            "view_id": self.view_id,
            "target_kind": self.target_kind.value,
            "target_court_instance_id": self.target_court_instance_id,
            "target_mode": self.target_mode.value,
            "coverage_mode": self.coverage_mode.value,
            "look_at_height_m": self.look_at_height_m,
            "hfov_degrees": self.hfov_degrees,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse a view and reject unknown fields and modes."""
        keys = {
            "view_id",
            "target_kind",
            "target_court_instance_id",
            "target_mode",
            "coverage_mode",
            "look_at_height_m",
            "hfov_degrees",
        }
        raw = _strict(value, keys=keys, name="orbit view")
        return cls(
            view_id=_text(raw["view_id"], name="view_id"),
            target_kind=OrbitTargetKind(_text(raw["target_kind"], name="target_kind")),
            target_court_instance_id=_optional_text(
                raw["target_court_instance_id"],
                name="target_court_instance_id",
            ),
            target_mode=OrbitTargetMode(_text(raw["target_mode"], name="target_mode")),
            coverage_mode=OrbitCoverageMode(
                _text(raw["coverage_mode"], name="coverage_mode")
            ),
            look_at_height_m=_finite(raw["look_at_height_m"], name="look_at_height_m"),
            hfov_degrees=_finite(raw["hfov_degrees"], name="hfov_degrees"),
        )


@dataclass(frozen=True, slots=True)
class OrbitViewSpecV2:
    """One v2 view whose court binding is resolved separately per sample."""

    view_id: str
    target_kind: OrbitTargetKind
    target_mode: OrbitTargetMode
    coverage_mode: OrbitCoverageMode
    look_at_height_m: float
    hfov_degrees: float

    def __post_init__(self) -> None:
        _text(self.view_id, name="view_id")
        if not isinstance(self.target_kind, OrbitTargetKind):
            raise TypeError("target_kind must be an OrbitTargetKind.")
        if not isinstance(self.target_mode, OrbitTargetMode):
            raise TypeError("target_mode must be an OrbitTargetMode.")
        if (
            self.target_kind is not OrbitTargetKind.COURT
            or self.target_mode is not OrbitTargetMode.COURT_CENTER
        ):
            raise ValueError(
                "Court v2 views require target_kind='court' and "
                "target_mode='court_center'."
            )
        if not isinstance(self.coverage_mode, OrbitCoverageMode):
            raise TypeError("coverage_mode must be an OrbitCoverageMode.")
        height = _finite(self.look_at_height_m, name="look_at_height_m")
        hfov = _finite(self.hfov_degrees, name="hfov_degrees")
        if height < 0.0:
            raise ValueError("look_at_height_m must be non-negative.")
        if not 0.0 < hfov < 180.0:
            raise ValueError("hfov_degrees must lie in (0, 180).")
        object.__setattr__(self, "look_at_height_m", height)
        object.__setattr__(self, "hfov_degrees", hfov)

    def semantic_key(self) -> tuple[object, ...]:
        """Return the complete typed view semantics."""
        return (
            self.target_kind,
            self.target_mode,
            self.coverage_mode,
            self.look_at_height_m,
            self.hfov_degrees,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict v2 view record without a static court binding."""
        return {
            "view_id": self.view_id,
            "target_kind": self.target_kind.value,
            "target_mode": self.target_mode.value,
            "coverage_mode": self.coverage_mode.value,
            "look_at_height_m": self.look_at_height_m,
            "hfov_degrees": self.hfov_degrees,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse one exact v2 view and reject v1's static target field."""
        raw = _strict(
            value,
            keys={
                "view_id",
                "target_kind",
                "target_mode",
                "coverage_mode",
                "look_at_height_m",
                "hfov_degrees",
            },
            name="v2 orbit view",
        )
        return cls(
            view_id=_text(raw["view_id"], name="view_id"),
            target_kind=OrbitTargetKind(_text(raw["target_kind"], name="target_kind")),
            target_mode=OrbitTargetMode(_text(raw["target_mode"], name="target_mode")),
            coverage_mode=OrbitCoverageMode(
                _text(raw["coverage_mode"], name="coverage_mode")
            ),
            look_at_height_m=_finite(raw["look_at_height_m"], name="look_at_height_m"),
            hfov_degrees=_finite(raw["hfov_degrees"], name="hfov_degrees"),
        )


@dataclass(frozen=True, slots=True)
class TargetCourtPolicyV2:
    """Strict group policy; absence is meaningful only in nearest mode."""

    mode: TargetCourtResolutionPolicy
    centre_court_instance_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TargetCourtResolutionPolicy):
            raise TypeError("mode must be a TargetCourtResolutionPolicy.")
        centre_id = _optional_text(
            self.centre_court_instance_id,
            name="centre_court_instance_id",
        )
        if (self.mode is TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT) != (
            centre_id is not None
        ):
            raise ValueError(
                "centre_court_instance_id is required exactly for "
                "trajectory_center_court policy."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the discriminated v2 target policy."""
        return {
            "mode": self.mode.value,
            "centre_court_instance_id": self.centre_court_instance_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse exact policy keys without treating null as a fallback."""
        raw = _strict(
            value,
            keys={"mode", "centre_court_instance_id"},
            name="v2 target court policy",
        )
        return cls(
            mode=TargetCourtResolutionPolicy(_text(raw["mode"], name="mode")),
            centre_court_instance_id=_optional_text(
                raw["centre_court_instance_id"],
                name="centre_court_instance_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedTargetCourtV2:
    """Sample-owned v2 binding and recomputable geometric evidence."""

    binding: TargetCourtBinding
    resolution_policy: TargetCourtResolutionPolicy
    camera_to_court_center_distance_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.binding, TargetCourtBinding):
            raise TypeError("binding must be a TargetCourtBinding.")
        if not isinstance(self.resolution_policy, TargetCourtResolutionPolicy):
            raise TypeError("resolution_policy must be a TargetCourtResolutionPolicy.")
        distance = _finite(
            self.camera_to_court_center_distance_m,
            name="camera_to_court_center_distance_m",
        )
        if distance < 0.0:
            raise ValueError("camera_to_court_center_distance_m must be non-negative.")
        object.__setattr__(
            self,
            "camera_to_court_center_distance_m",
            distance,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact sample-owned target record."""
        return {
            "binding": self.binding.to_dict(),
            "resolution_policy": self.resolution_policy.value,
            "camera_to_court_center_distance_m": (
                self.camera_to_court_center_distance_m
            ),
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse exact v2 target evidence and reject missing geometry."""
        raw = _strict(
            value,
            keys={
                "binding",
                "resolution_policy",
                "camera_to_court_center_distance_m",
            },
            name="v2 resolved target court",
        )
        return cls(
            binding=TargetCourtBinding.from_dict(raw["binding"]),
            resolution_policy=TargetCourtResolutionPolicy(
                _text(raw["resolution_policy"], name="resolution_policy")
            ),
            camera_to_court_center_distance_m=_finite(
                raw["camera_to_court_center_distance_m"],
                name="camera_to_court_center_distance_m",
            ),
        )


@dataclass(frozen=True, slots=True)
class OrbitSamplingPolicy:
    """Typed arc-length, selection, split, shard, and quality policy."""

    mode: OrbitSamplingMode
    max_arc_step_m: float
    minimum_sample_count: int
    sample_count_multiple: int
    seed: int
    stable_field_order: tuple[OrbitStableField, ...]
    coverage_objective: tuple[OrbitCoverageObjective, ...]
    proposal_budget: int
    minimum_trajectory_groups: int
    minimum_accepted_frames: int
    minimum_accepted_fraction: float
    split_fractions: tuple[float, float, float]
    shard_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.mode, OrbitSamplingMode):
            raise TypeError("mode must be an OrbitSamplingMode.")
        step = _finite(self.max_arc_step_m, name="max_arc_step_m")
        minimum_samples = _integer(
            self.minimum_sample_count,
            name="minimum_sample_count",
            minimum=8,
        )
        multiple = _integer(
            self.sample_count_multiple,
            name="sample_count_multiple",
            minimum=1,
        )
        seed = _integer(self.seed, name="seed", minimum=0)
        budget = _integer(self.proposal_budget, name="proposal_budget", minimum=1)
        minimum_groups = _integer(
            self.minimum_trajectory_groups,
            name="minimum_trajectory_groups",
            minimum=1,
        )
        minimum_frames = _integer(
            self.minimum_accepted_frames,
            name="minimum_accepted_frames",
            minimum=1,
        )
        accepted_fraction = _finite(
            self.minimum_accepted_fraction,
            name="minimum_accepted_fraction",
        )
        shard_count = _integer(self.shard_count, name="shard_count", minimum=1)
        if not 0.0 < step <= 1.05:
            raise ValueError("max_arc_step_m must lie in (0, 1.05].")
        if minimum_samples % multiple != 0:
            raise ValueError(
                "minimum_sample_count must be divisible by sample_count_multiple."
            )
        if (
            not self.stable_field_order
            or any(
                not isinstance(field, OrbitStableField)
                for field in self.stable_field_order
            )
            or len(self.stable_field_order) != len(set(self.stable_field_order))
        ):
            raise ValueError("stable_field_order must be non-empty and unique.")
        if (
            not self.coverage_objective
            or any(
                not isinstance(objective, OrbitCoverageObjective)
                for objective in self.coverage_objective
            )
            or len(self.coverage_objective) != len(set(self.coverage_objective))
        ):
            raise ValueError("coverage_objective must be non-empty, typed, and unique.")
        if budget > 5_000:
            raise ValueError("proposal_budget must not exceed 5,000.")
        if minimum_groups < 24:
            raise ValueError("Court production requires at least 24 trajectory groups.")
        if minimum_frames < 2_000:
            raise ValueError(
                "Court production requires at least 2,000 accepted frames."
            )
        if not 0.9 <= accepted_fraction <= 1.0:
            raise ValueError("minimum_accepted_fraction must lie in [0.9, 1].")
        if math.ceil(minimum_frames / accepted_fraction) > budget:
            raise ValueError(
                "proposal_budget cannot satisfy accepted frames at the minimum fraction."
            )
        fractions = tuple(
            _finite(value, name="split_fractions") for value in self.split_fractions
        )
        if (
            len(fractions) != 3
            or min(fractions) <= 0.0
            or not math.isclose(sum(fractions), 1.0, abs_tol=1.0e-12, rel_tol=0.0)
        ):
            raise ValueError(
                "split_fractions must be three positive values summing to one."
            )
        if shard_count > minimum_groups:
            raise ValueError("shard_count cannot exceed minimum_trajectory_groups.")
        object.__setattr__(self, "max_arc_step_m", step)
        object.__setattr__(self, "minimum_sample_count", minimum_samples)
        object.__setattr__(self, "sample_count_multiple", multiple)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "proposal_budget", budget)
        object.__setattr__(self, "minimum_trajectory_groups", minimum_groups)
        object.__setattr__(self, "minimum_accepted_frames", minimum_frames)
        object.__setattr__(self, "minimum_accepted_fraction", accepted_fraction)
        object.__setattr__(self, "split_fractions", fractions)
        object.__setattr__(self, "shard_count", shard_count)

    @classmethod
    def from_configuration(cls, value: CourtSamplingPolicy) -> Self:
        """Build the complete domain policy from strict shared Court configuration."""
        shard_count = _integer(
            value.shard_group_count,
            name="shard_group_count",
            minimum=1,
        )
        minimum_groups = _integer(
            value.minimum_trajectory_groups,
            name="minimum_trajectory_groups",
            minimum=1,
        )
        minimum_samples = (
            int(math.ceil(max(8, minimum_groups) / shard_count)) * shard_count
        )
        return cls(
            mode=value.mode,
            max_arc_step_m=_finite(
                value.maximum_adjacent_step_m,
                name="maximum_adjacent_step_m",
            ),
            minimum_sample_count=minimum_samples,
            sample_count_multiple=shard_count,
            seed=_integer(value.seed, name="seed", minimum=0),
            stable_field_order=tuple(value.stable_field_order),
            coverage_objective=tuple(value.coverage_objective),
            proposal_budget=_integer(
                value.proposal_budget,
                name="proposal_budget",
                minimum=1,
            ),
            minimum_trajectory_groups=minimum_groups,
            minimum_accepted_frames=_integer(
                value.minimum_accepted_frames,
                name="minimum_accepted_frames",
                minimum=1,
            ),
            minimum_accepted_fraction=_finite(
                value.minimum_accepted_fraction,
                name="minimum_accepted_fraction",
            ),
            split_fractions=(
                _finite(value.train_fraction, name="train_fraction"),
                _finite(
                    value.validation_fraction,
                    name="validation_fraction",
                ),
                _finite(value.test_fraction, name="test_fraction"),
            ),
            shard_count=shard_count,
        )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse a persisted sampling policy with exact keys and finite enums."""
        keys = {
            "mode",
            "max_arc_step_m",
            "minimum_sample_count",
            "sample_count_multiple",
            "seed",
            "stable_field_order",
            "coverage_objective",
            "proposal_budget",
            "minimum_trajectory_groups",
            "minimum_accepted_frames",
            "minimum_accepted_fraction",
            "split_fractions",
            "shard_count",
        }
        raw = _strict(value, keys=keys, name="orbit sampling policy")
        stable_fields = _enum_text_sequence(
            raw["stable_field_order"],
            enum_type=OrbitStableField,
            name="stable_field_order",
        )
        objectives = _enum_text_sequence(
            raw["coverage_objective"],
            enum_type=OrbitCoverageObjective,
            name="coverage_objective",
        )
        split_fractions = _finite_vector(
            _required_sequence(raw["split_fractions"], name="split_fractions"),
            size=3,
            name="split_fractions",
        )
        return cls(
            mode=OrbitSamplingMode(_text(raw["mode"], name="mode")),
            max_arc_step_m=_finite(raw["max_arc_step_m"], name="max_arc_step_m"),
            minimum_sample_count=_integer(
                raw["minimum_sample_count"],
                name="minimum_sample_count",
                minimum=8,
            ),
            sample_count_multiple=_integer(
                raw["sample_count_multiple"],
                name="sample_count_multiple",
                minimum=1,
            ),
            seed=_integer(raw["seed"], name="seed", minimum=0),
            stable_field_order=tuple(
                OrbitStableField(field) for field in stable_fields
            ),
            coverage_objective=tuple(
                OrbitCoverageObjective(objective) for objective in objectives
            ),
            proposal_budget=_integer(
                raw["proposal_budget"],
                name="proposal_budget",
                minimum=1,
            ),
            minimum_trajectory_groups=_integer(
                raw["minimum_trajectory_groups"],
                name="minimum_trajectory_groups",
                minimum=1,
            ),
            minimum_accepted_frames=_integer(
                raw["minimum_accepted_frames"],
                name="minimum_accepted_frames",
                minimum=1,
            ),
            minimum_accepted_fraction=_finite(
                raw["minimum_accepted_fraction"],
                name="minimum_accepted_fraction",
            ),
            split_fractions=(
                split_fractions[0],
                split_fractions[1],
                split_fractions[2],
            ),
            shard_count=_integer(
                raw["shard_count"],
                name="shard_count",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return all resolved values used by planning and release gates."""
        return {
            "mode": self.mode.value,
            "max_arc_step_m": self.max_arc_step_m,
            "minimum_sample_count": self.minimum_sample_count,
            "sample_count_multiple": self.sample_count_multiple,
            "seed": self.seed,
            "stable_field_order": [field.value for field in self.stable_field_order],
            "coverage_objective": [
                objective.value for objective in self.coverage_objective
            ],
            "proposal_budget": self.proposal_budget,
            "minimum_trajectory_groups": self.minimum_trajectory_groups,
            "minimum_accepted_frames": self.minimum_accepted_frames,
            "minimum_accepted_fraction": self.minimum_accepted_fraction,
            "split_fractions": list(self.split_fractions),
            "shard_count": self.shard_count,
        }


@dataclass(frozen=True, slots=True)
class OrbitCenter:
    """Resolved local frame and captured-offset radius for one orbit centre."""

    center_kind: OrbitCenterKind
    court_instance_id: str | None
    reference_court_instance_id: str
    scene_from_center: RigidTransform
    base_radius_m: float
    captured_offset_median_m: float
    captured_offset_q90_m: float
    captured_camera_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.center_kind, OrbitCenterKind):
            raise TypeError("center_kind must be an OrbitCenterKind.")
        if not isinstance(self.scene_from_center, RigidTransform):
            raise TypeError("scene_from_center must be a RigidTransform.")
        court_id = _optional_text(self.court_instance_id, name="court_instance_id")
        _text(self.reference_court_instance_id, name="reference_court_instance_id")
        if (self.center_kind is OrbitCenterKind.COURT) != (court_id is not None):
            raise ValueError("court_instance_id is required exactly for court centres.")
        base = _finite(self.base_radius_m, name="base_radius_m")
        median = _finite(
            self.captured_offset_median_m,
            name="captured_offset_median_m",
        )
        q90 = _finite(self.captured_offset_q90_m, name="captured_offset_q90_m")
        count = _integer(
            self.captured_camera_count,
            name="captured_camera_count",
            minimum=1,
        )
        if min(base, median, q90) <= 0.0 or q90 < median:
            raise ValueError("Captured offset radii must be positive and ordered.")
        object.__setattr__(self, "base_radius_m", base)
        object.__setattr__(self, "captured_offset_median_m", median)
        object.__setattr__(self, "captured_offset_q90_m", q90)
        object.__setattr__(self, "captured_camera_count", count)

    @property
    def center_scene_m(self) -> tuple[float, float, float]:
        """Return the resolved scene-space centre."""
        translation = self.scene_from_center.matrix()[:3, 3]
        return (
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        )

    def key(self) -> tuple[OrbitCenterKind, str | None]:
        """Return the typed centre lookup key."""
        return self.center_kind, self.court_instance_id

    def to_dict(self) -> dict[str, object]:
        """Return captured-offset derivation diagnostics."""
        return {
            "center_kind": self.center_kind.value,
            "court_instance_id": self.court_instance_id,
            "reference_court_instance_id": self.reference_court_instance_id,
            "scene_from_center": self.scene_from_center.to_list(),
            "center_scene_m": list(self.center_scene_m),
            "base_radius_m": self.base_radius_m,
            "captured_offset_median_m": self.captured_offset_median_m,
            "captured_offset_q90_m": self.captured_offset_q90_m,
            "captured_camera_count": self.captured_camera_count,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse one exact persisted centre and verify its canonical form."""
        raw = _strict(
            value,
            name="orbit center",
            keys={
                "center_kind",
                "court_instance_id",
                "reference_court_instance_id",
                "scene_from_center",
                "center_scene_m",
                "base_radius_m",
                "captured_offset_median_m",
                "captured_offset_q90_m",
                "captured_camera_count",
            },
        )
        transform_values = _required_sequence(
            raw["scene_from_center"],
            name="scene_from_center",
        )
        scene_from_center = RigidTransform(
            _finite_vector(
                transform_values,
                size=16,
                name="scene_from_center",
            )
        )
        persisted_center_scene_m = _finite_vector(
            _required_sequence(raw["center_scene_m"], name="center_scene_m"),
            size=3,
            name="center_scene_m",
        )
        center = cls(
            center_kind=OrbitCenterKind(
                _text(raw["center_kind"], name="center_kind")
            ),
            court_instance_id=_optional_text(
                raw["court_instance_id"],
                name="court_instance_id",
            ),
            reference_court_instance_id=_text(
                raw["reference_court_instance_id"],
                name="reference_court_instance_id",
            ),
            scene_from_center=scene_from_center,
            base_radius_m=_finite(raw["base_radius_m"], name="base_radius_m"),
            captured_offset_median_m=_finite(
                raw["captured_offset_median_m"],
                name="captured_offset_median_m",
            ),
            captured_offset_q90_m=_finite(
                raw["captured_offset_q90_m"],
                name="captured_offset_q90_m",
            ),
            captured_camera_count=_integer(
                raw["captured_camera_count"],
                name="captured_camera_count",
                minimum=1,
            ),
        )
        if persisted_center_scene_m != center.center_scene_m:
            raise ValueError(
                "center_scene_m disagrees with the scene_from_center translation."
            )
        if dict(raw) != center.to_dict():
            raise ValueError("Orbit center is not in canonical persisted form.")
        return center


@dataclass(frozen=True, slots=True)
class OrbitPathSamples:
    """Uniform 3-D arc-length points for one closed trajectory."""

    trajectory_group_id: str
    theta_radians: NDArray[np.float64]
    points_local_m: NDArray[np.float64]
    points_scene_m: NDArray[np.float64]
    adjacent_steps_m: NDArray[np.float64]
    total_arc_length_m: float

    def __post_init__(self) -> None:
        _text(self.trajectory_group_id, name="trajectory_group_id")
        theta = np.asarray(self.theta_radians, dtype=np.float64)
        local = np.asarray(self.points_local_m, dtype=np.float64)
        scene = np.asarray(self.points_scene_m, dtype=np.float64)
        steps = np.asarray(self.adjacent_steps_m, dtype=np.float64)
        if theta.ndim != 1 or len(theta) < 8:
            raise ValueError("theta_radians must contain at least eight samples.")
        if local.shape != (len(theta), 3) or scene.shape != local.shape:
            raise ValueError("Orbit point arrays must have shape (sample_count, 3).")
        if steps.shape != (len(theta),):
            raise ValueError("adjacent_steps_m must include the closed-loop step.")
        if not all(np.isfinite(value).all() for value in (theta, local, scene, steps)):
            raise ValueError("Orbit sampling arrays must contain only finite values.")
        if np.any(steps <= 0.0):
            raise ValueError("Every adjacent closed-loop step must be positive.")
        length = _finite(self.total_arc_length_m, name="total_arc_length_m")
        if length <= 0.0:
            raise ValueError("total_arc_length_m must be positive.")
        for array in (theta, local, scene, steps):
            array.setflags(write=False)
        object.__setattr__(self, "theta_radians", theta)
        object.__setattr__(self, "points_local_m", local)
        object.__setattr__(self, "points_scene_m", scene)
        object.__setattr__(self, "adjacent_steps_m", steps)
        object.__setattr__(self, "total_arc_length_m", length)


@dataclass(frozen=True, slots=True)
class TrajectoryGroupPlan:
    """One path group, all target variants, one split, shard, and court binding."""

    trajectory: OrbitTrajectorySpec
    center: OrbitCenter
    views: tuple[OrbitViewSpec, ...]
    split: DatasetSplit
    shard_id: str
    target_court: TargetCourtBinding
    sample_count: int
    maximum_adjacent_step_m: float
    total_arc_length_m: float

    def __post_init__(self) -> None:
        if self.center.key() != (
            self.trajectory.center_kind,
            self.trajectory.center_court_instance_id,
        ):
            raise ValueError(
                "Trajectory centre disagrees with resolved centre authority."
            )
        if not self.views:
            raise ValueError("Every trajectory group requires at least one view.")
        if len({view.view_id for view in self.views}) != len(self.views):
            raise ValueError("view_id values must be unique within a trajectory group.")
        if len({view.semantic_key() for view in self.views}) != len(self.views):
            raise ValueError("Duplicate typed view candidates are forbidden.")
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit.")
        _text(self.shard_id, name="shard_id")
        _integer(self.sample_count, name="sample_count", minimum=8)
        maximum_step = _finite(
            self.maximum_adjacent_step_m,
            name="maximum_adjacent_step_m",
        )
        arc_length = _finite(self.total_arc_length_m, name="total_arc_length_m")
        if maximum_step <= 0.0 or arc_length <= 0.0:
            raise ValueError("Arc diagnostics must be positive.")
        object.__setattr__(self, "maximum_adjacent_step_m", maximum_step)
        object.__setattr__(self, "total_arc_length_m", arc_length)

    @property
    def trajectory_group_id(self) -> str:
        """Return the opaque group ID carried by the typed trajectory."""
        return self.trajectory.trajectory_group_id

    def to_dict(self) -> dict[str, object]:
        """Return the complete group metadata."""
        return {
            "trajectory": self.trajectory.to_dict(),
            "center": self.center.to_dict(),
            "views": [view.to_dict() for view in self.views],
            "split": self.split.value,
            "shard_id": self.shard_id,
            "target_court": self.target_court.to_dict(),
            "sample_count": self.sample_count,
            "maximum_adjacent_step_m": self.maximum_adjacent_step_m,
            "total_arc_length_m": self.total_arc_length_m,
        }


@dataclass(frozen=True, slots=True)
class PlannedCourtSample:
    """One deterministic renderer request and label identity."""

    sample_index: int
    sample_id: str
    trajectory_group_id: str
    trajectory_id: str
    view_id: str
    trajectory_frame_index: int
    split: DatasetSplit
    shard_id: str
    camera_center_scene_m: tuple[float, float, float]
    camera: SceneCamera

    def __post_init__(self) -> None:
        _integer(self.sample_index, name="sample_index", minimum=0)
        for name, value in (
            ("sample_id", self.sample_id),
            ("trajectory_group_id", self.trajectory_group_id),
            ("trajectory_id", self.trajectory_id),
            ("view_id", self.view_id),
            ("shard_id", self.shard_id),
        ):
            _text(value, name=name)
        _integer(
            self.trajectory_frame_index,
            name="trajectory_frame_index",
            minimum=0,
        )
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit.")
        center = _finite_vector(
            self.camera_center_scene_m,
            size=3,
            name="camera_center_scene_m",
        )
        actual = self.camera.camera_to_scene.matrix()[:3, 3]
        if not np.allclose(actual, center, atol=1.0e-9, rtol=0.0):
            raise ValueError("camera_center_scene_m disagrees with camera_to_scene.")
        if self.camera.camera_id != self.sample_id:
            raise ValueError("The NHT camera ID must equal the stable sample ID.")
        object.__setattr__(self, "camera_center_scene_m", center)

    def to_dict(self) -> dict[str, object]:
        """Return the strict sample plan record."""
        return {
            "sample_index": self.sample_index,
            "sample_id": self.sample_id,
            "trajectory_group_id": self.trajectory_group_id,
            "trajectory_id": self.trajectory_id,
            "view_id": self.view_id,
            "trajectory_frame_index": self.trajectory_frame_index,
            "split": self.split.value,
            "shard_id": self.shard_id,
            "camera_center_scene_m": list(self.camera_center_scene_m),
            "camera": self.camera.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CourtDatasetPlan:
    """Resolved deterministic plan; the sole authority for render/release counts."""

    scene_id: str
    profile: str
    policy: OrbitSamplingPolicy
    groups: tuple[TrajectoryGroupPlan, ...]
    samples: tuple[PlannedCourtSample, ...]

    def __post_init__(self) -> None:
        _text(self.scene_id, name="scene_id")
        _text(self.profile, name="profile")
        if not self.groups:
            raise ValueError("Court plan must contain trajectory groups.")
        group_ids = [group.trajectory_group_id for group in self.groups]
        trajectory_ids = [group.trajectory.trajectory_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("trajectory_group_id values must be unique.")
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("trajectory_id values must be unique.")
        if len({group.trajectory.semantic_key() for group in self.groups}) != len(
            self.groups
        ):
            raise ValueError("Duplicate typed trajectory candidates are forbidden.")
        if len(self.groups) < self.policy.minimum_trajectory_groups:
            raise ValueError("Resolved plan has too few trajectory groups.")
        if not self.samples or len(self.samples) > self.policy.proposal_budget:
            raise ValueError(
                "Resolved sample count is empty or exceeds proposal_budget."
            )
        if tuple(sample.sample_index for sample in self.samples) != tuple(
            range(len(self.samples))
        ):
            raise ValueError("sample_index must cover 0..proposal_count-1 in order.")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id values must be unique.")
        by_group = {group.trajectory_group_id: group for group in self.groups}
        actual_groups = {sample.trajectory_group_id for sample in self.samples}
        if actual_groups != set(by_group):
            raise ValueError(
                "Samples must cover every and only resolved trajectory group."
            )
        variant_centers: dict[
            tuple[str, int],
            tuple[float, float, float],
        ] = {}
        frames_by_variant: dict[tuple[str, str], list[int]] = defaultdict(list)
        for sample in self.samples:
            group = by_group[sample.trajectory_group_id]
            if sample.trajectory_id != group.trajectory.trajectory_id:
                raise ValueError("Sample trajectory_id disagrees with its group.")
            if sample.view_id not in {view.view_id for view in group.views}:
                raise ValueError("Sample references an unknown typed view.")
            if sample.split is not group.split or sample.shard_id != group.shard_id:
                raise ValueError(
                    "Sample split/shard disagrees with its trajectory group."
                )
            if sample.trajectory_frame_index >= group.sample_count:
                raise ValueError("Sample frame index exceeds the group sample count.")
            frames_by_variant[(sample.trajectory_group_id, sample.view_id)].append(
                sample.trajectory_frame_index
            )
            path_key = (sample.trajectory_group_id, sample.trajectory_frame_index)
            previous = variant_centers.setdefault(
                path_key, sample.camera_center_scene_m
            )
            if not np.allclose(
                previous,
                sample.camera_center_scene_m,
                atol=1.0e-9,
                rtol=0.0,
            ):
                raise ValueError(
                    "Target variants in one trajectory group changed camera-centre path."
                )
        for group in self.groups:
            expected = list(range(group.sample_count))
            for view in group.views:
                if (
                    frames_by_variant[(group.trajectory_group_id, view.view_id)]
                    != expected
                ):
                    raise ValueError(
                        "Each target variant must cover its complete camera path."
                    )
        split_by_group: dict[str, set[DatasetSplit]] = defaultdict(set)
        for sample in self.samples:
            split_by_group[sample.trajectory_group_id].add(sample.split)
        leaking = [
            group_id for group_id, splits in split_by_group.items() if len(splits) != 1
        ]
        if leaking:
            raise ValueError(f"Trajectory group split leakage: {sorted(leaking)}.")
        if max(group.maximum_adjacent_step_m for group in self.groups) > (
            self.policy.max_arc_step_m + 1.0e-9
        ):
            raise ValueError("Resolved plan exceeds max_arc_step_m.")

    @property
    def proposal_count(self) -> int:
        """Return the renderer proposal count derived from resolved samples."""
        return len(self.samples)

    @property
    def schema_version(self) -> CourtDatasetSchemaVersion:
        """Return the explicit version while preserving v1 serialization."""
        return CourtDatasetSchemaVersion.V1

    def to_dict(self) -> dict[str, object]:
        """Return the complete deterministic plan."""
        return {
            "schema": COURT_PLAN_SCHEMA,
            "scene_id": self.scene_id,
            "profile": self.profile,
            "policy": self.policy.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
            "samples": [sample.to_dict() for sample in self.samples],
        }


# Explicit aliases make v1 preservation visible at version-aware boundaries.
OrbitViewSpecV1 = OrbitViewSpec
TrajectoryGroupPlanV1 = TrajectoryGroupPlan
PlannedCourtSampleV1 = PlannedCourtSample
CourtDatasetPlanV1 = CourtDatasetPlan


@dataclass(frozen=True, slots=True)
class TrajectoryGroupPlanV2:
    """One v2 path group with policy, never a nullable v1-style binding."""

    trajectory: OrbitTrajectorySpec
    center: OrbitCenter
    views: tuple[OrbitViewSpecV2, ...]
    split: DatasetSplit
    shard_id: str
    target_court_policy: TargetCourtPolicyV2
    sample_count: int
    maximum_adjacent_step_m: float
    total_arc_length_m: float

    def __post_init__(self) -> None:
        if self.center.key() != (
            self.trajectory.center_kind,
            self.trajectory.center_court_instance_id,
        ):
            raise ValueError(
                "Trajectory centre disagrees with resolved centre authority."
            )
        expected_policy = (
            TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT
            if self.trajectory.center_kind is OrbitCenterKind.COURT
            else TargetCourtResolutionPolicy.NEAREST_CAMERA
        )
        if self.target_court_policy.mode is not expected_policy:
            raise ValueError("V2 target policy disagrees with trajectory centre kind.")
        if (
            self.target_court_policy.centre_court_instance_id
            != self.trajectory.center_court_instance_id
        ):
            raise ValueError("V2 target policy disagrees with trajectory centre court.")
        if not self.views:
            raise ValueError("Every trajectory group requires at least one view.")
        if len({view.view_id for view in self.views}) != len(self.views):
            raise ValueError("view_id values must be unique within a trajectory group.")
        if len({view.semantic_key() for view in self.views}) != len(self.views):
            raise ValueError("Duplicate typed view candidates are forbidden.")
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit.")
        _text(self.shard_id, name="shard_id")
        _integer(self.sample_count, name="sample_count", minimum=8)
        maximum_step = _finite(
            self.maximum_adjacent_step_m,
            name="maximum_adjacent_step_m",
        )
        arc_length = _finite(self.total_arc_length_m, name="total_arc_length_m")
        if maximum_step <= 0.0 or arc_length <= 0.0:
            raise ValueError("Arc diagnostics must be positive.")
        object.__setattr__(self, "maximum_adjacent_step_m", maximum_step)
        object.__setattr__(self, "total_arc_length_m", arc_length)

    @property
    def trajectory_group_id(self) -> str:
        """Return the opaque group ID carried by the trajectory."""
        return self.trajectory.trajectory_group_id

    def to_dict(self) -> dict[str, object]:
        """Return the exact v2 group metadata."""
        return {
            "trajectory": self.trajectory.to_dict(),
            "center": self.center.to_dict(),
            "views": [view.to_dict() for view in self.views],
            "split": self.split.value,
            "shard_id": self.shard_id,
            "target_court_policy": self.target_court_policy.to_dict(),
            "sample_count": self.sample_count,
            "maximum_adjacent_step_m": self.maximum_adjacent_step_m,
            "total_arc_length_m": self.total_arc_length_m,
        }


@dataclass(frozen=True, slots=True)
class PlannedCourtSampleV2:
    """One v2 renderer request with a resolved sample-owned target court."""

    sample_index: int
    sample_id: str
    trajectory_group_id: str
    trajectory_id: str
    view_id: str
    trajectory_frame_index: int
    split: DatasetSplit
    shard_id: str
    camera_center_scene_m: tuple[float, float, float]
    camera: SceneCamera
    target_court: ResolvedTargetCourtV2

    def __post_init__(self) -> None:
        _integer(self.sample_index, name="sample_index", minimum=0)
        for name, value in (
            ("sample_id", self.sample_id),
            ("trajectory_group_id", self.trajectory_group_id),
            ("trajectory_id", self.trajectory_id),
            ("view_id", self.view_id),
            ("shard_id", self.shard_id),
        ):
            _text(value, name=name)
        _integer(
            self.trajectory_frame_index,
            name="trajectory_frame_index",
            minimum=0,
        )
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit.")
        if not isinstance(self.target_court, ResolvedTargetCourtV2):
            raise TypeError("target_court must be a ResolvedTargetCourtV2.")
        center = _finite_vector(
            self.camera_center_scene_m,
            size=3,
            name="camera_center_scene_m",
        )
        actual = self.camera.camera_to_scene.matrix()[:3, 3]
        if not np.allclose(actual, center, atol=1.0e-9, rtol=0.0):
            raise ValueError("camera_center_scene_m disagrees with camera_to_scene.")
        if self.camera.camera_id != self.sample_id:
            raise ValueError("The NHT camera ID must equal the stable sample ID.")
        object.__setattr__(self, "camera_center_scene_m", center)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v2 sample plan record."""
        return {
            "sample_index": self.sample_index,
            "sample_id": self.sample_id,
            "trajectory_group_id": self.trajectory_group_id,
            "trajectory_id": self.trajectory_id,
            "view_id": self.view_id,
            "trajectory_frame_index": self.trajectory_frame_index,
            "split": self.split.value,
            "shard_id": self.shard_id,
            "camera_center_scene_m": list(self.camera_center_scene_m),
            "camera": self.camera.to_dict(),
            "target_court": self.target_court.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CourtDatasetPlanV2:
    """Resolved v2 plan with sample-level geometric target authority."""

    scene_id: str
    profile: str
    policy: OrbitSamplingPolicy
    groups: tuple[TrajectoryGroupPlanV2, ...]
    samples: tuple[PlannedCourtSampleV2, ...]

    def __post_init__(self) -> None:
        _text(self.scene_id, name="scene_id")
        _text(self.profile, name="profile")
        if not self.groups:
            raise ValueError("Court plan must contain trajectory groups.")
        group_ids = [group.trajectory_group_id for group in self.groups]
        trajectory_ids = [group.trajectory.trajectory_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("trajectory_group_id values must be unique.")
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("trajectory_id values must be unique.")
        if len({group.trajectory.semantic_key() for group in self.groups}) != len(
            self.groups
        ):
            raise ValueError("Duplicate typed trajectory candidates are forbidden.")
        if len(self.groups) < self.policy.minimum_trajectory_groups:
            raise ValueError("Resolved plan has too few trajectory groups.")
        if not self.samples or len(self.samples) > self.policy.proposal_budget:
            raise ValueError(
                "Resolved sample count is empty or exceeds proposal_budget."
            )
        if tuple(sample.sample_index for sample in self.samples) != tuple(
            range(len(self.samples))
        ):
            raise ValueError("sample_index must cover 0..proposal_count-1 in order.")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id values must be unique.")
        by_group = {group.trajectory_group_id: group for group in self.groups}
        if {sample.trajectory_group_id for sample in self.samples} != set(by_group):
            raise ValueError(
                "Samples must cover every and only resolved trajectory group."
            )
        variant_centers: dict[
            tuple[str, int],
            tuple[float, float, float],
        ] = {}
        frames_by_variant: dict[tuple[str, str], list[int]] = defaultdict(list)
        for sample in self.samples:
            group = by_group[sample.trajectory_group_id]
            if sample.trajectory_id != group.trajectory.trajectory_id:
                raise ValueError("Sample trajectory_id disagrees with its group.")
            if sample.view_id not in {view.view_id for view in group.views}:
                raise ValueError("Sample references an unknown typed view.")
            if sample.split is not group.split or sample.shard_id != group.shard_id:
                raise ValueError(
                    "Sample split/shard disagrees with its trajectory group."
                )
            if sample.trajectory_frame_index >= group.sample_count:
                raise ValueError("Sample frame index exceeds the group sample count.")
            if (
                sample.target_court.resolution_policy
                is not group.target_court_policy.mode
            ):
                raise ValueError(
                    "Sample target policy disagrees with its trajectory group."
                )
            if (
                group.target_court_policy.mode
                is TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT
                and sample.target_court.binding.court_instance_id
                != group.target_court_policy.centre_court_instance_id
            ):
                raise ValueError(
                    "Court-centred sample target differs from the group centre court."
                )
            frames_by_variant[(sample.trajectory_group_id, sample.view_id)].append(
                sample.trajectory_frame_index
            )
            path_key = (
                sample.trajectory_group_id,
                sample.trajectory_frame_index,
            )
            previous = variant_centers.setdefault(
                path_key,
                sample.camera_center_scene_m,
            )
            if not np.allclose(
                previous,
                sample.camera_center_scene_m,
                atol=1.0e-9,
                rtol=0.0,
            ):
                raise ValueError(
                    "Target variants in one trajectory group changed camera-centre path."
                )
        for group in self.groups:
            expected = list(range(group.sample_count))
            for view in group.views:
                if (
                    frames_by_variant[(group.trajectory_group_id, view.view_id)]
                    != expected
                ):
                    raise ValueError(
                        "Each target variant must cover its complete camera path."
                    )
        split_by_group: dict[str, set[DatasetSplit]] = defaultdict(set)
        for sample in self.samples:
            split_by_group[sample.trajectory_group_id].add(sample.split)
        leaking = [
            group_id for group_id, splits in split_by_group.items() if len(splits) != 1
        ]
        if leaking:
            raise ValueError(f"Trajectory group split leakage: {sorted(leaking)}.")
        if max(group.maximum_adjacent_step_m for group in self.groups) > (
            self.policy.max_arc_step_m + 1.0e-9
        ):
            raise ValueError("Resolved plan exceeds max_arc_step_m.")

    @property
    def proposal_count(self) -> int:
        """Return the renderer proposal count derived from resolved samples."""
        return len(self.samples)

    @property
    def schema_version(self) -> CourtDatasetSchemaVersion:
        """Return the explicit plan version."""
        return CourtDatasetSchemaVersion.V2

    def to_dict(self) -> dict[str, object]:
        """Return the complete deterministic v2 plan."""
        return {
            "schema": COURT_PLAN_SCHEMA_V2,
            "scene_id": self.scene_id,
            "profile": self.profile,
            "policy": self.policy.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class CourtDatasetPlanV3(CourtDatasetPlanV2):
    """V3 plan with V2 target resolution and an exact revised artifact identity."""

    @property
    def schema_version(self) -> CourtDatasetSchemaVersion:
        """Return the corrected camera-view plan version."""
        return CourtDatasetSchemaVersion.V3

    def to_dict(self) -> dict[str, object]:
        """Return the complete deterministic V3 plan."""
        return {
            "schema": COURT_PLAN_SCHEMA_V3,
            "scene_id": self.scene_id,
            "profile": self.profile,
            "policy": self.policy.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
            "samples": [sample.to_dict() for sample in self.samples],
        }


TrajectoryGroupPlanAny: TypeAlias = TrajectoryGroupPlan | TrajectoryGroupPlanV2
PlannedCourtSampleAny: TypeAlias = PlannedCourtSample | PlannedCourtSampleV2
CourtDatasetPlanAny: TypeAlias = (
    CourtDatasetPlan | CourtDatasetPlanV2 | CourtDatasetPlanV3
)


__all__ = [
    "COURT_DATASET_SCHEMA",
    "COURT_DATASET_SCHEMA_V1",
    "COURT_DATASET_SCHEMA_V2",
    "COURT_DATASET_SCHEMA_V3",
    "COURT_PLAN_SCHEMA",
    "COURT_PLAN_SCHEMA_V1",
    "COURT_PLAN_SCHEMA_V2",
    "COURT_PLAN_SCHEMA_V3",
    "COURT_SAMPLE_SCHEMA",
    "COURT_SAMPLE_SCHEMA_V1",
    "COURT_SAMPLE_SCHEMA_V2",
    "COURT_SAMPLE_SCHEMA_V3",
    "CourtDatasetPlan",
    "CourtDatasetPlanAny",
    "CourtDatasetPlanV1",
    "CourtDatasetPlanV2",
    "CourtDatasetPlanV3",
    "DatasetSplit",
    "OrbitCenter",
    "OrbitCenterKind",
    "OrbitCoverageObjective",
    "OrbitCoverageMode",
    "OrbitCurveMode",
    "OrbitPathSamples",
    "OrbitSamplingMode",
    "OrbitSamplingPolicy",
    "OrbitShape",
    "OrbitStableField",
    "OrbitTargetKind",
    "OrbitTargetMode",
    "OrbitTrajectorySpec",
    "OrbitViewSpec",
    "OrbitViewSpecV1",
    "OrbitViewSpecV2",
    "PlannedCourtSample",
    "PlannedCourtSampleAny",
    "PlannedCourtSampleV1",
    "PlannedCourtSampleV2",
    "ResolvedTargetCourtV2",
    "TargetCourtPolicyV2",
    "TargetCourtResolutionPolicy",
    "TrajectoryGroupPlan",
    "TrajectoryGroupPlanAny",
    "TrajectoryGroupPlanV1",
    "TrajectoryGroupPlanV2",
]

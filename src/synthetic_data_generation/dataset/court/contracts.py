"""Strict semantic contracts for the canonical Court dataset stage."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.occupancy_artifact import (
    CourtV4SupportOccupancySnapshot,
)
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_DATASET_SCHEMA_V1,
    COURT_DATASET_SCHEMA_V2,
    COURT_DATASET_SCHEMA_V3,
    COURT_DATASET_SCHEMA_V4,
    COURT_PLAN_SCHEMA_V1,
    COURT_PLAN_SCHEMA_V2,
    COURT_PLAN_SCHEMA_V3,
    COURT_PLAN_SCHEMA_V4,
    COURT_SAMPLE_SCHEMA_V1,
    COURT_SAMPLE_SCHEMA_V2,
    COURT_SAMPLE_SCHEMA_V3,
    COURT_SAMPLE_SCHEMA_V4,
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera

if TYPE_CHECKING:
    from src.synthetic_data_generation.configuration import CourtSamplingPolicy

EnumT = TypeVar("EnumT", bound=StrEnum)

# Legacy public names remain exact v1 authorities.  Version-aware boundaries
# use the suffixed constants rather than changing these values in place.
COURT_DATASET_SCHEMA = COURT_DATASET_SCHEMA_V1
COURT_PLAN_SCHEMA = COURT_PLAN_SCHEMA_V1
COURT_SAMPLE_SCHEMA = COURT_SAMPLE_SCHEMA_V1


class OrbitShape(StrEnum):
    """Supported camera-centre curve shapes."""

    CIRCLE = "circle"
    ELLIPSE = "ellipse"


class PathFamilyV4(StrEnum):
    """Strict V4 path families kept outside the frozen legacy enum."""

    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    FREE_SPACE_CYCLE = "free_space_cycle"


class PathConstructorV4(StrEnum):
    """Strict provenance for how one V4 path was constructed."""

    ANALYTIC_GLOBAL = "analytic_global"
    FREE_SPACE_CYCLE = "free_space_cycle"
    ANCHORED_ROUNDED_RECTANGLE = "anchored_rounded_rectangle"


class OrbitCenterKind(StrEnum):
    """Typed centre authority for a camera-centre curve."""

    COMPLEX = "complex"
    COURT = "court"


class OrbitCurveMode(StrEnum):
    """Supported vertical trajectory modes."""

    PLANAR = "planar"
    SINUSOIDAL_HEIGHT = "sinusoidal_height"


class VerticalProfileV4(StrEnum):
    """Strict V4 vertical profiles kept outside the frozen legacy enum."""

    PLANAR = "planar"
    SINUSOIDAL_HEIGHT = "sinusoidal_height"
    RAISED_PHASES = "raised_phases"
    FREE_SPACE_CYCLE = "free_space_cycle"


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


class OrbitStableFieldV4(StrEnum):
    """V4 stable fields without changing the exact legacy vocabulary."""

    SHAPE = "shape"
    CENTER_KIND = "center_kind"
    RADIUS_SCALE = "radius_scale"
    AXIS_RATIO = "axis_ratio"
    ORIENTATION_DEGREES = "orientation_degrees"
    BASE_HEIGHT_M = "base_height_m"
    VERTICAL_MODULATION_M = "vertical_modulation_m"
    CURVE_MODE = "curve_mode"
    CORNER_RADIUS_RATIO = "corner_radius_ratio"
    VERTICAL_PHASE = "vertical_phase"
    CONSTRUCTOR = "constructor"


class OrbitCoverageObjective(StrEnum):
    """Ordered token families available to the greedy coverage selector."""

    COVERAGE_MODE = "coverage_mode"
    SEMANTIC_VISIBILITY = "semantic_visibility"
    TRAJECTORY_GROUP = "trajectory_group"


class TargetCourtResolutionPolicy(StrEnum):
    """Discriminant for the v2 group policy and resolved sample target."""

    TRAJECTORY_CENTER_COURT = "trajectory_center_court"
    NEAREST_CAMERA = "nearest_camera"


LEGACY_ORBIT_SHAPES = frozenset(OrbitShape)
LEGACY_ORBIT_CURVE_MODES = frozenset(OrbitCurveMode)
LEGACY_ORBIT_STABLE_FIELDS = frozenset(OrbitStableField)
V4_ORBIT_STABLE_FIELDS = frozenset(OrbitStableFieldV4)


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
class RequiredTrajectoryCoverage:
    """Typed V4 release minima, distinct from optional candidate diagnostics."""

    constructors: tuple[PathConstructorV4, ...]
    path_families: tuple[PathFamilyV4, ...]
    vertical_profiles: tuple[VerticalProfileV4, ...]
    target_modes: tuple[OrbitTargetMode, ...]
    minimum_total_groups: int
    minimum_free_space_cycle_groups: int
    minimum_anchored_rounded_rectangle_groups: int
    minimum_unique_anchors: int
    minimum_anchored_planar_groups: int
    minimum_anchored_raised_groups: int
    required_raised_lift_m: float
    minimum_anchored_frame_share: float

    def __post_init__(self) -> None:
        constructors = tuple(self.constructors)
        families = tuple(self.path_families)
        profiles = tuple(self.vertical_profiles)
        targets = tuple(self.target_modes)
        if constructors != (
            PathConstructorV4.FREE_SPACE_CYCLE,
            PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE,
        ):
            raise ValueError(
                "V4 required constructors must be free_space_cycle and "
                "anchored_rounded_rectangle in canonical order."
            )
        if families != (PathFamilyV4.ROUNDED_RECTANGLE,):
            raise ValueError(
                "V4 required path_families must be exactly [rounded_rectangle]."
            )
        if profiles != (
            VerticalProfileV4.PLANAR,
            VerticalProfileV4.RAISED_PHASES,
        ):
            raise ValueError(
                "V4 required vertical_profiles must be planar and raised_phases "
                "in canonical order."
            )
        if targets != (OrbitTargetMode.COURT_CENTER,):
            raise ValueError(
                "V4 required target_modes must equal the configured court_center target."
            )
        total = _integer(
            self.minimum_total_groups, name="minimum_total_groups", minimum=24
        )
        cycles = _integer(
            self.minimum_free_space_cycle_groups,
            name="minimum_free_space_cycle_groups",
            minimum=12,
        )
        anchored = _integer(
            self.minimum_anchored_rounded_rectangle_groups,
            name="minimum_anchored_rounded_rectangle_groups",
            minimum=6,
        )
        anchors = _integer(
            self.minimum_unique_anchors, name="minimum_unique_anchors", minimum=6
        )
        planar = _integer(
            self.minimum_anchored_planar_groups,
            name="minimum_anchored_planar_groups",
            minimum=3,
        )
        raised = _integer(
            self.minimum_anchored_raised_groups,
            name="minimum_anchored_raised_groups",
            minimum=3,
        )
        lift = _finite(self.required_raised_lift_m, name="required_raised_lift_m")
        share = _finite(
            self.minimum_anchored_frame_share,
            name="minimum_anchored_frame_share",
        )
        if anchors > anchored or planar + raised > anchored:
            raise ValueError("V4 anchored coverage minima are internally infeasible.")
        if cycles + anchored > total:
            raise ValueError("V4 constructor minima exceed minimum_total_groups.")
        if not math.isclose(lift, 0.25, abs_tol=1.0e-12, rel_tol=0.0):
            raise ValueError("V4 required_raised_lift_m must be exactly 0.25 m.")
        if not 0.08 <= share <= 1.0:
            raise ValueError(
                "V4 minimum_anchored_frame_share must lie within [0.08, 1]."
            )
        object.__setattr__(self, "constructors", constructors)
        object.__setattr__(self, "path_families", families)
        object.__setattr__(self, "vertical_profiles", profiles)
        object.__setattr__(self, "target_modes", targets)
        object.__setattr__(self, "minimum_total_groups", total)
        object.__setattr__(self, "minimum_free_space_cycle_groups", cycles)
        object.__setattr__(
            self, "minimum_anchored_rounded_rectangle_groups", anchored
        )
        object.__setattr__(self, "minimum_unique_anchors", anchors)
        object.__setattr__(self, "minimum_anchored_planar_groups", planar)
        object.__setattr__(self, "minimum_anchored_raised_groups", raised)
        object.__setattr__(self, "required_raised_lift_m", lift)
        object.__setattr__(self, "minimum_anchored_frame_share", share)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse the exact required-release coverage contract."""
        keys = {
            "constructors",
            "path_families",
            "vertical_profiles",
            "target_modes",
            "minimum_total_groups",
            "minimum_free_space_cycle_groups",
            "minimum_anchored_rounded_rectangle_groups",
            "minimum_unique_anchors",
            "minimum_anchored_planar_groups",
            "minimum_anchored_raised_groups",
            "required_raised_lift_m",
            "minimum_anchored_frame_share",
        }
        raw = _strict(value, keys=keys, name="required trajectory coverage")
        constructors = _enum_text_sequence(
            raw["constructors"],
            enum_type=PathConstructorV4,
            name="constructors",
        )
        families = _enum_text_sequence(
            raw["path_families"], enum_type=PathFamilyV4, name="path_families"
        )
        profiles = _enum_text_sequence(
            raw["vertical_profiles"],
            enum_type=VerticalProfileV4,
            name="vertical_profiles",
        )
        targets = _enum_text_sequence(
            raw["target_modes"], enum_type=OrbitTargetMode, name="target_modes"
        )
        return cls(
            constructors=tuple(PathConstructorV4(item) for item in constructors),
            path_families=tuple(PathFamilyV4(item) for item in families),
            vertical_profiles=tuple(VerticalProfileV4(item) for item in profiles),
            target_modes=tuple(OrbitTargetMode(item) for item in targets),
            minimum_total_groups=_integer(
                raw["minimum_total_groups"], name="minimum_total_groups", minimum=24
            ),
            minimum_free_space_cycle_groups=_integer(
                raw["minimum_free_space_cycle_groups"],
                name="minimum_free_space_cycle_groups",
                minimum=12,
            ),
            minimum_anchored_rounded_rectangle_groups=_integer(
                raw["minimum_anchored_rounded_rectangle_groups"],
                name="minimum_anchored_rounded_rectangle_groups",
                minimum=6,
            ),
            minimum_unique_anchors=_integer(
                raw["minimum_unique_anchors"],
                name="minimum_unique_anchors",
                minimum=6,
            ),
            minimum_anchored_planar_groups=_integer(
                raw["minimum_anchored_planar_groups"],
                name="minimum_anchored_planar_groups",
                minimum=3,
            ),
            minimum_anchored_raised_groups=_integer(
                raw["minimum_anchored_raised_groups"],
                name="minimum_anchored_raised_groups",
                minimum=3,
            ),
            required_raised_lift_m=_finite(
                raw["required_raised_lift_m"], name="required_raised_lift_m"
            ),
            minimum_anchored_frame_share=_finite(
                raw["minimum_anchored_frame_share"],
                name="minimum_anchored_frame_share",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize all release inventories and numeric minima."""
        return {
            "constructors": [item.value for item in self.constructors],
            "path_families": [item.value for item in self.path_families],
            "vertical_profiles": [item.value for item in self.vertical_profiles],
            "target_modes": [item.value for item in self.target_modes],
            "minimum_total_groups": self.minimum_total_groups,
            "minimum_free_space_cycle_groups": (
                self.minimum_free_space_cycle_groups
            ),
            "minimum_anchored_rounded_rectangle_groups": (
                self.minimum_anchored_rounded_rectangle_groups
            ),
            "minimum_unique_anchors": self.minimum_unique_anchors,
            "minimum_anchored_planar_groups": self.minimum_anchored_planar_groups,
            "minimum_anchored_raised_groups": self.minimum_anchored_raised_groups,
            "required_raised_lift_m": self.required_raised_lift_m,
            "minimum_anchored_frame_share": self.minimum_anchored_frame_share,
        }


@dataclass(frozen=True, slots=True)
class AnchoredRectangleProvenance:
    """Public-camera anchor plus persisted genuine rounded-rectangle geometry."""

    camera_inventory_digest: str
    camera_inventory_count: int
    ordered_camera_index: int
    camera_id: str
    source_frame_index: int
    anchor_center_scene_m: tuple[float, float, float]
    anchor_center_local_m: tuple[float, float, float]
    half_width_m: float
    half_height_m: float
    corner_radius_m: float
    orientation_radians: float
    vertical_profile: VerticalProfileV4
    lift_m: float
    reference_points_local_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        digest = _text(
            self.camera_inventory_digest, name="camera_inventory_digest"
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("camera_inventory_digest must be a lowercase SHA-256.")
        count = _integer(
            self.camera_inventory_count, name="camera_inventory_count", minimum=1
        )
        index = _integer(
            self.ordered_camera_index, name="ordered_camera_index", minimum=0
        )
        if index >= count:
            raise ValueError("ordered_camera_index exceeds camera_inventory_count.")
        _text(self.camera_id, name="camera_id")
        frame = _integer(
            self.source_frame_index, name="source_frame_index", minimum=0
        )
        scene_center = _finite_vector(
            self.anchor_center_scene_m, size=3, name="anchor_center_scene_m"
        )
        local_center = _finite_vector(
            self.anchor_center_local_m, size=3, name="anchor_center_local_m"
        )
        half_width = _finite(self.half_width_m, name="half_width_m")
        half_height = _finite(self.half_height_m, name="half_height_m")
        corner = _finite(self.corner_radius_m, name="corner_radius_m")
        orientation = _finite(
            self.orientation_radians, name="orientation_radians"
        )
        lift = _finite(self.lift_m, name="lift_m")
        if min(half_width, half_height) <= 0.0 or not 0.0 < corner < min(
            half_width, half_height
        ):
            raise ValueError("Anchored rounded-rectangle dimensions are invalid.")
        if self.vertical_profile not in (
            VerticalProfileV4.PLANAR,
            VerticalProfileV4.RAISED_PHASES,
        ):
            raise ValueError("Anchored rectangles require planar or raised_phases.")
        if self.vertical_profile is VerticalProfileV4.PLANAR and lift != 0.0:
            raise ValueError("Planar anchored rectangles require zero lift_m.")
        if self.vertical_profile is VerticalProfileV4.RAISED_PHASES and lift <= 0.0:
            raise ValueError("Raised anchored rectangles require positive lift_m.")
        points = tuple(
            _finite_vector(point, size=3, name="reference_points_local_m")
            for point in self.reference_points_local_m
        )
        if len(points) < 16 or any(
            math.dist(point, points[(point_index + 1) % len(points)]) <= 1.0e-9
            for point_index, point in enumerate(points)
        ):
            raise ValueError(
                "Anchored reference geometry requires at least 16 positive edges."
            )
        object.__setattr__(self, "camera_inventory_digest", digest)
        object.__setattr__(self, "camera_inventory_count", count)
        object.__setattr__(self, "ordered_camera_index", index)
        object.__setattr__(self, "source_frame_index", frame)
        object.__setattr__(self, "anchor_center_scene_m", scene_center)
        object.__setattr__(self, "anchor_center_local_m", local_center)
        object.__setattr__(self, "half_width_m", half_width)
        object.__setattr__(self, "half_height_m", half_height)
        object.__setattr__(self, "corner_radius_m", corner)
        object.__setattr__(self, "orientation_radians", orientation)
        object.__setattr__(self, "lift_m", lift)
        object.__setattr__(self, "reference_points_local_m", points)

    def to_dict(self) -> dict[str, object]:
        """Serialize anchor identity and the actual reference path points."""
        return {
            "camera_inventory_digest": self.camera_inventory_digest,
            "camera_inventory_count": self.camera_inventory_count,
            "ordered_camera_index": self.ordered_camera_index,
            "camera_id": self.camera_id,
            "source_frame_index": self.source_frame_index,
            "anchor_center_scene_m": list(self.anchor_center_scene_m),
            "anchor_center_local_m": list(self.anchor_center_local_m),
            "half_width_m": self.half_width_m,
            "half_height_m": self.half_height_m,
            "corner_radius_m": self.corner_radius_m,
            "orientation_radians": self.orientation_radians,
            "vertical_profile": self.vertical_profile.value,
            "lift_m": self.lift_m,
            "reference_points_local_m": [
                list(point) for point in self.reference_points_local_m
            ],
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse the exact anchor provenance shape."""
        keys = {
            "camera_inventory_digest",
            "camera_inventory_count",
            "ordered_camera_index",
            "camera_id",
            "source_frame_index",
            "anchor_center_scene_m",
            "anchor_center_local_m",
            "half_width_m",
            "half_height_m",
            "corner_radius_m",
            "orientation_radians",
            "vertical_profile",
            "lift_m",
            "reference_points_local_m",
        }
        raw = _strict(value, keys=keys, name="anchored rectangle provenance")
        scene = _finite_vector(
            _required_sequence(
                raw["anchor_center_scene_m"], name="anchor_center_scene_m"
            ),
            size=3,
            name="anchor_center_scene_m",
        )
        local = _finite_vector(
            _required_sequence(
                raw["anchor_center_local_m"], name="anchor_center_local_m"
            ),
            size=3,
            name="anchor_center_local_m",
        )
        points = tuple(
            _finite_vector(
                _required_sequence(point, name="reference_points_local_m"),
                size=3,
                name="reference_points_local_m",
            )
            for point in _required_sequence(
                raw["reference_points_local_m"], name="reference_points_local_m"
            )
        )
        return cls(
            camera_inventory_digest=_text(
                raw["camera_inventory_digest"], name="camera_inventory_digest"
            ),
            camera_inventory_count=_integer(
                raw["camera_inventory_count"],
                name="camera_inventory_count",
                minimum=1,
            ),
            ordered_camera_index=_integer(
                raw["ordered_camera_index"], name="ordered_camera_index", minimum=0
            ),
            camera_id=_text(raw["camera_id"], name="camera_id"),
            source_frame_index=_integer(
                raw["source_frame_index"], name="source_frame_index", minimum=0
            ),
            anchor_center_scene_m=(scene[0], scene[1], scene[2]),
            anchor_center_local_m=(local[0], local[1], local[2]),
            half_width_m=_finite(raw["half_width_m"], name="half_width_m"),
            half_height_m=_finite(raw["half_height_m"], name="half_height_m"),
            corner_radius_m=_finite(
                raw["corner_radius_m"], name="corner_radius_m"
            ),
            orientation_radians=_finite(
                raw["orientation_radians"], name="orientation_radians"
            ),
            vertical_profile=VerticalProfileV4(
                _text(raw["vertical_profile"], name="vertical_profile")
            ),
            lift_m=_finite(raw["lift_m"], name="lift_m"),
            reference_points_local_m=tuple(
                (point[0], point[1], point[2]) for point in points
            ),
        )


@dataclass(frozen=True, slots=True)
class SelectedTrajectoryCoverage:
    """Strict recomputed inventory for one selected V4 release plan."""

    total_group_count: int
    total_frame_count: int
    constructors: tuple[PathConstructorV4, ...]
    constructor_group_counts: tuple[tuple[PathConstructorV4, int], ...]
    constructor_frame_counts: tuple[tuple[PathConstructorV4, int], ...]
    path_families: tuple[PathFamilyV4, ...]
    family_group_counts: tuple[tuple[PathFamilyV4, int], ...]
    family_frame_counts: tuple[tuple[PathFamilyV4, int], ...]
    vertical_profiles: tuple[VerticalProfileV4, ...]
    profile_group_counts: tuple[tuple[VerticalProfileV4, int], ...]
    profile_frame_counts: tuple[tuple[VerticalProfileV4, int], ...]
    target_modes: tuple[OrbitTargetMode, ...]
    target_group_counts: tuple[tuple[OrbitTargetMode, int], ...]
    target_frame_counts: tuple[tuple[OrbitTargetMode, int], ...]
    anchor_camera_indices: tuple[int, ...]
    anchor_camera_ids: tuple[str, ...]
    unique_anchor_count: int
    anchored_group_count: int
    anchored_frame_count: int
    anchored_frame_share: float
    anchored_planar_group_count: int
    anchored_raised_group_count: int
    anchored_required_lift_group_count: int

    def __post_init__(self) -> None:
        groups = _integer(
            self.total_group_count, name="total_group_count", minimum=1
        )
        frames = _integer(
            self.total_frame_count, name="total_frame_count", minimum=1
        )
        constructors = _validate_enum_count_inventory(
            self.constructors,
            self.constructor_group_counts,
            enum_type=PathConstructorV4,
            name="constructor_group_counts",
        )
        _validate_enum_count_inventory(
            self.constructors,
            self.constructor_frame_counts,
            enum_type=PathConstructorV4,
            name="constructor_frame_counts",
        )
        families = _validate_enum_count_inventory(
            self.path_families,
            self.family_group_counts,
            enum_type=PathFamilyV4,
            name="family_group_counts",
        )
        _validate_enum_count_inventory(
            self.path_families,
            self.family_frame_counts,
            enum_type=PathFamilyV4,
            name="family_frame_counts",
        )
        profiles = _validate_enum_count_inventory(
            self.vertical_profiles,
            self.profile_group_counts,
            enum_type=VerticalProfileV4,
            name="profile_group_counts",
        )
        _validate_enum_count_inventory(
            self.vertical_profiles,
            self.profile_frame_counts,
            enum_type=VerticalProfileV4,
            name="profile_frame_counts",
        )
        targets = _validate_enum_count_inventory(
            self.target_modes,
            self.target_group_counts,
            enum_type=OrbitTargetMode,
            name="target_group_counts",
        )
        _validate_enum_count_inventory(
            self.target_modes,
            self.target_frame_counts,
            enum_type=OrbitTargetMode,
            name="target_frame_counts",
        )
        if sum(count for _key, count in self.constructor_group_counts) != groups:
            raise ValueError("Constructor group counts do not sum to total_group_count.")
        if sum(count for _key, count in self.constructor_frame_counts) != frames:
            raise ValueError("Constructor frame counts do not sum to total_frame_count.")
        for inventory_name, counts in (
            ("family_group_counts", self.family_group_counts),
            ("profile_group_counts", self.profile_group_counts),
            ("target_group_counts", self.target_group_counts),
        ):
            if sum(count for _key, count in counts) != groups:
                raise ValueError(f"{inventory_name} does not sum to total_group_count.")
        for inventory_name, counts in (
            ("family_frame_counts", self.family_frame_counts),
            ("profile_frame_counts", self.profile_frame_counts),
            ("target_frame_counts", self.target_frame_counts),
        ):
            if sum(count for _key, count in counts) != frames:
                raise ValueError(f"{inventory_name} does not sum to total_frame_count.")
        anchor_indices = tuple(self.anchor_camera_indices)
        anchor_ids = tuple(self.anchor_camera_ids)
        if (
            anchor_indices != tuple(sorted(set(anchor_indices)))
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in anchor_indices
            )
            or len(anchor_ids) != len(anchor_indices)
            or len(set(anchor_ids)) != len(anchor_ids)
            or any(not isinstance(value, str) or not value for value in anchor_ids)
        ):
            raise ValueError("Selected anchor inventory is invalid.")
        unique = _integer(
            self.unique_anchor_count, name="unique_anchor_count", minimum=0
        )
        anchored_groups = _integer(
            self.anchored_group_count, name="anchored_group_count", minimum=0
        )
        anchored_frames = _integer(
            self.anchored_frame_count, name="anchored_frame_count", minimum=0
        )
        planar = _integer(
            self.anchored_planar_group_count,
            name="anchored_planar_group_count",
            minimum=0,
        )
        raised = _integer(
            self.anchored_raised_group_count,
            name="anchored_raised_group_count",
            minimum=0,
        )
        required_lift = _integer(
            self.anchored_required_lift_group_count,
            name="anchored_required_lift_group_count",
            minimum=0,
        )
        share = _finite(self.anchored_frame_share, name="anchored_frame_share")
        anchored_constructor_groups = dict(self.constructor_group_counts).get(
            PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, 0
        )
        anchored_constructor_frames = dict(self.constructor_frame_counts).get(
            PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, 0
        )
        if (
            unique != len(anchor_indices)
            or anchored_groups != anchored_constructor_groups
            or anchored_frames != anchored_constructor_frames
            or planar + raised != anchored_groups
            or required_lift > raised
            or not math.isclose(
                share,
                anchored_frames / frames,
                abs_tol=1.0e-12,
                rel_tol=0.0,
            )
        ):
            raise ValueError("Selected anchored coverage metrics are inconsistent.")
        object.__setattr__(self, "total_group_count", groups)
        object.__setattr__(self, "total_frame_count", frames)
        object.__setattr__(self, "constructors", constructors)
        object.__setattr__(self, "path_families", families)
        object.__setattr__(self, "vertical_profiles", profiles)
        object.__setattr__(self, "target_modes", targets)
        object.__setattr__(self, "anchor_camera_indices", anchor_indices)
        object.__setattr__(self, "anchor_camera_ids", anchor_ids)
        object.__setattr__(self, "unique_anchor_count", unique)
        object.__setattr__(self, "anchored_group_count", anchored_groups)
        object.__setattr__(self, "anchored_frame_count", anchored_frames)
        object.__setattr__(self, "anchored_frame_share", share)
        object.__setattr__(self, "anchored_planar_group_count", planar)
        object.__setattr__(self, "anchored_raised_group_count", raised)
        object.__setattr__(
            self, "anchored_required_lift_group_count", required_lift
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize selected inventories, counts, anchors, and frame shares."""
        return {
            "total_group_count": self.total_group_count,
            "total_frame_count": self.total_frame_count,
            "constructors": [item.value for item in self.constructors],
            "constructor_group_counts": _enum_counts_to_dict(
                self.constructor_group_counts
            ),
            "constructor_frame_counts": _enum_counts_to_dict(
                self.constructor_frame_counts
            ),
            "path_families": [item.value for item in self.path_families],
            "family_group_counts": _enum_counts_to_dict(self.family_group_counts),
            "family_frame_counts": _enum_counts_to_dict(self.family_frame_counts),
            "vertical_profiles": [item.value for item in self.vertical_profiles],
            "profile_group_counts": _enum_counts_to_dict(self.profile_group_counts),
            "profile_frame_counts": _enum_counts_to_dict(self.profile_frame_counts),
            "target_modes": [item.value for item in self.target_modes],
            "target_group_counts": _enum_counts_to_dict(self.target_group_counts),
            "target_frame_counts": _enum_counts_to_dict(self.target_frame_counts),
            "anchor_camera_indices": list(self.anchor_camera_indices),
            "anchor_camera_ids": list(self.anchor_camera_ids),
            "unique_anchor_count": self.unique_anchor_count,
            "anchored_group_count": self.anchored_group_count,
            "anchored_frame_count": self.anchored_frame_count,
            "anchored_frame_share": self.anchored_frame_share,
            "anchored_planar_group_count": self.anchored_planar_group_count,
            "anchored_raised_group_count": self.anchored_raised_group_count,
            "anchored_required_lift_group_count": (
                self.anchored_required_lift_group_count
            ),
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse the exact selected-coverage inventory."""
        keys = {
            "total_group_count",
            "total_frame_count",
            "constructors",
            "constructor_group_counts",
            "constructor_frame_counts",
            "path_families",
            "family_group_counts",
            "family_frame_counts",
            "vertical_profiles",
            "profile_group_counts",
            "profile_frame_counts",
            "target_modes",
            "target_group_counts",
            "target_frame_counts",
            "anchor_camera_indices",
            "anchor_camera_ids",
            "unique_anchor_count",
            "anchored_group_count",
            "anchored_frame_count",
            "anchored_frame_share",
            "anchored_planar_group_count",
            "anchored_raised_group_count",
            "anchored_required_lift_group_count",
        }
        raw = _strict(value, keys=keys, name="selected trajectory coverage")
        constructors = _parse_enum_inventory(
            raw["constructors"], PathConstructorV4, name="constructors"
        )
        families = _parse_enum_inventory(
            raw["path_families"], PathFamilyV4, name="path_families"
        )
        profiles = _parse_enum_inventory(
            raw["vertical_profiles"], VerticalProfileV4, name="vertical_profiles"
        )
        targets = _parse_enum_inventory(
            raw["target_modes"], OrbitTargetMode, name="target_modes"
        )
        return cls(
            total_group_count=_integer(
                raw["total_group_count"], name="total_group_count", minimum=1
            ),
            total_frame_count=_integer(
                raw["total_frame_count"], name="total_frame_count", minimum=1
            ),
            constructors=constructors,
            constructor_group_counts=_parse_enum_count_mapping(
                raw["constructor_group_counts"],
                PathConstructorV4,
                name="constructor_group_counts",
            ),
            constructor_frame_counts=_parse_enum_count_mapping(
                raw["constructor_frame_counts"],
                PathConstructorV4,
                name="constructor_frame_counts",
            ),
            path_families=families,
            family_group_counts=_parse_enum_count_mapping(
                raw["family_group_counts"],
                PathFamilyV4,
                name="family_group_counts",
            ),
            family_frame_counts=_parse_enum_count_mapping(
                raw["family_frame_counts"],
                PathFamilyV4,
                name="family_frame_counts",
            ),
            vertical_profiles=profiles,
            profile_group_counts=_parse_enum_count_mapping(
                raw["profile_group_counts"],
                VerticalProfileV4,
                name="profile_group_counts",
            ),
            profile_frame_counts=_parse_enum_count_mapping(
                raw["profile_frame_counts"],
                VerticalProfileV4,
                name="profile_frame_counts",
            ),
            target_modes=targets,
            target_group_counts=_parse_enum_count_mapping(
                raw["target_group_counts"],
                OrbitTargetMode,
                name="target_group_counts",
            ),
            target_frame_counts=_parse_enum_count_mapping(
                raw["target_frame_counts"],
                OrbitTargetMode,
                name="target_frame_counts",
            ),
            anchor_camera_indices=tuple(
                _integer(item, name="anchor_camera_indices", minimum=0)
                for item in _required_sequence(
                    raw["anchor_camera_indices"], name="anchor_camera_indices"
                )
            ),
            anchor_camera_ids=tuple(
                _text(item, name="anchor_camera_ids")
                for item in _required_sequence(
                    raw["anchor_camera_ids"], name="anchor_camera_ids"
                )
            ),
            unique_anchor_count=_integer(
                raw["unique_anchor_count"], name="unique_anchor_count", minimum=0
            ),
            anchored_group_count=_integer(
                raw["anchored_group_count"], name="anchored_group_count", minimum=0
            ),
            anchored_frame_count=_integer(
                raw["anchored_frame_count"], name="anchored_frame_count", minimum=0
            ),
            anchored_frame_share=_finite(
                raw["anchored_frame_share"], name="anchored_frame_share"
            ),
            anchored_planar_group_count=_integer(
                raw["anchored_planar_group_count"],
                name="anchored_planar_group_count",
                minimum=0,
            ),
            anchored_raised_group_count=_integer(
                raw["anchored_raised_group_count"],
                name="anchored_raised_group_count",
                minimum=0,
            ),
            anchored_required_lift_group_count=_integer(
                raw["anchored_required_lift_group_count"],
                name="anchored_required_lift_group_count",
                minimum=0,
            ),
        )


def _validate_enum_count_inventory(
    inventory: Sequence[EnumT],
    counts: Sequence[tuple[EnumT, int]],
    *,
    enum_type: type[EnumT],
    name: str,
) -> tuple[EnumT, ...]:
    values = tuple(inventory)
    pairs = tuple(counts)
    if (
        not values
        or any(not isinstance(item, enum_type) for item in values)
        or values != tuple(sorted(set(values), key=lambda item: item.value))
        or tuple(key for key, _count in pairs) != values
        or any(
            not isinstance(key, enum_type)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for key, count in pairs
        )
    ):
        raise ValueError(f"{name} must exactly cover its sorted typed inventory.")
    return values


def _enum_counts_to_dict(counts: Sequence[tuple[StrEnum, int]]) -> dict[str, int]:
    return {key.value: count for key, count in counts}


def _parse_enum_inventory(
    value: object,
    enum_type: type[EnumT],
    *,
    name: str,
) -> tuple[EnumT, ...]:
    texts = _enum_text_sequence(value, enum_type=enum_type, name=name)
    return tuple(enum_type(item) for item in texts)


def _parse_enum_count_mapping(
    value: object,
    enum_type: type[EnumT],
    *,
    name: str,
) -> tuple[tuple[EnumT, int], ...]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    result = tuple(
        sorted(
            (
                enum_type(_text(key, name=name)),
                _integer(count, name=name, minimum=1),
            )
            for key, count in value.items()
        )
    )
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def required_coverage_shortfall(
    required: RequiredTrajectoryCoverage,
    selected: SelectedTrajectoryCoverage,
) -> tuple[str, ...]:
    """Return the complete deterministic release shortfall inventory."""
    if not isinstance(required, RequiredTrajectoryCoverage) or not isinstance(
        selected, SelectedTrajectoryCoverage
    ):
        raise TypeError("Required/selected coverage contracts are invalid.")
    constructor_counts = dict(selected.constructor_group_counts)
    missing: list[str] = []
    missing.extend(
        f"constructor:{item.value}"
        for item in required.constructors
        if item not in selected.constructors
    )
    missing.extend(
        f"family:{item.value}"
        for item in required.path_families
        if item not in selected.path_families
    )
    missing.extend(
        f"profile:{item.value}"
        for item in required.vertical_profiles
        if item not in selected.vertical_profiles
    )
    missing.extend(
        f"target:{item.value}"
        for item in required.target_modes
        if item not in selected.target_modes
    )
    if selected.total_group_count < required.minimum_total_groups:
        missing.append("minimum_total_groups")
    if (
        constructor_counts.get(PathConstructorV4.FREE_SPACE_CYCLE, 0)
        < required.minimum_free_space_cycle_groups
    ):
        missing.append("minimum_free_space_cycle_groups")
    if (
        selected.anchored_group_count
        < required.minimum_anchored_rounded_rectangle_groups
    ):
        missing.append("minimum_anchored_rounded_rectangle_groups")
    if selected.unique_anchor_count < required.minimum_unique_anchors:
        missing.append("minimum_unique_anchors")
    if (
        selected.anchored_planar_group_count
        < required.minimum_anchored_planar_groups
    ):
        missing.append("minimum_anchored_planar_groups")
    if (
        selected.anchored_raised_group_count
        < required.minimum_anchored_raised_groups
    ):
        missing.append("minimum_anchored_raised_groups")
    if (
        selected.anchored_required_lift_group_count
        < required.minimum_anchored_raised_groups
    ):
        missing.append("required_raised_lift_m")
    if (
        selected.anchored_frame_share + 1.0e-12
        < required.minimum_anchored_frame_share
    ):
        missing.append("minimum_anchored_frame_share")
    return tuple(sorted(set(missing)))


@dataclass(frozen=True, slots=True)
class OrbitTrajectorySpec:
    """One typed camera-centre path, independent of view and sample policy."""

    trajectory_id: str
    trajectory_group_id: str
    shape: OrbitShape | PathFamilyV4
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
    curve_mode: OrbitCurveMode | VerticalProfileV4

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
class OrbitTrajectorySpecV4(OrbitTrajectorySpec):
    """Strict V4 closed-path parameters with explicit corner and vertical phases."""

    corner_radius_ratio: float | None
    vertical_phase_offsets_m: tuple[float, ...]
    control_points_local_m: tuple[tuple[float, float, float], ...] | None = None
    constructor: PathConstructorV4 = PathConstructorV4.ANALYTIC_GLOBAL
    anchor_provenance: AnchoredRectangleProvenance | None = None

    def __post_init__(self) -> None:
        _text(self.trajectory_id, name="trajectory_id")
        _text(self.trajectory_group_id, name="trajectory_group_id")
        if not isinstance(self.shape, PathFamilyV4):
            raise TypeError("V4 shape must be a PathFamilyV4.")
        if not isinstance(self.center_kind, OrbitCenterKind):
            raise TypeError("center_kind must be an OrbitCenterKind.")
        if not isinstance(self.curve_mode, VerticalProfileV4):
            raise TypeError("V4 curve_mode must be a VerticalProfileV4.")
        if not isinstance(self.constructor, PathConstructorV4):
            raise TypeError("V4 constructor must be a PathConstructorV4.")
        center_id = _optional_text(
            self.center_court_instance_id, name="center_court_instance_id"
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
        if base_radius <= 0.0 or radius_scale <= 0.0 or not 0.0 < axis_ratio <= 1.0:
            raise ValueError("V4 radius and axis ratio parameters are invalid.")
        is_free_space_cycle = self.shape is PathFamilyV4.FREE_SPACE_CYCLE
        if self.shape is PathFamilyV4.CIRCLE and axis_ratio != 1.0:
            raise ValueError("A V4 circle must have axis_ratio exactly 1.0.")
        if (
            self.shape in (PathFamilyV4.ELLIPSE, PathFamilyV4.ROUNDED_RECTANGLE)
            and axis_ratio > 0.8
            and self.constructor is not PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
        ):
            raise ValueError("V4 ellipse/rounded_rectangle requires axis_ratio <= 0.8.")
        if base_height <= 0.0 or amplitude < 0.0:
            raise ValueError("V4 base height/amplitude parameters are invalid.")
        if self.curve_mode is VerticalProfileV4.PLANAR and (
            amplitude != 0.0 or cycles != 0
        ):
            raise ValueError("V4 planar paths require zero amplitude and cycles.")
        if self.curve_mode is VerticalProfileV4.SINUSOIDAL_HEIGHT and (
            amplitude <= 0.0 or cycles <= 0
        ):
            raise ValueError(
                "V4 sinusoidal paths require positive amplitude and cycles."
            )
        if self.curve_mode is VerticalProfileV4.RAISED_PHASES and (
            amplitude <= 0.0 or cycles != 0
        ):
            raise ValueError(
                "V4 raised phases require positive amplitude and zero cycles."
            )
        if self.curve_mode is VerticalProfileV4.FREE_SPACE_CYCLE and cycles != 0:
            raise ValueError("V4 free-space cycles require zero vertical_cycles.")
        if is_free_space_cycle != (
            self.curve_mode is VerticalProfileV4.FREE_SPACE_CYCLE
        ):
            raise ValueError(
                "The free_space_cycle shape and vertical profile must be used together."
            )
        if is_free_space_cycle != (
            self.constructor is PathConstructorV4.FREE_SPACE_CYCLE
        ):
            raise ValueError(
                "The free_space_cycle family requires its matching constructor."
            )
        is_anchored = (
            self.constructor is PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
        )
        if is_anchored and self.shape is not PathFamilyV4.ROUNDED_RECTANGLE:
            raise ValueError(
                "anchored_rounded_rectangle requires rounded_rectangle shape."
            )
        if (
            self.constructor is PathConstructorV4.ANALYTIC_GLOBAL
            and is_free_space_cycle
        ):
            raise ValueError("Analytic-global paths cannot be free_space_cycle.")
        object.__setattr__(self, "base_radius_m", base_radius)
        object.__setattr__(self, "radius_scale", radius_scale)
        object.__setattr__(self, "axis_ratio", axis_ratio)
        object.__setattr__(self, "orientation_radians", orientation)
        object.__setattr__(self, "base_height_m", base_height)
        object.__setattr__(self, "vertical_amplitude_m", amplitude)
        object.__setattr__(self, "vertical_cycles", cycles)
        object.__setattr__(self, "vertical_phase_radians", phase)
        offsets = tuple(
            _finite(value, name="vertical_phase_offsets_m")
            for value in self.vertical_phase_offsets_m
        )
        if not offsets:
            raise ValueError("V4 vertical_phase_offsets_m must not be empty.")
        if self.shape is PathFamilyV4.ROUNDED_RECTANGLE:
            ratio = _finite(self.corner_radius_ratio, name="corner_radius_ratio")
            if not 0.0 < ratio < 1.0:
                raise ValueError(
                    "Rounded-rectangle corner_radius_ratio must lie in (0, 1)."
                )
            object.__setattr__(self, "corner_radius_ratio", ratio)
        elif self.corner_radius_ratio is not None:
            raise ValueError("Only rounded rectangles may define corner_radius_ratio.")
        if self.curve_mode is VerticalProfileV4.RAISED_PHASES:
            if len(offsets) < 4 or min(offsets) < 0.0 or max(offsets) <= 0.0:
                raise ValueError(
                    "Raised phases require at least four non-negative offsets with one lift."
                )
            if not math.isclose(
                self.vertical_amplitude_m,
                max(offsets),
                abs_tol=1.0e-12,
                rel_tol=0.0,
            ):
                raise ValueError(
                    "vertical_amplitude_m must equal the maximum phase offset."
                )
        elif offsets != (0.0,):
            raise ValueError(
                "Planar/sinusoidal V4 paths require phase offsets exactly [0.0]."
            )
        if not is_free_space_cycle and self.base_height_m + min(offsets) <= 0.0:
            raise ValueError("V4 vertical phases produce a non-positive camera height.")
        object.__setattr__(self, "vertical_phase_offsets_m", offsets)
        controls_raw = self.control_points_local_m
        if is_free_space_cycle:
            if controls_raw is None or len(controls_raw) < 8:
                raise ValueError(
                    "V4 free-space cycles require at least eight local control points."
                )
            controls = tuple(
                _finite_vector(point, size=3, name="control_points_local_m")
                for point in controls_raw
            )
            if any(
                math.dist(point, controls[(index + 1) % len(controls)]) <= 1.0e-9
                for index, point in enumerate(controls)
            ):
                raise ValueError(
                    "V4 free-space cycle control edges must have positive length."
                )
            if self.base_height_m + min(point[2] for point in controls) <= 0.0:
                raise ValueError(
                    "V4 free-space cycle controls produce a non-positive camera height."
                )
            object.__setattr__(self, "control_points_local_m", controls)
        elif controls_raw is not None:
            raise ValueError("Only free-space cycles may define control points.")
        anchor = self.anchor_provenance
        if is_anchored:
            if not isinstance(anchor, AnchoredRectangleProvenance):
                raise ValueError(
                    "anchored_rounded_rectangle requires typed anchor_provenance."
                )
            expected_ratio = anchor.half_height_m / anchor.half_width_m
            expected_corner_ratio = anchor.corner_radius_m / min(
                anchor.half_width_m, anchor.half_height_m
            )
            expected_lift = (
                self.vertical_amplitude_m
                if self.curve_mode is VerticalProfileV4.RAISED_PHASES
                else 0.0
            )
            if (
                self.center_kind is not OrbitCenterKind.COMPLEX
                or self.center_court_instance_id is not None
                or not math.isclose(
                    self.base_radius_m,
                    anchor.half_width_m,
                    abs_tol=1.0e-12,
                    rel_tol=0.0,
                )
                or self.radius_scale != 1.0
                or not math.isclose(
                    self.axis_ratio,
                    expected_ratio,
                    abs_tol=1.0e-12,
                    rel_tol=0.0,
                )
                or not math.isclose(
                    self.corner_radius_ratio or 0.0,
                    expected_corner_ratio,
                    abs_tol=1.0e-12,
                    rel_tol=0.0,
                )
                or not math.isclose(
                    self.orientation_radians,
                    anchor.orientation_radians,
                    abs_tol=1.0e-12,
                    rel_tol=0.0,
                )
                or not math.isclose(
                    self.base_height_m,
                    anchor.anchor_center_local_m[2],
                    abs_tol=1.0e-12,
                    rel_tol=0.0,
                )
                or self.curve_mode is not anchor.vertical_profile
                or not math.isclose(
                    expected_lift,
                    anchor.lift_m,
                    abs_tol=1.0e-12,
                    rel_tol=0.0,
                )
            ):
                raise ValueError(
                    "Anchored path dimensions/profile disagree with anchor provenance."
                )
        elif anchor is not None:
            raise ValueError(
                "Only anchored_rounded_rectangle may define anchor_provenance."
            )

    @property
    def corner_radius_m(self) -> float | None:
        """Return the explicit metric rounded-corner radius when applicable."""
        if self.corner_radius_ratio is None:
            return None
        return min(self.radius_x_m, self.radius_y_m) * self.corner_radius_ratio

    def semantic_key(self) -> tuple[object, ...]:
        """Return the complete V4 geometry identity without opaque IDs."""
        return (
            *super(OrbitTrajectorySpecV4, self).semantic_key(),
            self.corner_radius_ratio,
            self.vertical_phase_offsets_m,
            self.control_points_local_m,
            self.constructor,
            self.anchor_provenance,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact V4 path representation."""
        return {
            **super(OrbitTrajectorySpecV4, self).to_dict(),
            "constructor": self.constructor.value,
            "corner_radius_ratio": self.corner_radius_ratio,
            "vertical_phase_offsets_m": list(self.vertical_phase_offsets_m),
            "control_points_local_m": (
                None
                if self.control_points_local_m is None
                else [list(point) for point in self.control_points_local_m]
            ),
            "anchor_provenance": (
                None
                if self.anchor_provenance is None
                else self.anchor_provenance.to_dict()
            ),
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse only the exact V4 path shape."""
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
            "constructor",
            "corner_radius_ratio",
            "vertical_phase_offsets_m",
            "control_points_local_m",
            "anchor_provenance",
        }
        raw = _strict(value, keys=keys, name="V4 trajectory")
        offsets = _finite_vector(
            _required_sequence(
                raw["vertical_phase_offsets_m"], name="vertical_phase_offsets_m"
            ),
            size=len(
                _required_sequence(
                    raw["vertical_phase_offsets_m"], name="vertical_phase_offsets_m"
                )
            ),
            name="vertical_phase_offsets_m",
        )
        corner_raw = raw["corner_radius_ratio"]
        controls_raw = raw["control_points_local_m"]
        controls: tuple[tuple[float, float, float], ...] | None
        if controls_raw is None:
            controls = None
        else:
            parsed_controls = tuple(
                _finite_vector(
                    _required_sequence(point, name="control_points_local_m"),
                    size=3,
                    name="control_points_local_m",
                )
                for point in _required_sequence(
                    controls_raw, name="control_points_local_m"
                )
            )
            controls = tuple(
                (point[0], point[1], point[2]) for point in parsed_controls
            )
        return cls(
            trajectory_id=_text(raw["trajectory_id"], name="trajectory_id"),
            trajectory_group_id=_text(
                raw["trajectory_group_id"], name="trajectory_group_id"
            ),
            shape=PathFamilyV4(_text(raw["shape"], name="shape")),
            center_kind=OrbitCenterKind(_text(raw["center_kind"], name="center_kind")),
            center_court_instance_id=_optional_text(
                raw["center_court_instance_id"], name="center_court_instance_id"
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
            curve_mode=VerticalProfileV4(_text(raw["curve_mode"], name="curve_mode")),
            constructor=PathConstructorV4(
                _text(raw["constructor"], name="constructor")
            ),
            corner_radius_ratio=(
                None
                if corner_raw is None
                else _finite(corner_raw, name="corner_radius_ratio")
            ),
            vertical_phase_offsets_m=offsets,
            control_points_local_m=controls,
            anchor_provenance=(
                None
                if raw["anchor_provenance"] is None
                else AnchoredRectangleProvenance.from_mapping(
                    raw["anchor_provenance"]
                )
            ),
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


class TrajectorySafetyReason(StrEnum):
    """Stable fail-closed V4 trajectory rejection reasons."""

    MISSING_SUPPORT_CAPABILITY = "missing_support_capability"
    NONFINITE_SUPPORT_INPUT = "nonfinite_support_input"
    INSUFFICIENT_CAPTURED_CAMERAS = "insufficient_captured_cameras"
    INSUFFICIENT_PUBLIC_POINTS = "insufficient_public_points"
    EMPTY_SUPPORT_FREE_SPACE = "empty_support_free_space"
    POINT_OUTSIDE_SUPPORT = "point_outside_support"
    POINT_HITS_INFLATED_OBSTACLE = "point_hits_inflated_obstacle"
    SWEPT_SEGMENT_OUTSIDE_SUPPORT = "swept_segment_outside_support"
    SWEPT_SEGMENT_HITS_INFLATED_OBSTACLE = "swept_segment_hits_inflated_obstacle"
    SAFE_CANDIDATE_EXHAUSTION = "safe_candidate_exhaustion"


@dataclass(frozen=True, slots=True)
class TrajectorySupportPolicy:
    """Frozen V4 public-camera and public-point safety authority."""

    decision_id: str
    support_radius_m: float
    endpoint_radius_m: float
    maximum_camera_link_distance_m: float
    maximum_source_frame_gap: int
    occupancy_voxel_size_m: float
    minimum_points_per_voxel: int
    obstacle_inflation_m: float
    camera_ball_clearance_m: float
    camera_capsule_clearance_m: float
    sweep_step_m: float
    boundary_epsilon_m: float
    minimum_captured_cameras: int
    minimum_public_points: int
    maximum_capsule_index_cells: int
    maximum_occupancy_cells: int
    minimum_cycle_frame_span: int
    maximum_cycle_frame_span: int
    maximum_cycle_closure_distance_m: float
    maximum_constructive_cycle_count: int
    cycle_smoothing_distance_m: float

    def __post_init__(self) -> None:
        _text(self.decision_id, name="decision_id")
        numeric = {
            "support_radius_m": self.support_radius_m,
            "endpoint_radius_m": self.endpoint_radius_m,
            "maximum_camera_link_distance_m": self.maximum_camera_link_distance_m,
            "occupancy_voxel_size_m": self.occupancy_voxel_size_m,
            "obstacle_inflation_m": self.obstacle_inflation_m,
            "camera_ball_clearance_m": self.camera_ball_clearance_m,
            "camera_capsule_clearance_m": self.camera_capsule_clearance_m,
            "sweep_step_m": self.sweep_step_m,
            "boundary_epsilon_m": self.boundary_epsilon_m,
            "maximum_cycle_closure_distance_m": (self.maximum_cycle_closure_distance_m),
            "cycle_smoothing_distance_m": self.cycle_smoothing_distance_m,
        }
        resolved = {name: _finite(value, name=name) for name, value in numeric.items()}
        if min(resolved.values()) <= 0.0:
            raise ValueError(
                "V4 support distances and boundary epsilon must be positive."
            )
        if resolved["endpoint_radius_m"] > resolved["support_radius_m"]:
            raise ValueError("endpoint_radius_m must not exceed support_radius_m.")
        if (
            max(
                resolved["camera_ball_clearance_m"],
                resolved["camera_capsule_clearance_m"],
                resolved["cycle_smoothing_distance_m"],
            )
            >= resolved["endpoint_radius_m"]
        ):
            raise ValueError(
                "Camera carving and cycle smoothing must stay below endpoint_radius_m."
            )
        if resolved["maximum_cycle_closure_distance_m"] >= (
            2.0 * resolved["endpoint_radius_m"]
        ):
            raise ValueError(
                "maximum_cycle_closure_distance_m must stay below two endpoint radii."
            )
        if resolved["sweep_step_m"] > min(
            resolved["occupancy_voxel_size_m"],
            resolved["support_radius_m"] / 2.0,
            resolved["obstacle_inflation_m"],
        ):
            raise ValueError(
                "sweep_step_m must not exceed support, occupancy, or inflation resolution."
            )
        for name, minimum in (
            ("maximum_source_frame_gap", 1),
            ("minimum_points_per_voxel", 1),
            ("minimum_captured_cameras", 2),
            ("minimum_public_points", 1),
            ("maximum_capsule_index_cells", 1),
            ("maximum_occupancy_cells", 1),
            ("minimum_cycle_frame_span", 8),
            ("maximum_cycle_frame_span", 8),
            ("maximum_constructive_cycle_count", 24),
        ):
            integer_value = _integer(getattr(self, name), name=name, minimum=minimum)
            object.__setattr__(self, name, integer_value)
        if self.maximum_cycle_frame_span <= self.minimum_cycle_frame_span:
            raise ValueError(
                "maximum_cycle_frame_span must exceed minimum_cycle_frame_span."
            )
        for name, value in resolved.items():
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        """Return all frozen construction and verification parameters."""
        return {
            "decision_id": self.decision_id,
            "support_radius_m": self.support_radius_m,
            "endpoint_radius_m": self.endpoint_radius_m,
            "maximum_camera_link_distance_m": self.maximum_camera_link_distance_m,
            "maximum_source_frame_gap": self.maximum_source_frame_gap,
            "occupancy_voxel_size_m": self.occupancy_voxel_size_m,
            "minimum_points_per_voxel": self.minimum_points_per_voxel,
            "obstacle_inflation_m": self.obstacle_inflation_m,
            "camera_ball_clearance_m": self.camera_ball_clearance_m,
            "camera_capsule_clearance_m": self.camera_capsule_clearance_m,
            "sweep_step_m": self.sweep_step_m,
            "boundary_epsilon_m": self.boundary_epsilon_m,
            "minimum_captured_cameras": self.minimum_captured_cameras,
            "minimum_public_points": self.minimum_public_points,
            "maximum_capsule_index_cells": self.maximum_capsule_index_cells,
            "maximum_occupancy_cells": self.maximum_occupancy_cells,
            "minimum_cycle_frame_span": self.minimum_cycle_frame_span,
            "maximum_cycle_frame_span": self.maximum_cycle_frame_span,
            "maximum_cycle_closure_distance_m": (self.maximum_cycle_closure_distance_m),
            "maximum_constructive_cycle_count": (self.maximum_constructive_cycle_count),
            "cycle_smoothing_distance_m": self.cycle_smoothing_distance_m,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse the exact V4 safety policy without defaults."""
        keys = {
            "decision_id",
            "support_radius_m",
            "endpoint_radius_m",
            "maximum_camera_link_distance_m",
            "maximum_source_frame_gap",
            "occupancy_voxel_size_m",
            "minimum_points_per_voxel",
            "obstacle_inflation_m",
            "camera_ball_clearance_m",
            "camera_capsule_clearance_m",
            "sweep_step_m",
            "boundary_epsilon_m",
            "minimum_captured_cameras",
            "minimum_public_points",
            "maximum_capsule_index_cells",
            "maximum_occupancy_cells",
            "minimum_cycle_frame_span",
            "maximum_cycle_frame_span",
            "maximum_cycle_closure_distance_m",
            "maximum_constructive_cycle_count",
            "cycle_smoothing_distance_m",
        }
        raw = _strict(value, keys=keys, name="trajectory support policy")
        return cls(
            decision_id=_text(raw["decision_id"], name="decision_id"),
            support_radius_m=_finite(raw["support_radius_m"], name="support_radius_m"),
            endpoint_radius_m=_finite(
                raw["endpoint_radius_m"], name="endpoint_radius_m"
            ),
            maximum_camera_link_distance_m=_finite(
                raw["maximum_camera_link_distance_m"],
                name="maximum_camera_link_distance_m",
            ),
            maximum_source_frame_gap=_integer(
                raw["maximum_source_frame_gap"],
                name="maximum_source_frame_gap",
                minimum=1,
            ),
            occupancy_voxel_size_m=_finite(
                raw["occupancy_voxel_size_m"], name="occupancy_voxel_size_m"
            ),
            minimum_points_per_voxel=_integer(
                raw["minimum_points_per_voxel"],
                name="minimum_points_per_voxel",
                minimum=1,
            ),
            obstacle_inflation_m=_finite(
                raw["obstacle_inflation_m"], name="obstacle_inflation_m"
            ),
            camera_ball_clearance_m=_finite(
                raw["camera_ball_clearance_m"], name="camera_ball_clearance_m"
            ),
            camera_capsule_clearance_m=_finite(
                raw["camera_capsule_clearance_m"],
                name="camera_capsule_clearance_m",
            ),
            sweep_step_m=_finite(raw["sweep_step_m"], name="sweep_step_m"),
            boundary_epsilon_m=_finite(
                raw["boundary_epsilon_m"], name="boundary_epsilon_m"
            ),
            minimum_captured_cameras=_integer(
                raw["minimum_captured_cameras"],
                name="minimum_captured_cameras",
                minimum=2,
            ),
            minimum_public_points=_integer(
                raw["minimum_public_points"], name="minimum_public_points", minimum=1
            ),
            maximum_capsule_index_cells=_integer(
                raw["maximum_capsule_index_cells"],
                name="maximum_capsule_index_cells",
                minimum=1,
            ),
            maximum_occupancy_cells=_integer(
                raw["maximum_occupancy_cells"],
                name="maximum_occupancy_cells",
                minimum=1,
            ),
            minimum_cycle_frame_span=_integer(
                raw["minimum_cycle_frame_span"],
                name="minimum_cycle_frame_span",
                minimum=8,
            ),
            maximum_cycle_frame_span=_integer(
                raw["maximum_cycle_frame_span"],
                name="maximum_cycle_frame_span",
                minimum=8,
            ),
            maximum_cycle_closure_distance_m=_finite(
                raw["maximum_cycle_closure_distance_m"],
                name="maximum_cycle_closure_distance_m",
            ),
            maximum_constructive_cycle_count=_integer(
                raw["maximum_constructive_cycle_count"],
                name="maximum_constructive_cycle_count",
                minimum=24,
            ),
            cycle_smoothing_distance_m=_finite(
                raw["cycle_smoothing_distance_m"],
                name="cycle_smoothing_distance_m",
            ),
        )


@dataclass(frozen=True, slots=True)
class SupportModelSummary:
    """Bounded construction evidence for one immutable V4 support model."""

    input_digest: str
    coordinate_space: str
    captured_camera_count: int
    public_point_count: int
    density_qualified_voxel_count: int
    raw_inflated_occupancy_cell_count: int
    inflated_occupancy_cell_count: int
    camera_ball_carved_cell_count: int
    camera_capsule_carved_cell_count: int
    captured_camera_occupied_count: int
    endpoint_ball_count: int
    capsule_count: int
    skipped_gap_link_count: int
    skipped_obstacle_link_count: int
    capsule_index_cell_count: int

    def __post_init__(self) -> None:
        digest = _text(self.input_digest, name="input_digest")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("input_digest must be a lowercase SHA-256 value.")
        if self.coordinate_space != "metric_scene_metres":
            raise ValueError("Support model must use metric_scene_metres.")
        for name, minimum in (
            ("captured_camera_count", 2),
            ("public_point_count", 1),
            ("density_qualified_voxel_count", 1),
            ("raw_inflated_occupancy_cell_count", 1),
            ("inflated_occupancy_cell_count", 1),
            ("camera_ball_carved_cell_count", 0),
            ("camera_capsule_carved_cell_count", 0),
            ("captured_camera_occupied_count", 0),
            ("endpoint_ball_count", 2),
            ("capsule_count", 1),
            ("skipped_gap_link_count", 0),
            ("skipped_obstacle_link_count", 0),
            ("capsule_index_cell_count", 1),
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name=name, minimum=minimum)
            )
        if self.captured_camera_occupied_count != 0:
            raise ValueError(
                "Validated captured camera centres must not be residual occupancy."
            )
        carved = (
            self.camera_ball_carved_cell_count + self.camera_capsule_carved_cell_count
        )
        if self.raw_inflated_occupancy_cell_count - carved != (
            self.inflated_occupancy_cell_count
        ):
            raise ValueError("Support occupancy carving counts are inconsistent.")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_digest": self.input_digest,
            "coordinate_space": self.coordinate_space,
            "captured_camera_count": self.captured_camera_count,
            "public_point_count": self.public_point_count,
            "density_qualified_voxel_count": self.density_qualified_voxel_count,
            "raw_inflated_occupancy_cell_count": (
                self.raw_inflated_occupancy_cell_count
            ),
            "inflated_occupancy_cell_count": self.inflated_occupancy_cell_count,
            "camera_ball_carved_cell_count": self.camera_ball_carved_cell_count,
            "camera_capsule_carved_cell_count": (self.camera_capsule_carved_cell_count),
            "captured_camera_occupied_count": self.captured_camera_occupied_count,
            "endpoint_ball_count": self.endpoint_ball_count,
            "capsule_count": self.capsule_count,
            "skipped_gap_link_count": self.skipped_gap_link_count,
            "skipped_obstacle_link_count": self.skipped_obstacle_link_count,
            "capsule_index_cell_count": self.capsule_index_cell_count,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        keys = {
            "input_digest",
            "coordinate_space",
            "captured_camera_count",
            "public_point_count",
            "density_qualified_voxel_count",
            "raw_inflated_occupancy_cell_count",
            "inflated_occupancy_cell_count",
            "camera_ball_carved_cell_count",
            "camera_capsule_carved_cell_count",
            "captured_camera_occupied_count",
            "endpoint_ball_count",
            "capsule_count",
            "skipped_gap_link_count",
            "skipped_obstacle_link_count",
            "capsule_index_cell_count",
        }
        raw = _strict(value, keys=keys, name="support model summary")
        return cls(
            input_digest=_text(raw["input_digest"], name="input_digest"),
            coordinate_space=_text(raw["coordinate_space"], name="coordinate_space"),
            captured_camera_count=_integer(
                raw["captured_camera_count"], name="captured_camera_count", minimum=2
            ),
            public_point_count=_integer(
                raw["public_point_count"], name="public_point_count", minimum=1
            ),
            density_qualified_voxel_count=_integer(
                raw["density_qualified_voxel_count"],
                name="density_qualified_voxel_count",
                minimum=1,
            ),
            raw_inflated_occupancy_cell_count=_integer(
                raw["raw_inflated_occupancy_cell_count"],
                name="raw_inflated_occupancy_cell_count",
                minimum=1,
            ),
            inflated_occupancy_cell_count=_integer(
                raw["inflated_occupancy_cell_count"],
                name="inflated_occupancy_cell_count",
                minimum=1,
            ),
            camera_ball_carved_cell_count=_integer(
                raw["camera_ball_carved_cell_count"],
                name="camera_ball_carved_cell_count",
                minimum=0,
            ),
            camera_capsule_carved_cell_count=_integer(
                raw["camera_capsule_carved_cell_count"],
                name="camera_capsule_carved_cell_count",
                minimum=0,
            ),
            captured_camera_occupied_count=_integer(
                raw["captured_camera_occupied_count"],
                name="captured_camera_occupied_count",
                minimum=0,
            ),
            endpoint_ball_count=_integer(
                raw["endpoint_ball_count"], name="endpoint_ball_count", minimum=2
            ),
            capsule_count=_integer(
                raw["capsule_count"], name="capsule_count", minimum=1
            ),
            skipped_gap_link_count=_integer(
                raw["skipped_gap_link_count"], name="skipped_gap_link_count", minimum=0
            ),
            skipped_obstacle_link_count=_integer(
                raw["skipped_obstacle_link_count"],
                name="skipped_obstacle_link_count",
                minimum=0,
            ),
            capsule_index_cell_count=_integer(
                raw["capsule_index_cell_count"],
                name="capsule_index_cell_count",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class TrajectorySafetyEvaluation:
    """One deterministic decision over every point and closed swept edge."""

    trajectory_id: str
    trajectory_group_id: str
    support_input_digest: str
    safe: bool
    reasons: tuple[TrajectorySafetyReason, ...]
    path_point_count: int
    closed_segment_count: int
    swept_sample_count: int
    violating_point_indices: tuple[int, ...]
    violating_segment_indices: tuple[int, ...]
    minimum_support_margin_m: float
    minimum_obstacle_clearance_m: float

    def __post_init__(self) -> None:
        _text(self.trajectory_id, name="trajectory_id")
        _text(self.trajectory_group_id, name="trajectory_group_id")
        digest = _text(self.support_input_digest, name="support_input_digest")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("support_input_digest must be a lowercase SHA-256 value.")
        if not isinstance(self.safe, bool):
            raise TypeError("safe must be boolean.")
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, TrajectorySafetyReason) for reason in reasons):
            raise TypeError("reasons must contain TrajectorySafetyReason values.")
        canonical_reasons = tuple(
            reason for reason in TrajectorySafetyReason if reason in reasons
        )
        if reasons != canonical_reasons:
            raise ValueError("Safety reasons must be unique and ordered.")
        if self.safe == bool(reasons):
            raise ValueError(
                "A safe evaluation has no reasons; an unsafe one has reasons."
            )
        for name, minimum in (
            ("path_point_count", 8),
            ("closed_segment_count", 8),
            ("swept_sample_count", 8),
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name=name, minimum=minimum)
            )
        if self.closed_segment_count != self.path_point_count:
            raise ValueError("A closed path must evaluate exactly one edge per point.")
        for name in ("violating_point_indices", "violating_segment_indices"):
            values = tuple(getattr(self, name))
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in values
            ):
                raise TypeError(f"{name} must contain non-negative integers.")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique.")
            upper_bound = (
                self.path_point_count
                if name == "violating_point_indices"
                else self.closed_segment_count
            )
            if any(index >= upper_bound for index in values):
                raise ValueError(f"{name} contains an out-of-range index.")
            object.__setattr__(self, name, values)
        has_violations = bool(
            self.violating_point_indices or self.violating_segment_indices
        )
        if self.safe == has_violations:
            raise ValueError(
                "Safe evaluations have no violations; unsafe evaluations identify one."
            )
        object.__setattr__(
            self,
            "minimum_support_margin_m",
            _finite(self.minimum_support_margin_m, name="minimum_support_margin_m"),
        )
        object.__setattr__(
            self,
            "minimum_obstacle_clearance_m",
            _finite(
                self.minimum_obstacle_clearance_m, name="minimum_obstacle_clearance_m"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "trajectory_id": self.trajectory_id,
            "trajectory_group_id": self.trajectory_group_id,
            "support_input_digest": self.support_input_digest,
            "safe": self.safe,
            "reasons": [reason.value for reason in self.reasons],
            "path_point_count": self.path_point_count,
            "closed_segment_count": self.closed_segment_count,
            "swept_sample_count": self.swept_sample_count,
            "violating_point_indices": list(self.violating_point_indices),
            "violating_segment_indices": list(self.violating_segment_indices),
            "minimum_support_margin_m": self.minimum_support_margin_m,
            "minimum_obstacle_clearance_m": self.minimum_obstacle_clearance_m,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        keys = {
            "trajectory_id",
            "trajectory_group_id",
            "support_input_digest",
            "safe",
            "reasons",
            "path_point_count",
            "closed_segment_count",
            "swept_sample_count",
            "violating_point_indices",
            "violating_segment_indices",
            "minimum_support_margin_m",
            "minimum_obstacle_clearance_m",
        }
        raw = _strict(value, keys=keys, name="trajectory safety evaluation")
        if not isinstance(raw["safe"], bool):
            raise TypeError("safe must be boolean.")
        reason_values = _required_sequence(raw["reasons"], name="reasons")
        point_indices = _required_sequence(
            raw["violating_point_indices"], name="violating_point_indices"
        )
        segment_indices = _required_sequence(
            raw["violating_segment_indices"], name="violating_segment_indices"
        )
        return cls(
            trajectory_id=_text(raw["trajectory_id"], name="trajectory_id"),
            trajectory_group_id=_text(
                raw["trajectory_group_id"], name="trajectory_group_id"
            ),
            support_input_digest=_text(
                raw["support_input_digest"], name="support_input_digest"
            ),
            safe=raw["safe"],
            reasons=tuple(
                TrajectorySafetyReason(_text(reason, name="reason"))
                for reason in reason_values
            ),
            path_point_count=_integer(
                raw["path_point_count"], name="path_point_count", minimum=8
            ),
            closed_segment_count=_integer(
                raw["closed_segment_count"], name="closed_segment_count", minimum=8
            ),
            swept_sample_count=_integer(
                raw["swept_sample_count"], name="swept_sample_count", minimum=8
            ),
            violating_point_indices=tuple(
                _integer(index, name="violating_point_indices", minimum=0)
                for index in point_indices
            ),
            violating_segment_indices=tuple(
                _integer(index, name="violating_segment_indices", minimum=0)
                for index in segment_indices
            ),
            minimum_support_margin_m=_finite(
                raw["minimum_support_margin_m"], name="minimum_support_margin_m"
            ),
            minimum_obstacle_clearance_m=_finite(
                raw["minimum_obstacle_clearance_m"], name="minimum_obstacle_clearance_m"
            ),
        )


@dataclass(frozen=True, slots=True)
class TrajectorySemanticPhaseEvaluation:
    """One V4 path evaluated at one explicit semantic-view sampling phase."""

    trajectory_id: str
    trajectory_group_id: str
    phase_index: int
    phase_count: int
    view: OrbitViewSpecV2
    expected_frame_count: int
    expected_valid_frame_count: int
    semantically_viable: bool
    rejection_counts: tuple[tuple[str, int], ...]
    disposition_digest: str

    def __post_init__(self) -> None:
        _text(self.trajectory_id, name="trajectory_id")
        _text(self.trajectory_group_id, name="trajectory_group_id")
        phase_count = _integer(self.phase_count, name="phase_count", minimum=1)
        phase_index = _integer(self.phase_index, name="phase_index", minimum=0)
        if phase_index >= phase_count:
            raise ValueError("phase_index must be smaller than phase_count.")
        if not isinstance(self.view, OrbitViewSpecV2):
            raise TypeError("Semantic phase evaluation requires OrbitViewSpecV2.")
        frame_count = _integer(
            self.expected_frame_count,
            name="expected_frame_count",
            minimum=8,
        )
        valid_count = _integer(
            self.expected_valid_frame_count,
            name="expected_valid_frame_count",
            minimum=0,
        )
        if valid_count > frame_count:
            raise ValueError("expected_valid_frame_count exceeds expected_frame_count.")
        if not isinstance(self.semantically_viable, bool):
            raise TypeError("semantically_viable must be boolean.")
        if self.semantically_viable != (valid_count > 0):
            raise ValueError(
                "semantically_viable must exactly describe positive expected validity."
            )
        rejection_counts = tuple(self.rejection_counts)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or item[0] != item[0].strip()
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] <= 0
            for item in rejection_counts
        ):
            raise TypeError(
                "rejection_counts must contain non-empty reason/positive-count pairs."
            )
        if rejection_counts != tuple(sorted(rejection_counts)) or len(
            {reason for reason, _count in rejection_counts}
        ) != len(rejection_counts):
            raise ValueError("rejection_counts must be sorted with unique reasons.")
        if sum(count for _reason, count in rejection_counts) != (
            frame_count - valid_count
        ):
            raise ValueError(
                "Semantic phase rejection accounting does not partition its frames."
            )
        digest = _text(self.disposition_digest, name="disposition_digest")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("disposition_digest must be a lowercase SHA-256 value.")
        object.__setattr__(self, "phase_index", phase_index)
        object.__setattr__(self, "phase_count", phase_count)
        object.__setattr__(self, "expected_frame_count", frame_count)
        object.__setattr__(self, "expected_valid_frame_count", valid_count)
        object.__setattr__(self, "rejection_counts", rejection_counts)

    @property
    def expected_rejected_frame_count(self) -> int:
        """Return the exact projected pre-render rejection count."""
        return self.expected_frame_count - self.expected_valid_frame_count

    @property
    def expected_valid_fraction(self) -> float:
        """Return the projected accepted fraction for this candidate/phase pair."""
        return self.expected_valid_frame_count / self.expected_frame_count

    def to_dict(self) -> dict[str, object]:
        """Return the strict persisted semantic-phase authority."""
        return {
            "trajectory_id": self.trajectory_id,
            "trajectory_group_id": self.trajectory_group_id,
            "phase_index": self.phase_index,
            "phase_count": self.phase_count,
            "view": self.view.to_dict(),
            "expected_frame_count": self.expected_frame_count,
            "expected_valid_frame_count": self.expected_valid_frame_count,
            "semantically_viable": self.semantically_viable,
            "rejection_counts": [
                {"reason": reason, "count": count}
                for reason, count in self.rejection_counts
            ],
            "disposition_digest": self.disposition_digest,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse one exact semantic-phase record and reject accounting drift."""
        raw = _strict(
            value,
            keys={
                "trajectory_id",
                "trajectory_group_id",
                "phase_index",
                "phase_count",
                "view",
                "expected_frame_count",
                "expected_valid_frame_count",
                "semantically_viable",
                "rejection_counts",
                "disposition_digest",
            },
            name="trajectory semantic phase evaluation",
        )
        if not isinstance(raw["semantically_viable"], bool):
            raise TypeError("semantically_viable must be boolean.")
        rejection_counts: list[tuple[str, int]] = []
        for value_item in _required_sequence(
            raw["rejection_counts"], name="rejection_counts"
        ):
            item = _strict(
                value_item,
                keys={"reason", "count"},
                name="semantic phase rejection count",
            )
            rejection_counts.append(
                (
                    _text(item["reason"], name="reason"),
                    _integer(item["count"], name="count", minimum=1),
                )
            )
        return cls(
            trajectory_id=_text(raw["trajectory_id"], name="trajectory_id"),
            trajectory_group_id=_text(
                raw["trajectory_group_id"], name="trajectory_group_id"
            ),
            phase_index=_integer(raw["phase_index"], name="phase_index", minimum=0),
            phase_count=_integer(raw["phase_count"], name="phase_count", minimum=1),
            view=OrbitViewSpecV2.from_mapping(raw["view"]),
            expected_frame_count=_integer(
                raw["expected_frame_count"],
                name="expected_frame_count",
                minimum=8,
            ),
            expected_valid_frame_count=_integer(
                raw["expected_valid_frame_count"],
                name="expected_valid_frame_count",
                minimum=0,
            ),
            semantically_viable=raw["semantically_viable"],
            rejection_counts=tuple(rejection_counts),
            disposition_digest=_text(
                raw["disposition_digest"], name="disposition_digest"
            ),
        )


def semantic_phase_inventory_digest(
    evaluations: Sequence[TrajectorySemanticPhaseEvaluation],
) -> str:
    """Hash the canonical ordered semantic-phase inventory."""
    values = tuple(evaluations)
    if not values or any(
        not isinstance(item, TrajectorySemanticPhaseEvaluation) for item in values
    ):
        raise TypeError("Semantic-phase inventory must be non-empty and typed.")
    payload = json.dumps(
        [item.to_dict() for item in values],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class OrbitSamplingPolicy:
    """Typed arc-length, selection, split, shard, and quality policy."""

    mode: OrbitSamplingMode
    max_arc_step_m: float
    minimum_sample_count: int
    sample_count_multiple: int
    seed: int
    stable_field_order: tuple[OrbitStableField | OrbitStableFieldV4, ...]
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
                not isinstance(field, OrbitStableField | OrbitStableFieldV4)
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
        stable_values = _required_sequence(
            raw["stable_field_order"], name="stable_field_order"
        )
        stable_texts = tuple(
            _text(item, name="stable_field_order") for item in stable_values
        )
        if not stable_texts or len(stable_texts) != len(set(stable_texts)):
            raise ValueError("stable_field_order must be non-empty and unique.")
        stable_enum: type[OrbitStableField] | type[OrbitStableFieldV4]
        stable_enum = (
            OrbitStableFieldV4
            if set(stable_texts) == {field.value for field in OrbitStableFieldV4}
            else OrbitStableField
        )
        try:
            stable_fields = tuple(stable_enum(field) for field in stable_texts)
        except ValueError as error:
            raise ValueError("stable_field_order contains an unknown field.") from error
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
            stable_field_order=stable_fields,
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
            center_kind=OrbitCenterKind(_text(raw["center_kind"], name="center_kind")),
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


@dataclass(frozen=True, slots=True)
class TrajectoryGroupPlanV4(TrajectoryGroupPlanV2):
    """One selected V4 group bound to its complete swept-safety decision."""

    trajectory: OrbitTrajectorySpecV4
    safety_evaluation: TrajectorySafetyEvaluation
    semantic_phase_evaluation: TrajectorySemanticPhaseEvaluation

    def __post_init__(self) -> None:
        super(TrajectoryGroupPlanV4, self).__post_init__()
        if not isinstance(self.trajectory, OrbitTrajectorySpecV4):
            raise TypeError("V4 groups require OrbitTrajectorySpecV4.")
        if not isinstance(self.safety_evaluation, TrajectorySafetyEvaluation):
            raise TypeError("V4 groups require TrajectorySafetyEvaluation.")
        if not isinstance(
            self.semantic_phase_evaluation, TrajectorySemanticPhaseEvaluation
        ):
            raise TypeError("V4 groups require TrajectorySemanticPhaseEvaluation.")
        if (
            not self.safety_evaluation.safe
            or self.safety_evaluation.trajectory_id != self.trajectory.trajectory_id
            or self.safety_evaluation.trajectory_group_id
            != self.trajectory.trajectory_group_id
        ):
            raise ValueError("V4 selected group disagrees with its safe evaluation.")
        phase = self.semantic_phase_evaluation
        if (
            phase.trajectory_id != self.trajectory.trajectory_id
            or phase.trajectory_group_id != self.trajectory.trajectory_group_id
            or not phase.semantically_viable
            or phase.expected_frame_count != self.sample_count
            or self.views != (phase.view,)
        ):
            raise ValueError(
                "V4 selected group disagrees with its semantic-phase evaluation."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            **super(TrajectoryGroupPlanV4, self).to_dict(),
            "safety_evaluation": self.safety_evaluation.to_dict(),
            "semantic_phase_evaluation": self.semantic_phase_evaluation.to_dict(),
        }


def build_selected_trajectory_coverage(
    groups: Sequence[TrajectoryGroupPlanV4],
    *,
    required_raised_lift_m: float,
) -> SelectedTrajectoryCoverage:
    """Recompute every selected constructor/family/profile/target/anchor count."""
    values = tuple(groups)
    if not values or any(not isinstance(group, TrajectoryGroupPlanV4) for group in values):
        raise TypeError("Selected coverage requires non-empty typed V4 groups.")
    if any(len(group.views) != 1 for group in values):
        raise ValueError("V4 selected coverage requires one semantic view per group.")
    return build_selected_coverage_from_records(
        tuple(
            (group.trajectory, group.views[0].target_mode, group.sample_count)
            for group in values
        ),
        required_raised_lift_m=required_raised_lift_m,
    )


def build_selected_coverage_from_records(
    records: Sequence[tuple[OrbitTrajectorySpecV4, OrbitTargetMode, int]],
    *,
    required_raised_lift_m: float,
) -> SelectedTrajectoryCoverage:
    """Recompute coverage before groups exist and again from persisted groups."""
    values = tuple(records)
    if not values or any(
        not isinstance(trajectory, OrbitTrajectorySpecV4)
        or not isinstance(target, OrbitTargetMode)
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
        for trajectory, target, frame_count in values
    ):
        raise TypeError("Selected coverage records are invalid.")
    lift = _finite(required_raised_lift_m, name="required_raised_lift_m")
    constructor_groups: Counter[PathConstructorV4] = Counter()
    constructor_frames: Counter[PathConstructorV4] = Counter()
    family_groups: Counter[PathFamilyV4] = Counter()
    family_frames: Counter[PathFamilyV4] = Counter()
    profile_groups: Counter[VerticalProfileV4] = Counter()
    profile_frames: Counter[VerticalProfileV4] = Counter()
    target_groups: Counter[OrbitTargetMode] = Counter()
    target_frames: Counter[OrbitTargetMode] = Counter()
    anchor_ids_by_index: dict[int, str] = {}
    anchor_inventory_authority: tuple[str, int] | None = None
    anchored_planar = 0
    anchored_raised = 0
    anchored_required_lift = 0
    for trajectory, target, frame_count in values:
        if not isinstance(trajectory.shape, PathFamilyV4) or not isinstance(
            trajectory.curve_mode, VerticalProfileV4
        ):
            raise TypeError("Selected V4 trajectory vocabulary is invalid.")
        constructor_groups[trajectory.constructor] += 1
        constructor_frames[trajectory.constructor] += frame_count
        family_groups[trajectory.shape] += 1
        family_frames[trajectory.shape] += frame_count
        profile_groups[trajectory.curve_mode] += 1
        profile_frames[trajectory.curve_mode] += frame_count
        target_groups[target] += 1
        target_frames[target] += frame_count
        if trajectory.constructor is not PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE:
            continue
        anchor = trajectory.anchor_provenance
        if anchor is None:  # pragma: no cover - strict trajectory excludes this
            raise ValueError("Selected anchored trajectory lacks provenance.")
        authority = (anchor.camera_inventory_digest, anchor.camera_inventory_count)
        if anchor_inventory_authority is None:
            anchor_inventory_authority = authority
        elif anchor_inventory_authority != authority:
            raise ValueError("Selected anchors mix public-camera inventory authorities.")
        previous = anchor_ids_by_index.setdefault(
            anchor.ordered_camera_index, anchor.camera_id
        )
        if previous != anchor.camera_id:
            raise ValueError("Selected anchor index maps to multiple camera IDs.")
        if trajectory.curve_mode is VerticalProfileV4.PLANAR:
            anchored_planar += 1
        elif trajectory.curve_mode is VerticalProfileV4.RAISED_PHASES:
            anchored_raised += 1
            if math.isclose(anchor.lift_m, lift, abs_tol=1.0e-12, rel_tol=0.0):
                anchored_required_lift += 1
    total_frames = sum(frame_count for _trajectory, _target, frame_count in values)
    anchored_groups = constructor_groups.get(
        PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, 0
    )
    anchored_frames = constructor_frames.get(
        PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, 0
    )
    anchor_indices = tuple(sorted(anchor_ids_by_index))
    return SelectedTrajectoryCoverage(
        total_group_count=len(values),
        total_frame_count=total_frames,
        constructors=tuple(sorted(constructor_groups, key=lambda item: item.value)),
        constructor_group_counts=tuple(
            sorted(constructor_groups.items(), key=lambda item: item[0].value)
        ),
        constructor_frame_counts=tuple(
            sorted(constructor_frames.items(), key=lambda item: item[0].value)
        ),
        path_families=tuple(sorted(family_groups, key=lambda item: item.value)),
        family_group_counts=tuple(
            sorted(family_groups.items(), key=lambda item: item[0].value)
        ),
        family_frame_counts=tuple(
            sorted(family_frames.items(), key=lambda item: item[0].value)
        ),
        vertical_profiles=tuple(sorted(profile_groups, key=lambda item: item.value)),
        profile_group_counts=tuple(
            sorted(profile_groups.items(), key=lambda item: item[0].value)
        ),
        profile_frame_counts=tuple(
            sorted(profile_frames.items(), key=lambda item: item[0].value)
        ),
        target_modes=tuple(sorted(target_groups, key=lambda item: item.value)),
        target_group_counts=tuple(
            sorted(target_groups.items(), key=lambda item: item[0].value)
        ),
        target_frame_counts=tuple(
            sorted(target_frames.items(), key=lambda item: item[0].value)
        ),
        anchor_camera_indices=anchor_indices,
        anchor_camera_ids=tuple(anchor_ids_by_index[index] for index in anchor_indices),
        unique_anchor_count=len(anchor_indices),
        anchored_group_count=anchored_groups,
        anchored_frame_count=anchored_frames,
        anchored_frame_share=anchored_frames / total_frames,
        anchored_planar_group_count=anchored_planar,
        anchored_raised_group_count=anchored_raised,
        anchored_required_lift_group_count=anchored_required_lift,
    )


@dataclass(frozen=True, slots=True)
class PlannedCourtSampleV4(PlannedCourtSampleV2):
    """V4 renderer request explicitly tied to one group safety authority."""

    safety_support_input_digest: str
    semantic_phase_index: int
    semantic_phase_disposition_digest: str

    def __post_init__(self) -> None:
        super(PlannedCourtSampleV4, self).__post_init__()
        digest = _text(
            self.safety_support_input_digest, name="safety_support_input_digest"
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "safety_support_input_digest must be a lowercase SHA-256 value."
            )
        phase_index = _integer(
            self.semantic_phase_index, name="semantic_phase_index", minimum=0
        )
        semantic_digest = _text(
            self.semantic_phase_disposition_digest,
            name="semantic_phase_disposition_digest",
        )
        if len(semantic_digest) != 64 or any(
            character not in "0123456789abcdef" for character in semantic_digest
        ):
            raise ValueError(
                "semantic_phase_disposition_digest must be a lowercase SHA-256 value."
            )
        object.__setattr__(self, "semantic_phase_index", phase_index)

    def to_dict(self) -> dict[str, object]:
        return {
            **super(PlannedCourtSampleV4, self).to_dict(),
            "safety_support_input_digest": self.safety_support_input_digest,
            "semantic_phase_index": self.semantic_phase_index,
            "semantic_phase_disposition_digest": (
                self.semantic_phase_disposition_digest
            ),
        }


@dataclass(frozen=True, slots=True)
class CourtDatasetPlanV4(CourtDatasetPlanV2):
    """Strict V4 plan carrying immutable support and complete candidate evidence."""

    groups: tuple[TrajectoryGroupPlanV4, ...]
    samples: tuple[PlannedCourtSampleV4, ...]
    support_policy: TrajectorySupportPolicy
    support_summary: SupportModelSummary
    support_occupancy_snapshot: CourtV4SupportOccupancySnapshot
    candidate_safety_evaluations: tuple[TrajectorySafetyEvaluation, ...]
    candidate_semantic_phase_evaluations: tuple[TrajectorySemanticPhaseEvaluation, ...]
    semantic_phase_inventory_digest: str
    required_coverage: RequiredTrajectoryCoverage
    selected_coverage: SelectedTrajectoryCoverage
    required_coverage_shortfall: tuple[str, ...]
    optional_candidate_coverage_shortfall: tuple[str, ...]

    def __post_init__(self) -> None:
        super(CourtDatasetPlanV4, self).__post_init__()
        if not isinstance(self.support_policy, TrajectorySupportPolicy):
            raise TypeError("V4 plan support_policy is invalid.")
        if not isinstance(self.support_summary, SupportModelSummary):
            raise TypeError("V4 plan support_summary is invalid.")
        if not isinstance(
            self.support_occupancy_snapshot,
            CourtV4SupportOccupancySnapshot,
        ):
            raise TypeError("V4 plan support_occupancy_snapshot is invalid.")
        occupancy = self.support_occupancy_snapshot
        if (
            occupancy.coordinate_space != self.support_summary.coordinate_space
            or occupancy.voxel_size_m != self.support_policy.occupancy_voxel_size_m
            or occupancy.support_input_digest != self.support_summary.input_digest
            or occupancy.policy_decision_id != self.support_policy.decision_id
            or occupancy.cell_count
            != self.support_summary.inflated_occupancy_cell_count
        ):
            raise ValueError(
                "V4 support occupancy snapshot disagrees with its support authority."
            )
        evaluations = tuple(self.candidate_safety_evaluations)
        if not evaluations or any(
            not isinstance(item, TrajectorySafetyEvaluation) for item in evaluations
        ):
            raise TypeError(
                "V4 candidate safety inventory must be non-empty and typed."
            )
        evaluation_by_group = {item.trajectory_group_id: item for item in evaluations}
        if len(evaluation_by_group) != len(evaluations) or len(
            {item.trajectory_id for item in evaluations}
        ) != len(evaluations):
            raise ValueError("V4 candidate safety IDs must be unique.")
        expected_id_pairs = tuple(
            (f"trajectory-{index:05d}", f"group-{index:05d}")
            for index in range(len(evaluations))
        )
        if (
            tuple(
                (item.trajectory_id, item.trajectory_group_id) for item in evaluations
            )
            != expected_id_pairs
        ):
            raise ValueError(
                "V4 candidate safety inventory must contain the exact canonical IDs."
            )
        if any(
            item.support_input_digest != self.support_summary.input_digest
            for item in evaluations
        ):
            raise ValueError("V4 candidate safety inventory mixes support authorities.")
        for group in self.groups:
            if (
                evaluation_by_group.get(group.trajectory_group_id)
                != group.safety_evaluation
                or group.safety_evaluation.trajectory_id
                != group.trajectory.trajectory_id
            ):
                raise ValueError(
                    "V4 selected group safety is absent from candidate inventory."
                )
        semantic_evaluations = tuple(self.candidate_semantic_phase_evaluations)
        if not semantic_evaluations or any(
            not isinstance(item, TrajectorySemanticPhaseEvaluation)
            for item in semantic_evaluations
        ):
            raise TypeError(
                "V4 candidate semantic-phase inventory must be non-empty and typed."
            )
        if semantic_evaluations != tuple(
            sorted(
                semantic_evaluations,
                key=lambda item: (item.trajectory_group_id, item.phase_index),
            )
        ):
            raise ValueError(
                "V4 candidate semantic-phase inventory must use canonical order."
            )
        safe_id_pairs = {
            (item.trajectory_id, item.trajectory_group_id)
            for item in evaluations
            if item.safe
        }
        semantic_by_pair: dict[
            tuple[str, str], list[TrajectorySemanticPhaseEvaluation]
        ] = defaultdict(list)
        for item in semantic_evaluations:
            semantic_by_pair[item.trajectory_id, item.trajectory_group_id].append(item)
        if set(semantic_by_pair) != safe_id_pairs:
            raise ValueError(
                "V4 semantic phases must cover every and only geometry-safe candidate."
            )
        for items in semantic_by_pair.values():
            phase_count = items[0].phase_count
            if any(item.phase_count != phase_count for item in items) or tuple(
                item.phase_index for item in items
            ) != tuple(range(phase_count)):
                raise ValueError(
                    "V4 safe candidates require one exact record for every semantic phase."
                )
        inventory_digest = _text(
            self.semantic_phase_inventory_digest,
            name="semantic_phase_inventory_digest",
        )
        if inventory_digest != semantic_phase_inventory_digest(semantic_evaluations):
            raise ValueError("V4 semantic-phase inventory digest disagrees.")
        semantic_set = set(semantic_evaluations)
        for group in self.groups:
            if group.semantic_phase_evaluation not in semantic_set:
                raise ValueError(
                    "V4 selected group semantic phase is absent from candidate inventory."
                )
        if any(
            sample.safety_support_input_digest != self.support_summary.input_digest
            for sample in self.samples
        ):
            raise ValueError("V4 samples mix support authorities.")
        phase_by_group = {
            group.trajectory_group_id: group.semantic_phase_evaluation
            for group in self.groups
        }
        if any(
            sample.semantic_phase_index
            != phase_by_group[sample.trajectory_group_id].phase_index
            or sample.semantic_phase_disposition_digest
            != phase_by_group[sample.trajectory_group_id].disposition_digest
            for sample in self.samples
        ):
            raise ValueError("V4 samples disagree with group semantic-phase authority.")
        expected_valid_count = sum(
            group.semantic_phase_evaluation.expected_valid_frame_count
            for group in self.groups
        )
        if expected_valid_count < self.policy.minimum_accepted_frames:
            raise ValueError(
                "V4 projected semantic validity is below minimum_accepted_frames."
            )
        if expected_valid_count / len(self.samples) < (
            self.policy.minimum_accepted_fraction
        ):
            raise ValueError(
                "V4 projected semantic validity is below minimum_accepted_fraction."
            )
        if not isinstance(self.required_coverage, RequiredTrajectoryCoverage):
            raise TypeError("V4 plan required_coverage is invalid.")
        if not isinstance(self.selected_coverage, SelectedTrajectoryCoverage):
            raise TypeError("V4 plan selected_coverage is invalid.")
        recomputed_coverage = build_selected_trajectory_coverage(
            self.groups,
            required_raised_lift_m=self.required_coverage.required_raised_lift_m,
        )
        if self.selected_coverage != recomputed_coverage:
            raise ValueError("V4 selected_coverage disagrees with selected groups.")
        required_shortfall = tuple(self.required_coverage_shortfall)
        recomputed_shortfall = required_coverage_shortfall(
            self.required_coverage,
            recomputed_coverage,
        )
        if required_shortfall != recomputed_shortfall or required_shortfall:
            raise ValueError(
                "V4 required_coverage_shortfall must be recomputed and empty."
            )
        if (
            self.required_coverage.minimum_total_groups
            < self.policy.minimum_trajectory_groups
            or recomputed_coverage.total_frame_count != len(self.samples)
        ):
            raise ValueError("V4 selected coverage disagrees with release policy/samples.")
        optional_shortfall = tuple(self.optional_candidate_coverage_shortfall)
        if optional_shortfall != tuple(sorted(set(optional_shortfall))) or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in optional_shortfall
        ):
            raise ValueError(
                "V4 optional_candidate_coverage_shortfall must be sorted unique strings."
            )
        object.__setattr__(self, "candidate_safety_evaluations", evaluations)
        object.__setattr__(
            self,
            "candidate_semantic_phase_evaluations",
            semantic_evaluations,
        )
        object.__setattr__(self, "semantic_phase_inventory_digest", inventory_digest)
        object.__setattr__(self, "selected_coverage", recomputed_coverage)
        object.__setattr__(
            self, "required_coverage_shortfall", required_shortfall
        )
        object.__setattr__(
            self,
            "optional_candidate_coverage_shortfall",
            optional_shortfall,
        )

    @property
    def projected_semantic_valid_frame_count(self) -> int:
        """Return the exact pre-render accepted count frozen at selection time."""
        return sum(
            group.semantic_phase_evaluation.expected_valid_frame_count
            for group in self.groups
        )

    @property
    def projected_semantic_valid_fraction(self) -> float:
        """Return the exact projected pre-render acceptance fraction."""
        return self.projected_semantic_valid_frame_count / self.proposal_count

    @property
    def schema_version(self) -> CourtDatasetSchemaVersion:
        return CourtDatasetSchemaVersion.V4

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": COURT_PLAN_SCHEMA_V4,
            "scene_id": self.scene_id,
            "profile": self.profile,
            "policy": self.policy.to_dict(),
            "support_policy": self.support_policy.to_dict(),
            "support_summary": self.support_summary.to_dict(),
            "support_occupancy_identity": (
                self.support_occupancy_snapshot.identity.to_dict()
            ),
            "candidate_safety_evaluations": [
                item.to_dict() for item in self.candidate_safety_evaluations
            ],
            "candidate_semantic_phase_evaluations": [
                item.to_dict() for item in self.candidate_semantic_phase_evaluations
            ],
            "semantic_phase_inventory_digest": self.semantic_phase_inventory_digest,
            "projected_semantic_valid_frame_count": (
                self.projected_semantic_valid_frame_count
            ),
            "projected_semantic_valid_fraction": (
                self.projected_semantic_valid_fraction
            ),
            "required_coverage": self.required_coverage.to_dict(),
            "selected_coverage": self.selected_coverage.to_dict(),
            "required_coverage_shortfall": list(
                self.required_coverage_shortfall
            ),
            "optional_candidate_coverage_shortfall": list(
                self.optional_candidate_coverage_shortfall
            ),
            "groups": [group.to_dict() for group in self.groups],
            "samples": [sample.to_dict() for sample in self.samples],
        }


TrajectoryGroupPlanAny: TypeAlias = (
    TrajectoryGroupPlan | TrajectoryGroupPlanV2 | TrajectoryGroupPlanV4
)
PlannedCourtSampleAny: TypeAlias = (
    PlannedCourtSample | PlannedCourtSampleV2 | PlannedCourtSampleV4
)
CourtDatasetPlanAny: TypeAlias = (
    CourtDatasetPlan | CourtDatasetPlanV2 | CourtDatasetPlanV3 | CourtDatasetPlanV4
)


__all__ = [
    "COURT_DATASET_SCHEMA",
    "COURT_DATASET_SCHEMA_V1",
    "COURT_DATASET_SCHEMA_V2",
    "COURT_DATASET_SCHEMA_V3",
    "COURT_DATASET_SCHEMA_V4",
    "COURT_PLAN_SCHEMA",
    "COURT_PLAN_SCHEMA_V1",
    "COURT_PLAN_SCHEMA_V2",
    "COURT_PLAN_SCHEMA_V3",
    "COURT_PLAN_SCHEMA_V4",
    "COURT_SAMPLE_SCHEMA",
    "COURT_SAMPLE_SCHEMA_V1",
    "COURT_SAMPLE_SCHEMA_V2",
    "COURT_SAMPLE_SCHEMA_V3",
    "COURT_SAMPLE_SCHEMA_V4",
    "CourtDatasetPlan",
    "CourtDatasetPlanAny",
    "CourtDatasetPlanV1",
    "CourtDatasetPlanV2",
    "CourtDatasetPlanV3",
    "CourtDatasetPlanV4",
    "AnchoredRectangleProvenance",
    "DatasetSplit",
    "OrbitCenter",
    "OrbitCenterKind",
    "OrbitCoverageObjective",
    "OrbitCoverageMode",
    "OrbitCurveMode",
    "OrbitPathSamples",
    "PathFamilyV4",
    "PathConstructorV4",
    "OrbitSamplingMode",
    "OrbitSamplingPolicy",
    "OrbitShape",
    "OrbitStableField",
    "OrbitStableFieldV4",
    "OrbitTargetKind",
    "OrbitTargetMode",
    "OrbitTrajectorySpec",
    "OrbitTrajectorySpecV4",
    "OrbitViewSpec",
    "OrbitViewSpecV1",
    "OrbitViewSpecV2",
    "PlannedCourtSample",
    "PlannedCourtSampleAny",
    "PlannedCourtSampleV1",
    "PlannedCourtSampleV2",
    "PlannedCourtSampleV4",
    "ResolvedTargetCourtV2",
    "RequiredTrajectoryCoverage",
    "SelectedTrajectoryCoverage",
    "TargetCourtPolicyV2",
    "TargetCourtResolutionPolicy",
    "TrajectorySemanticPhaseEvaluation",
    "TrajectorySafetyEvaluation",
    "TrajectorySafetyReason",
    "TrajectorySupportPolicy",
    "TrajectoryGroupPlan",
    "TrajectoryGroupPlanAny",
    "TrajectoryGroupPlanV1",
    "TrajectoryGroupPlanV2",
    "TrajectoryGroupPlanV4",
    "SupportModelSummary",
    "LEGACY_ORBIT_CURVE_MODES",
    "LEGACY_ORBIT_SHAPES",
    "LEGACY_ORBIT_STABLE_FIELDS",
    "V4_ORBIT_STABLE_FIELDS",
    "VerticalProfileV4",
    "build_selected_coverage_from_records",
    "build_selected_trajectory_coverage",
    "required_coverage_shortfall",
    "semantic_phase_inventory_digest",
]

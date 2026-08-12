"""Canonical AMASS/SMPL-H coordinate and initial-support contracts for PLCS."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

PLCS_COORDINATE_CONTRACT_SCHEMA = "plcs_amass_smplh_z_up_v1"
PLCS_SUPPORT_PLANE_SCHEMA = "plcs_initial_smplh_surface_support_v1"
PLCS_SUPPORT_PLACEMENT_TOLERANCE_M = 1.0e-5
SMPLH_SURFACE_VERTEX_COUNT = 6_890

_SURFACE_DEFINITION = (
    "frame-0 posed full SMPL-H surface after pose blend, LBS, and global_orient; "
    "before root translation; minimum local Z"
)


@dataclass(frozen=True, slots=True)
class PLCSCoordinateContract:
    """Exact source-to-court identity for AMASS-driven SMPL-H geometry.

    AMASS ``poses[:, :3]`` is the SMPL-H root ``global_orient`` and is already
    consumed by LBS. AMASS ``trans`` uses that same right-handed, Z-up, metre
    source frame. Court coordinates have the same handedness, up axis, and unit,
    so placement may add only the configured yaw about court +Z.
    """

    schema: str = PLCS_COORDINATE_CONTRACT_SCHEMA
    handedness: str = "right-handed"
    up_axis: str = "+Z"
    linear_unit: str = "metre"
    global_orient_application: str = "smplh_lbs"
    root_translation_frame: str = "amass_source_frame"
    court_orientation: str = "configured_positive_z_yaw_only"

    def __post_init__(self) -> None:
        if self.to_dict() != _coordinate_contract_payload():
            raise ValueError("PLCS coordinate contract fields must match the schema.")

    def to_dict(self) -> dict[str, object]:
        """Return the exact persisted v5 coordinate identity."""
        return {
            "schema": self.schema,
            "handedness": self.handedness,
            "up_axis": self.up_axis,
            "linear_unit": self.linear_unit,
            "global_orient_application": self.global_orient_application,
            "root_translation_frame": self.root_translation_frame,
            "court_orientation": self.court_orientation,
        }

    @classmethod
    def from_dict(cls, value: object) -> PLCSCoordinateContract:
        """Parse only the exact current coordinate contract."""
        record = _mapping(value, name="coordinate_contract")
        if dict(record) != _coordinate_contract_payload():
            raise ValueError("PLCS coordinate_contract does not match the v5 schema.")
        return cls()


@dataclass(frozen=True, slots=True)
class PLCSSourceSupportPlane:
    """Frame-zero full-surface support evidence used for one motion track."""

    source_frame_index: int
    vertex_count: int
    initial_root_translation_z_m: float
    support_local_z_m: float
    support_plane_source_z_m: float
    placement_tolerance_m: float = PLCS_SUPPORT_PLACEMENT_TOLERANCE_M

    schema: ClassVar[str] = PLCS_SUPPORT_PLANE_SCHEMA
    surface_definition: ClassVar[str] = _SURFACE_DEFINITION

    def __post_init__(self) -> None:
        if self.source_frame_index != 0:
            raise ValueError("PLCS support plane must be evaluated at source frame 0.")
        if self.vertex_count != SMPLH_SURFACE_VERTEX_COUNT:
            raise ValueError("PLCS support plane must use all 6,890 SMPL-H vertices.")
        values = (
            self.initial_root_translation_z_m,
            self.support_local_z_m,
            self.support_plane_source_z_m,
            self.placement_tolerance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("PLCS support-plane values must be finite.")
        if self.placement_tolerance_m != PLCS_SUPPORT_PLACEMENT_TOLERANCE_M:
            raise ValueError("PLCS support-plane tolerance must be exactly 1e-5 m.")
        expected = self.initial_root_translation_z_m + self.support_local_z_m
        if not math.isclose(
            self.support_plane_source_z_m,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "PLCS source support Z must equal initial trans.z plus local min Z."
            )

    @classmethod
    def from_surface_minimum(
        cls,
        *,
        initial_root_translation_z_m: float,
        support_local_z_m: float,
    ) -> PLCSSourceSupportPlane:
        """Bind one deterministic frame-zero CUDA full-surface minimum."""
        return cls(
            source_frame_index=0,
            vertex_count=SMPLH_SURFACE_VERTEX_COUNT,
            initial_root_translation_z_m=float(initial_root_translation_z_m),
            support_local_z_m=float(support_local_z_m),
            support_plane_source_z_m=(
                float(initial_root_translation_z_m) + float(support_local_z_m)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return complete support provenance for one v5 track."""
        return {
            "schema": self.schema,
            "source_frame_index": self.source_frame_index,
            "surface_definition": self.surface_definition,
            "vertex_count": self.vertex_count,
            "initial_root_translation_z_m": self.initial_root_translation_z_m,
            "support_local_z_m": self.support_local_z_m,
            "support_plane_source_z_m": self.support_plane_source_z_m,
            "placement_tolerance_m": self.placement_tolerance_m,
        }

    @classmethod
    def from_dict(cls, value: object) -> PLCSSourceSupportPlane:
        """Parse exact, non-optional full-surface support provenance."""
        record = _mapping(value, name="support_plane")
        expected_keys = {
            "schema",
            "source_frame_index",
            "surface_definition",
            "vertex_count",
            "initial_root_translation_z_m",
            "support_local_z_m",
            "support_plane_source_z_m",
            "placement_tolerance_m",
        }
        if set(record) != expected_keys:
            raise ValueError("PLCS support_plane keys differ from the v5 schema.")
        if record["schema"] != PLCS_SUPPORT_PLANE_SCHEMA:
            raise ValueError("Unsupported PLCS support-plane schema.")
        if record["surface_definition"] != _SURFACE_DEFINITION:
            raise ValueError("Unsupported PLCS support surface definition.")
        return cls(
            source_frame_index=_integer(
                record["source_frame_index"], name="source_frame_index"
            ),
            vertex_count=_integer(record["vertex_count"], name="vertex_count"),
            initial_root_translation_z_m=_number(
                record["initial_root_translation_z_m"],
                name="initial_root_translation_z_m",
            ),
            support_local_z_m=_number(
                record["support_local_z_m"], name="support_local_z_m"
            ),
            support_plane_source_z_m=_number(
                record["support_plane_source_z_m"],
                name="support_plane_source_z_m",
            ),
            placement_tolerance_m=_number(
                record["placement_tolerance_m"], name="placement_tolerance_m"
            ),
        )


def _coordinate_contract_payload() -> dict[str, object]:
    return {
        "schema": PLCS_COORDINATE_CONTRACT_SCHEMA,
        "handedness": "right-handed",
        "up_axis": "+Z",
        "linear_unit": "metre",
        "global_orient_application": "smplh_lbs",
        "root_translation_frame": "amass_source_frame",
        "court_orientation": "configured_positive_z_yaw_only",
    }


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


PLCS_COORDINATE_CONTRACT = PLCSCoordinateContract()


__all__ = [
    "PLCS_COORDINATE_CONTRACT",
    "PLCS_COORDINATE_CONTRACT_SCHEMA",
    "PLCS_SUPPORT_PLACEMENT_TOLERANCE_M",
    "PLCS_SUPPORT_PLANE_SCHEMA",
    "PLCSCoordinateContract",
    "PLCSSourceSupportPlane",
    "SMPLH_SURFACE_VERTEX_COUNT",
]

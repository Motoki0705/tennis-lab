"""Canonical AMASS/SMPL-H coordinate and initial-support contracts for PLCS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

PLCS_COORDINATE_CONTRACT_SCHEMA = "plcs_amass_smplh_z_up_v1"
SMPLH_SURFACE_VERTEX_COUNT = 6_890


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


PLCS_COORDINATE_CONTRACT = PLCSCoordinateContract()

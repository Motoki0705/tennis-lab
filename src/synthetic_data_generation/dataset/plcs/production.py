"""Typed production-mode authority for PLCS dataset generation."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Literal


class PLCSProductionMode(StrEnum):
    """Finite production topology selected by the PLCS Hydra config group."""

    SINGLE_OBJECT = "single_object"
    MULTI_OBJECT_GLOBAL_TIMELINE = "multi_object_global_timeline"

    @property
    def persisted_timeline_mode(self) -> Literal["single", "multi"]:
        """Return the existing compact-v4 logical-scene mode marker."""
        if self is PLCSProductionMode.SINGLE_OBJECT:
            return "single"
        return "multi"

    @classmethod
    def from_persisted_timeline_mode(cls, value: object) -> PLCSProductionMode:
        """Resolve the compact-v4 marker without accepting aliases."""
        if value == "single":
            return cls.SINGLE_OBJECT
        if value == "multi":
            return cls.MULTI_OBJECT_GLOBAL_TIMELINE
        raise ValueError("PLCS logical scene mode is unsupported.")


def validate_plcs_production_contract(
    *,
    mode: PLCSProductionMode,
    configured_motion_categories: Iterable[str],
    object_motion_categories: Iterable[str],
    object_start_frames: Iterable[int],
) -> None:
    """Fail closed on every mode/category/cardinality/start-frame mismatch."""
    if not isinstance(mode, PLCSProductionMode):
        raise TypeError("PLCS production mode must be a PLCSProductionMode.")
    configured = tuple(configured_motion_categories)
    requested = tuple(object_motion_categories)
    starts = tuple(object_start_frames)
    if not configured or any(not isinstance(value, str) or not value for value in configured):
        raise ValueError("PLCS configured motion categories must be explicit.")
    if not requested or any(not isinstance(value, str) or not value for value in requested):
        raise ValueError("PLCS object motion categories must be explicit.")
    supported = {"running", "walking", "general"}
    if len(configured) != len(set(configured)) or not set(configured).issubset(supported):
        raise ValueError(
            "PLCS configured motion categories must be unique production categories."
        )
    if not set(requested).issubset(supported):
        raise ValueError("PLCS objects contain an unsupported motion category.")
    if len(requested) != len(starts):
        raise ValueError("PLCS object categories and start frames differ in cardinality.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in starts):
        raise ValueError("PLCS object start frames must be non-negative integers.")
    if set(requested) != set(configured):
        raise ValueError(
            "PLCS objects must explicitly request every configured motion category."
        )

    if mode is PLCSProductionMode.SINGLE_OBJECT:
        if len(requested) != 1:
            raise ValueError("PLCS single-object mode requires exactly one object.")
        if starts != (0,):
            raise ValueError("PLCS single-object mode requires start_frame=0.")
        return

    if len(requested) < 2:
        raise ValueError(
            "PLCS multi-object global-timeline mode requires at least two objects."
        )
    if set(configured) != {"running", "walking", "general"}:
        raise ValueError(
            "PLCS multi-object global-timeline mode requires running, walking, "
            "and general motion categories."
        )
    if min(starts) != 0:
        raise ValueError(
            "PLCS multi-object global-timeline mode requires a track at frame zero."
        )


__all__ = ["PLCSProductionMode", "validate_plcs_production_contract"]

"""Explicit CPU solver and segmentation settings; all times use source frames."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SegmentPhysicsConfig:
    observation: str
    gravity_mps2: float
    initial_drag_per_m: float
    max_drag_per_m: float
    offset_bound_frames: float
    sigma_px: float
    robust_scale: float
    event_velocity_change_mps: float
    bounce_height_m: float
    event_window_frames: int
    min_segment_frames: int
    max_segment_frames: int
    max_gap_frames: int
    integration_steps: int
    max_nfev: int

    def __post_init__(self) -> None:
        if self.observation not in {"candidates", "heatmap"}:
            raise ValueError("observation must be candidates or heatmap")
        integer_fields = {
            "event_window_frames",
            "min_segment_frames",
            "max_segment_frames",
            "max_gap_frames",
            "integration_steps",
            "max_nfev",
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "observation":
                continue
            if field.name in integer_fields:
                if type(value) is not int or value < (
                    0 if field.name == "max_gap_frames" else 1
                ):
                    raise ValueError(
                        f"{field.name} must be a positive integer (gap may be zero)"
                    )
            elif (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name} must be finite and positive")
        if (
            not self.initial_drag_per_m < self.max_drag_per_m
            or not self.offset_bound_frames < 1
        ):
            raise ValueError(
                "Drag initialization must be interior; offsets must be subframe"
            )
        if (
            self.event_window_frames < 2
            or self.min_segment_frames < 5
            or self.max_segment_frames < self.min_segment_frames
            or self.integration_steps < 8
        ):
            raise ValueError("Insufficient event, segment or integration support")

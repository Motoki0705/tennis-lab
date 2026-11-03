"""Explicit CPU solver and segmentation settings; all times use source frames."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SegmentPhysicsConfig:
    observation: str = "candidates"
    gravity_mps2: float = 9.81
    initial_drag_per_m: float = 0.015
    max_drag_per_m: float = 0.1
    offset_bound_frames: float = 0.99
    sigma_px: float = 6.0
    robust_scale: float = 2.0
    event_velocity_change_mps: float = 8.0
    bounce_height_m: float = 1.0
    event_window_frames: int = 4
    min_segment_frames: int = 8
    max_segment_frames: int = 100
    max_gap_frames: int = 3
    integration_steps: int = 48
    max_nfev: int = 100

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

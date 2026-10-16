"""Recorded physics teachers; none of these fields belong to the model call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.utils.physics.ball.record import BallPhysicsRecord
from src.utils.physics.ball.surfaces import SURFACE_NAMES
from src.utils.schema.court_normalization import denormalize_court_velocity

TIME_FIELDS = ("physics_spin", "physics_segment", "physics_event", "frame_index")
STATIC_FIELDS = (
    "physics_field",
    "physics_clock",
    "physics_surface",
    "clip_frames",
    "bounce_count",
)


def record_teachers(
    arrays: Mapping[str, Any], *, start: int, end: int, full_length: int
) -> dict[str, Tensor]:
    """Validate the complete record before cropping derived frame teachers."""
    record = BallPhysicsRecord.from_arrays(dict(arrays))
    if record.frames != full_length or not 0 <= start < end <= record.frames:
        raise ValueError("BLCS physics record and trajectory/crop lengths disagree")
    if not np.allclose(
        denormalize_court_velocity(np.asarray(arrays["ball_vel_norm"])),
        record.velocity,
        rtol=1e-6,
        atol=1e-5,
    ):
        raise ValueError(
            "BLCS velocity target disagrees with the saved simulator velocity"
        )
    event = record.event_mask()
    return {
        "physics_spin": torch.from_numpy(record.frame_spin()[start:end].copy()),
        "physics_segment": torch.from_numpy(record.frame_segment()[start:end].copy()),
        "physics_event": torch.from_numpy(event[start:end].astype(np.int64)),
        "frame_index": torch.arange(start, end),
        # Double preserves the recorded dt and Python gravity exactly. Force
        # coefficients are cast to state dtype only when invoking the simulator.
        "physics_field": torch.tensor(
            [record.gravity, record.k_drag, record.k_magnus, *record.wind],
            dtype=torch.float64,
        ),
        "physics_clock": torch.tensor(
            [record.dt, record.sim_fps, record.output_fps, record.stride],
            dtype=torch.float64,
        ),
        "physics_surface": torch.tensor(SURFACE_NAMES.index(record.surface)),
        "clip_frames": torch.tensor(full_length),
        "bounce_count": torch.tensor(int(((event & 2) != 0).sum())),
    }


def collate_teachers(
    samples: Sequence[Mapping[str, Any]], length: int
) -> dict[str, Tensor]:
    """Pad frame teachers independently of camera visibility."""
    result = {}
    for key in TIME_FIELDS:
        rows = []
        for sample in samples:
            value = sample[key]
            fill = -1 if key in {"physics_segment", "frame_index"} else 0
            padded = value.new_full((length, *value.shape[1:]), fill)
            padded[: len(value)] = value
            rows.append(padded)
        result[key] = torch.stack(rows)
    for key in STATIC_FIELDS:
        result[key] = torch.stack([sample[key] for sample in samples])
    return result


@dataclass(frozen=True)
class PhysicsTargets:
    spin: Tensor
    segment: Tensor
    event: Tensor
    field: Tensor
    clock: Tensor
    surface: Tensor
    frame_index: Tensor
    clip_frames: Tensor
    bounce_count: Tensor

    @classmethod
    def from_batch(cls, batch: Mapping[str, Any], valid: Tensor) -> PhysicsTargets:
        b, t = valid.shape
        shapes = {
            "physics_spin": (b, t, 3),
            "physics_segment": (b, t),
            "physics_event": (b, t),
            "physics_field": (b, 6),
            "physics_clock": (b, 4),
            "physics_surface": (b,),
            "frame_index": (b, t),
            "clip_frames": (b,),
            "bounce_count": (b,),
        }
        for key, shape in shapes.items():
            value = batch[key]
            if not isinstance(value, Tensor) or value.shape != shape:
                raise ValueError(f"{key} must be a tensor of shape {shape}")
            if value.device != valid.device or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key} must be finite and on the batch device")
        for key in (
            "physics_segment",
            "physics_event",
            "physics_surface",
            "frame_index",
            "clip_frames",
            "bounce_count",
        ):
            if batch[key].dtype != torch.int64:
                raise ValueError(f"{key} must be int64")
        clock, field = batch["physics_clock"], batch["physics_field"]
        if bool((clock <= 0).any()) or not bool(
            (clock[:, 1:] == clock[:, 1:].round()).all()
        ):
            raise ValueError(
                "Physics clock requires positive dt and integer FPS/substeps"
            )
        if not bool(
            (clock[:, 1] == clock[:, 2] * clock[:, 3]).all()
        ) or not torch.allclose(
            clock[:, 0] * clock[:, 1],
            torch.ones_like(clock[:, 0]),
            rtol=1e-5,
            atol=0,
        ):
            raise ValueError("Inconsistent recorded simulation clock")
        if bool((field[:, 0] <= 0).any()) or bool((field[:, 1:3] < 0).any()):
            raise ValueError("Invalid physics field")
        if bool((batch["physics_segment"][valid] < 0).any()):
            raise ValueError("Valid frames require a flight segment")
        if bool((batch["physics_segment"][~valid] != -1).any()):
            raise ValueError("Padded physics segments must be -1")
        return cls(
            spin=batch["physics_spin"],
            segment=batch["physics_segment"],
            event=batch["physics_event"],
            field=field,
            clock=clock,
            surface=batch["physics_surface"],
            frame_index=batch["frame_index"],
            clip_frames=batch["clip_frames"],
            bounce_count=batch["bounce_count"],
        )

    def support(self, valid: Tensor, horizon: int) -> Tensor:
        """Every point in the closed interval must be real and in one flight."""
        if horizon < 1:
            raise ValueError("Rollout horizons must be positive")
        width = max(valid.shape[1] - horizon, 0)
        support = valid[:, :width].clone()
        for offset in range(1, horizon + 1):
            support &= valid[:, offset : offset + width]
            support &= (
                self.segment[:, :width] == self.segment[:, offset : offset + width]
            )
        return support

    def frame_dt(self) -> Tensor:
        """Actual output interval, including the generator's rounded sim dt."""
        return self.clock[:, 0] * self.clock[:, 3]

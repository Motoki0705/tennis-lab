"""Pack non-overlapping physical track intervals into reusable query slots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class _TrackInterval:
    track_index: int
    birth_frame: int
    death_frame: int


@dataclass(frozen=True)
class LifecycleSlotAssignment:
    """Physical-track to lifecycle-slot assignment and generated targets."""

    track_to_slot: Tensor
    target_presence: Tensor
    target_instance_id: Tensor

    def pack_tensor(
        self,
        values: Tensor,
        physical_presence: Tensor,
        *,
        fill_value: float | int | Tensor = 0,
    ) -> Tensor:
        """Pack ``(T, P, ...)`` values into ``(T, Q, ...)`` lifecycle slots."""
        if values.shape[:2] != physical_presence.shape:
            raise ValueError(
                "values leading dimensions must match physical_presence: "
                f"got {tuple(values.shape[:2])} and {tuple(physical_presence.shape)}."
            )
        output_shape = (
            values.shape[0],
            self.target_presence.shape[1],
            *values.shape[2:],
        )
        if isinstance(fill_value, Tensor):
            output = (
                fill_value.to(dtype=values.dtype, device=values.device)
                .expand(output_shape)
                .clone()
            )
        else:
            output = torch.full(
                output_shape,
                fill_value,
                dtype=values.dtype,
                device=values.device,
            )
        for track_index, slot_index in enumerate(self.track_to_slot.tolist()):
            if slot_index < 0:
                continue
            active = physical_presence[:, track_index]
            output[active, slot_index] = values[active, track_index]
        return output


def _extract_intervals(presence: Tensor) -> list[_TrackInterval]:
    intervals: list[_TrackInterval] = []
    for track_index in range(presence.shape[1]):
        active = torch.nonzero(presence[:, track_index], as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        birth = int(active[0])
        death = int(active[-1]) + 1
        if not bool(presence[birth:death, track_index].all()):
            raise ValueError(
                "Each physical track must contain one contiguous birth/death interval; "
                f"track {track_index} is disjoint."
            )
        intervals.append(
            _TrackInterval(
                track_index=track_index,
                birth_frame=birth,
                death_frame=death,
            )
        )
    return sorted(intervals, key=lambda item: (item.birth_frame, item.track_index))


def pack_lifecycle_slots(
    physical_presence: Tensor,
    *,
    num_slots: int,
    min_reuse_gap_frames: int = 0,
    randomize_slots: bool = False,
    rng: np.random.Generator | None = None,
) -> LifecycleSlotAssignment:
    """Color physical intervals into reusable slots without illegal overlap."""
    if physical_presence.ndim != 2:
        raise ValueError("physical_presence must have shape (T, P).")
    if num_slots <= 0:
        raise ValueError("num_slots must be positive.")
    if min_reuse_gap_frames < 0:
        raise ValueError("min_reuse_gap_frames must be non-negative.")
    presence = physical_presence.bool()
    intervals = _extract_intervals(presence)
    track_to_slot = torch.full(
        (presence.shape[1],), -1, dtype=torch.long, device=presence.device
    )
    target_presence = torch.zeros(
        (presence.shape[0], num_slots), dtype=torch.bool, device=presence.device
    )
    target_instance_id = torch.full(
        (presence.shape[0], num_slots),
        -1,
        dtype=torch.long,
        device=presence.device,
    )
    slot_death = [-min_reuse_gap_frames] * num_slots
    generator = rng or np.random.default_rng()

    for interval in intervals:
        available = [
            slot
            for slot, previous_death in enumerate(slot_death)
            if previous_death + min_reuse_gap_frames <= interval.birth_frame
        ]
        if not available:
            raise ValueError(
                "Lifecycle intervals cannot be packed into the configured query slots; "
                f"track={interval.track_index}, birth={interval.birth_frame}, "
                f"num_slots={num_slots}."
            )
        if randomize_slots:
            slot_index = int(generator.choice(np.asarray(available)))
        else:
            slot_index = available[0]
        track_to_slot[interval.track_index] = slot_index
        target_presence[interval.birth_frame : interval.death_frame, slot_index] = True
        target_instance_id[interval.birth_frame : interval.death_frame, slot_index] = (
            interval.track_index
        )
        slot_death[slot_index] = interval.death_frame

    return LifecycleSlotAssignment(
        track_to_slot=track_to_slot,
        target_presence=target_presence,
        target_instance_id=target_instance_id,
    )


__all__ = ["LifecycleSlotAssignment", "pack_lifecycle_slots"]

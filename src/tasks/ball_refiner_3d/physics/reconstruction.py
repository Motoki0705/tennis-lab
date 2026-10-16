"""Integrate predicted parameters into a per-frame trajectory.

Every flight segment is integrated from its predicted initial state with the
rally's predicted field by the shared simulator dynamics; all segments of a
batch run in parallel for as many frames as the longest one.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from src.tasks.ball_refiner_3d.physics.targets import FlightClock
from src.tasks.ball_refiner_3d.physics.units import decode_field, decode_state
from src.utils.physics.ball import BallField, integrate_flight


class SegmentLayout(NamedTuple):
    """``member (B,S,T)``, ``start``/``length (B,S)``; empty rows are padding."""

    member: Tensor
    start: Tensor
    length: Tensor

    @property
    def valid(self) -> Tensor:
        valid: Tensor = self.length > 0
        return valid


def segment_layout(segment: Tensor) -> SegmentLayout:
    """Layout of contiguous labels ``(B,T)`` numbered from 0; -1 marks padding."""
    if segment.ndim != 2 or segment.dtype != torch.long:
        raise ValueError("Segment labels must be a (B,T) long tensor")
    if int(segment.max()) < 0:
        raise ValueError("Every sequence needs at least one segment")
    count = int(segment.max()) + 1
    member = (
        segment[:, None, :] == torch.arange(count, device=segment.device)[None, :, None]
    )
    frames = segment.shape[1]
    index = torch.arange(frames, device=segment.device)
    start = torch.where(member, index, frames).amin(dim=-1)
    length = member.sum(dim=-1)
    last = torch.where(member, index, -1).amax(dim=-1)
    if torch.any((length > 0) & (last - start + 1 != length)):
        raise ValueError("Segment labels must form contiguous runs")
    return SegmentLayout(member, start, length)


def integrate_segments(
    states: Tensor, field: Tensor, segment: Tensor, clock: FlightClock
) -> Tensor:
    """Physical positions ``(B,T,3)`` from network-unit ``states (B,S,9)`` and
    ``field (B,4)``; frames labelled -1 are zero."""
    layout = segment_layout(segment)
    batch, count = layout.start.shape
    if states.shape[:2] != (batch, count) or field.shape[0] != batch:
        raise ValueError("States must be (B,S,9) for the labelled segments")
    state = decode_state(states)
    decoded = decode_field(field)
    positions, _ = integrate_flight(
        state.position,
        state.velocity,
        state.spin,
        BallField(
            clock.gravity,
            decoded.k_drag[:, None],
            decoded.k_magnus[:, None],
            decoded.wind[:, None, :],
        ),
        dt=clock.dt,
        substeps=clock.substeps,
        frames=int(layout.length.max()),
    )  # (B,S,L,3)
    # Padded frames (-1) read an arbitrary in-range sample and are zeroed below.
    label = segment.clamp(min=0)
    frame = torch.arange(segment.shape[1], device=segment.device)[None]
    local = (frame - layout.start.gather(1, label)).clamp(0, positions.shape[2] - 1)
    rows = torch.arange(batch, device=segment.device)[:, None]
    result = positions[rows, label, local]
    return torch.where(segment[..., None] >= 0, result, torch.zeros_like(result))

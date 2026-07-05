"""Temporal window generation for streaming video inference."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

from src.utils.video.types import FramePacket, TemporalWindow

TFrame = TypeVar("TFrame")


def build_window_starts(
    *,
    frame_count: int,
    sequence_length: int,
    stride: int,
) -> list[int]:
    """Compute sliding-window start indices covering every frame.

    The final window is always anchored at ``frame_count - sequence_length`` so
    the tail of the clip is covered even when ``stride`` does not divide evenly.
    """
    if frame_count < sequence_length:
        raise ValueError(
            f"frame_count must be >= sequence_length, got {frame_count} < {sequence_length}."
        )

    starts = list(range(0, frame_count - sequence_length + 1, stride))
    last_start = frame_count - sequence_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def chunked(values: Sequence[int], chunk_size: int) -> Iterator[list[int]]:
    """Yield ``values`` in lists of at most ``chunk_size`` items."""
    for start_index in range(0, len(values), chunk_size):
        yield list(values[start_index : start_index + chunk_size])


def iter_temporal_windows(
    frames: Iterable[FramePacket[TFrame]],
    *,
    sequence_length: int,
    stride: int,
    tail_policy: str = "backfill",
) -> Iterator[TemporalWindow[TFrame]]:
    """Yield fixed-length temporal windows from a frame stream.

    ``tail_policy="backfill"`` avoids repeated-tail padding for the final
    partial segment. If the final regular window does not end at the last
    frame, a final window ending at the last frame is emitted instead.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if tail_policy not in {"backfill", "drop"}:
        raise ValueError(
            f"tail_policy must be one of ['backfill', 'drop'], got '{tail_policy}'."
        )

    rolling: deque[FramePacket[TFrame]] = deque(maxlen=sequence_length)
    next_start = 0
    last_yielded_start: int | None = None
    frame_count = 0

    for packet in frames:
        rolling.append(packet)
        frame_count = packet.index + 1
        current_start = packet.index - sequence_length + 1
        if current_start == next_start and len(rolling) == sequence_length:
            window = _make_window(start_index=current_start, packets=tuple(rolling))
            yield window
            last_yielded_start = current_start
            next_start += stride

    if frame_count == 0:
        return

    if len(rolling) < sequence_length:
        if tail_policy == "drop":
            return
        padded = list(rolling)
        while len(padded) < sequence_length:
            padded.append(padded[-1])
        if last_yielded_start != 0:
            yield _make_window(start_index=0, packets=tuple(padded))
        return

    final_start = frame_count - sequence_length
    if tail_policy == "drop":
        return
    if tail_policy == "backfill" and (
        last_yielded_start is None or final_start > last_yielded_start
    ):
        yield _make_window(start_index=final_start, packets=tuple(rolling))


def _make_window(
    *,
    start_index: int,
    packets: tuple[FramePacket[TFrame], ...],
) -> TemporalWindow[TFrame]:
    return TemporalWindow(
        start_index=start_index,
        frame_indices=tuple(packet.index for packet in packets),
        frames=tuple(packet.frame for packet in packets),
    )

"""Exact global-frame and cross-chunk continuity validation for domain outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimelineFrameRecord:
    """One object/camera label record on the compositor's global timeline."""

    frame_index: int
    chunk_index: int
    track_id: str
    present: bool
    source_frame_index: int | None
    camera_id: str
    label_id: str
    court_instance_id: str

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.chunk_index < 0:
            raise ValueError("frame_index and chunk_index must be non-negative.")
        for name, value in (
            ("track_id", self.track_id),
            ("camera_id", self.camera_id),
            ("label_id", self.label_id),
            ("court_instance_id", self.court_instance_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty.")
        if self.present != (self.source_frame_index is not None):
            raise ValueError("present must agree with source_frame_index availability.")
        if self.source_frame_index is not None and self.source_frame_index < 0:
            raise ValueError("source_frame_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class FrameContinuityReport:
    """Machine-readable exact-set and boundary continuity result."""

    frame_count: int
    chunk_count: int
    track_count: int
    camera_count: int
    record_count: int


def validate_frame_continuity(
    records: Sequence[TimelineFrameRecord],
    *,
    frame_count: int,
) -> FrameContinuityReport:
    """Reject incomplete/duplicate labels and discontinuous source mappings."""
    if frame_count <= 0 or not records:
        raise ValueError("frame_count and records must be non-empty.")
    by_key: dict[tuple[int, str, str], TimelineFrameRecord] = {}
    for record in records:
        key = (record.frame_index, record.track_id, record.camera_id)
        if key in by_key:
            raise ValueError(f"Duplicate frame/track/camera record: {key}.")
        if record.frame_index >= frame_count:
            raise ValueError(f"Frame index exceeds global timeline: {record.frame_index}.")
        by_key[key] = record
    frame_indices = {record.frame_index for record in records}
    expected = set(range(frame_count))
    if frame_indices != expected:
        raise ValueError(
            f"Global timeline coverage mismatch; missing={sorted(expected - frame_indices)}, "
            f"unexpected={sorted(frame_indices - expected)}."
        )

    by_track_camera: dict[tuple[str, str], list[TimelineFrameRecord]] = defaultdict(list)
    for record in records:
        by_track_camera[(record.track_id, record.camera_id)].append(record)
    for track_camera_key, sequence in by_track_camera.items():
        sequence.sort(key=lambda item: item.frame_index)
        sequence_frame_indices = {item.frame_index for item in sequence}
        if sequence_frame_indices != expected:
            raise ValueError(
                "Track/camera timeline coverage mismatch for "
                f"{track_camera_key}; missing="
                f"{sorted(expected - sequence_frame_indices)}, unexpected="
                f"{sorted(sequence_frame_indices - expected)}."
            )
        courts = {item.court_instance_id for item in sequence}
        if len(courts) != 1:
            raise ValueError(
                f"Track/camera target-court binding changed: {track_camera_key}."
            )
        for previous, current in zip(sequence, sequence[1:], strict=False):
            previous_source = previous.source_frame_index
            current_source = current.source_frame_index
            if (
                previous.present
                and current.present
                and (
                    previous_source is None
                    or current_source is None
                    or current_source != previous_source + 1
                )
            ):
                raise ValueError(
                    f"Source frame mapping is discontinuous for {track_camera_key}."
                )
            if (
                current.frame_index == previous.frame_index + 1
                and (
                    current.chunk_index < previous.chunk_index
                    or current.chunk_index > previous.chunk_index + 1
                )
            ):
                raise ValueError(
                    f"Chunk index is discontinuous for {track_camera_key}."
                )
    label_ids = [record.label_id for record in records]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("label_id values must be globally unique.")
    return FrameContinuityReport(
        frame_count=frame_count,
        chunk_count=len({record.chunk_index for record in records}),
        track_count=len({record.track_id for record in records}),
        camera_count=len({record.camera_id for record in records}),
        record_count=len(records),
    )


__all__ = [
    "FrameContinuityReport",
    "TimelineFrameRecord",
    "validate_frame_continuity",
]

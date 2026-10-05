"""Tracking review states retain the saved observation and reconstruction masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.evaluation.labels import AMBIGUOUS, ClipLabels


class FrameSource(Protocol):
    def read(self, frame: int) -> NDArray[np.uint8]: ...


def runs(mask: NDArray[np.bool_]) -> list[list[int]]:
    """Return half-open runs without joining an absent frame."""
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("Timeline mask must be one-dimensional bool")
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [
        [int(start), int(stop)]
        for start, stop in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
        )
    ]


@dataclass(frozen=True)
class ReviewSequence:
    key: str
    clip_id: str
    camera_id: str
    width: int
    height: int
    frame_index: NDArray[np.integer[Any]]
    pts: NDArray[np.integer[Any]]
    time_base: str
    track_ids: NDArray[np.int64]
    boxes: NDArray[np.float32]
    observed: NDArray[np.bool_]
    detection_rows: NDArray[np.int64] | None
    source_track_ids: tuple[tuple[int, ...], ...]
    reconstructed_boxes: NDArray[np.float32] | None
    interpolated: NDArray[np.bool_] | None
    selected: NDArray[np.bool_] | None
    group_ids: NDArray[np.int64] | None
    intervals: tuple[dict[str, Any], ...]
    reference: ClipLabels | None
    metadata: dict[str, Any]
    images: FrameSource

    def __post_init__(self) -> None:
        count = len(self.frame_index)
        shape = (len(self.track_ids), count)
        if count < 1 or self.width <= 0 or self.height <= 0:
            raise ValueError("Review requires nonempty frame and image axes")
        if (
            not np.issubdtype(self.frame_index.dtype, np.integer)
            or self.frame_index.shape != (count,)
            or not np.array_equal(self.frame_index, np.arange(count))
        ):
            raise ValueError(
                "Tracking frame indices must match the complete source timeline"
            )
        if (
            self.pts.shape != (count,)
            or not np.issubdtype(self.pts.dtype, np.integer)
            or (np.diff(self.pts) <= 0).any()
        ):
            raise ValueError(
                "Tracking PTS must be actual, strictly increasing integers"
            )
        if (
            self.track_ids.dtype != np.int64
            or self.track_ids.ndim != 1
            or len(np.unique(self.track_ids)) != len(self.track_ids)
        ):
            raise ValueError("Raw track IDs must be unique int64 values")
        if (
            self.boxes.shape != (*shape, 4)
            or self.boxes.dtype != np.float32
            or not np.isfinite(self.boxes).all()
        ):
            raise ValueError("Tracking boxes must be finite float32 P,T,4")
        if self.observed.shape != shape or self.observed.dtype != np.bool_:
            raise ValueError("Tracking observed mask must preserve the P,T axes")
        if (self.boxes[self.observed, 2:] <= self.boxes[self.observed, :2]).any():
            raise ValueError("Observed boxes must have positive width and height")
        if len(self.source_track_ids) != len(self.track_ids):
            raise ValueError("AFLink source ID groups must preserve the raw track axis")
        if any(
            not ids or ids[0] != int(track) or len(set(ids)) != len(ids)
            for track, ids in zip(self.track_ids, self.source_track_ids, strict=True)
        ):
            raise ValueError("Stable ID must be the first unique AFLink source ID")
        members = [member for ids in self.source_track_ids for member in ids]
        if len(set(members)) != len(members):
            raise ValueError("An AFLink source ID belongs to one stable track")
        if self.detection_rows is not None and (
            self.detection_rows.shape != shape
            or self.detection_rows.dtype != np.int64
            or (self.detection_rows < -1).any()
            or not np.array_equal(self.detection_rows >= 0, self.observed)
        ):
            raise ValueError(
                "Detection rows must identify exactly the real observations"
            )
        if (self.interpolated is None) != (self.reconstructed_boxes is None):
            raise ValueError("Synthetic boxes require their saved interpolation mask")
        if self.interpolated is not None:
            assert self.reconstructed_boxes is not None
            if (
                self.interpolated.shape != shape
                or self.interpolated.dtype != np.bool_
                or (self.interpolated & self.observed).any()
                or self.reconstructed_boxes.shape != self.boxes.shape
                or self.reconstructed_boxes.dtype != np.float32
                or not np.isfinite(self.reconstructed_boxes).all()
            ):
                raise ValueError(
                    "Saved synthetic boxes must stay separate from observed boxes"
                )
            synthetic = self.reconstructed_boxes[self.interpolated]
            if (synthetic[:, 2:] <= synthetic[:, :2]).any():
                raise ValueError(
                    "Saved synthetic boxes must have positive width and height"
                )
        if self.selected is not None and (
            self.selected.shape != shape
            or self.selected.dtype != np.bool_
            or (self.selected & ~self.observed).any()
        ):
            raise ValueError("Selection must contain only saved real observations")
        if self.group_ids is not None and (
            self.group_ids.shape != shape
            or self.group_ids.dtype != np.int64
            or (self.group_ids < -1).any()
            or ((self.group_ids >= 0) & ~self.observed).any()
        ):
            raise ValueError("Court group IDs must refer to real observations")
        if self.reference is not None and (
            self.reference.clip_id != self.clip_id
            or self.reference.num_frames != count
            or self.camera_id not in self.reference.cameras
        ):
            raise ValueError(
                "Partial reference clip/camera/frame axes differ from the source"
            )

    @property
    def frame_count(self) -> int:
        return len(self.frame_index)

    def track_row(self, track_id: int) -> int:
        indices = np.flatnonzero(self.track_ids == track_id)
        if not len(indices):
            raise KeyError(f"Unknown raw track ID {track_id}")
        return int(indices[0])

    def assignment(self, track_id: int, frame: int) -> dict[str, Any] | None:
        # An interval can cover absent frames, but it cannot turn them into labels.
        if not self.observed[self.track_row(track_id), frame]:
            return None
        for interval in self.intervals:
            if (
                interval["raw_track_id"] == track_id
                and interval["start_frame"] <= frame < interval["stop_frame"]
            ):
                return interval
        return None

    def track_frame(self, row: int, frame: int) -> dict[str, Any]:
        track_id = int(self.track_ids[row])
        state, box = "missing", None
        if self.observed[row, frame]:
            state, box = "observed", self.boxes[row, frame].tolist()
        elif self.interpolated is not None and self.interpolated[row, frame]:
            assert self.reconstructed_boxes is not None
            state, box = "interpolated", self.reconstructed_boxes[row, frame].tolist()
        assignment = self.assignment(track_id, frame)
        frames = np.flatnonzero(self.observed[row])
        before, after = frames[frames < frame], frames[frames > frame]
        return {
            "track_id": track_id,
            "state": state,
            "box": box,
            "detection_row": None
            if self.detection_rows is None or state != "observed"
            else int(self.detection_rows[row, frame]),
            "assignment": assignment,
            "selected": None
            if self.selected is None
            else bool(self.selected[row, frame]),
            "group_id": None
            if self.group_ids is None or self.group_ids[row, frame] < 0
            else int(self.group_ids[row, frame]),
            "previous_observed": int(before[-1]) if len(before) else None,
            "next_observed": int(after[0]) if len(after) else None,
        }

    def frame_info(self, frame: int) -> dict[str, Any]:
        if not 0 <= frame < self.frame_count:
            raise IndexError(f"Source has no frame {frame}")
        references = []
        if self.reference is not None:
            labelled = self.reference.cameras[self.camera_id]
            section = labelled.at(frame)
            for box, person in zip(
                labelled.boxes_xyxy[section],
                labelled.person_index[section],
                strict=True,
            ):
                identity = (
                    None if person == AMBIGUOUS else self.reference.people[int(person)]
                )
                references.append(
                    {
                        "box": box.tolist(),
                        "person": None if identity is None else identity.person_id,
                        "role": "ambiguous" if identity is None else identity.role,
                    }
                )
        return {
            "frame": frame,
            "pts": int(self.pts[frame]),
            "time_base": self.time_base,
            "tracks": [
                self.track_frame(row, frame) for row in range(len(self.track_ids))
            ],
            "reference_boxes": references,
        }

    def summary(self) -> dict[str, Any]:
        tracks, events = [], []
        for row, track_id in enumerate(self.track_ids):
            observed_runs = runs(self.observed[row])
            synthetic_runs = (
                [] if self.interpolated is None else runs(self.interpolated[row])
            )
            for previous, following in zip(
                observed_runs[:-1], observed_runs[1:], strict=True
            ):
                start, stop = previous[1], following[0]
                synthetic = (
                    0
                    if self.interpolated is None
                    else int(self.interpolated[row, start:stop].sum())
                )
                events.append(
                    {
                        "kind": "observation_gap",
                        "track_id": int(track_id),
                        "start": start,
                        "stop": stop,
                        "previous": start - 1,
                        "next": stop,
                        "synthetic_frames": synthetic,
                    }
                )
            tracks.append(
                {
                    "track_id": int(track_id),
                    "source_ids": list(self.source_track_ids[row]),
                    "observed_count": int(self.observed[row].sum()),
                    "observed_runs": observed_runs,
                    "synthetic_runs": synthetic_runs,
                    "selected_runs": None
                    if self.selected is None
                    else runs(self.selected[row]),
                    "intervals": [
                        i for i in self.intervals if i["raw_track_id"] == int(track_id)
                    ],
                }
            )
        labelled = sum(
            self.assignment(int(track), int(frame)) is not None
            for row, track in enumerate(self.track_ids)
            for frame in np.flatnonzero(self.observed[row])
        )
        return {
            "key": self.key,
            "clip_id": self.clip_id,
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "observations": int(self.observed.sum()),
            "labelled_observations": labelled,
            "saved_synthetic": None
            if self.interpolated is None
            else int(self.interpolated.sum()),
            "selected_observations": None
            if self.selected is None
            else int(self.selected.sum()),
            "tracks": tracks,
            "events": sorted(
                events,
                key=lambda event: (
                    -int(event["stop"]) + int(event["start"]),
                    int(event["start"]),
                ),
            ),
            "metadata": self.metadata,
        }

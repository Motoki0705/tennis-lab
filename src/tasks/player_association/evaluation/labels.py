"""Tracker-independent cross-camera person labels of one clip.

A label is ``(camera, frame, box) -> person``. Evaluation matches the boxes of
any tracker to these by IoU, so the labels survive tracker changes. Labels are
made by reviewing the tracks of one observation run (``materialize``); only the
boxes that run produced are labelled, so a person the tracker never boxed is
absent rather than negative.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SCHEMA = "player_association_labels_v1"
ROLES = ("player", "non_player")
AMBIGUOUS = -1
"""``person_index`` of a box that covers two people or mostly background; evaluation ignores it."""


@dataclass(frozen=True)
class LabelledPerson:
    person_id: str
    role: str  # "player": cross-camera identity is evaluated; "non_player": must be excluded
    description: str

    def __post_init__(self) -> None:
        if not self.person_id or self.role not in ROLES:
            raise ValueError(f"Invalid labelled person: {self}")


@dataclass(frozen=True)
class CameraLabels:
    """Labelled boxes of one camera, sorted by frame. Several boxes may share a frame and a person (duplicates)."""

    frames: NDArray[np.int64]  # (N,)
    person_index: NDArray[np.int64]  # (N,), index into ClipLabels.people or AMBIGUOUS
    boxes_xyxy: NDArray[np.float64]  # (N, 4)

    def __post_init__(self) -> None:
        count = len(self.frames)
        if self.frames.shape != (count,) or self.person_index.shape != (count,) or self.boxes_xyxy.shape != (count, 4):
            raise ValueError("Camera label arrays must be (N,), (N,), (N, 4)")
        if count and (np.diff(self.frames) < 0).any():
            raise ValueError("Camera labels must be sorted by frame")
        if not np.isfinite(self.boxes_xyxy).all() or (self.boxes_xyxy[:, 2:] <= self.boxes_xyxy[:, :2]).any():
            raise ValueError("Labelled boxes must be finite positive xyxy rectangles")

    def at(self, frame: int) -> slice:
        """Rows of ``frame``."""
        return slice(int(np.searchsorted(self.frames, frame, "left")), int(np.searchsorted(self.frames, frame, "right")))


@dataclass(frozen=True)
class ClipLabels:
    clip_id: str
    num_frames: int
    people: tuple[LabelledPerson, ...]
    cameras: dict[str, CameraLabels]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if len({person.person_id for person in self.people}) != len(self.people):
            raise ValueError("Labelled person IDs must be unique")
        for camera, labels in self.cameras.items():
            if len(labels.frames) and (labels.frames[0] < 0 or labels.frames[-1] >= self.num_frames):
                raise ValueError(f"{camera} labels fall outside the clip timeline")
            if ((labels.person_index < AMBIGUOUS) | (labels.person_index >= len(self.people))).any():
                raise ValueError(f"{camera} labels reference an unknown person")

    @property
    def roles(self) -> NDArray[np.str_]:
        return np.asarray([person.role for person in self.people])

    def to_json(self) -> dict[str, Any]:
        """Columnar JSON; boxes are rounded to 0.1 px."""
        return {"schema": SCHEMA, "clip_id": self.clip_id, "num_frames": self.num_frames,
                "people": [{"person_id": p.person_id, "role": p.role, "description": p.description} for p in self.people],
                "provenance": self.provenance,
                "cameras": {camera: {"frames": labels.frames.tolist(),
                                     "person": [None if index == AMBIGUOUS else self.people[index].person_id
                                                for index in labels.person_index.tolist()],
                                     "boxes_xyxy": np.round(labels.boxes_xyxy, 1).tolist()}
                            for camera, labels in self.cameras.items()}}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ClipLabels:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Expected schema {SCHEMA}, got {data.get('schema')!r}")
        people = tuple(LabelledPerson(str(p["person_id"]), str(p["role"]), str(p["description"])) for p in data["people"])
        index = {person.person_id: row for row, person in enumerate(people)}
        cameras = {}
        for camera, columns in data["cameras"].items():
            unknown = {name for name in columns["person"] if name is not None and name not in index}
            if unknown:
                raise ValueError(f"{camera} labels name undeclared people {sorted(unknown)}")
            cameras[str(camera)] = CameraLabels(
                np.asarray(columns["frames"], np.int64).reshape(-1),
                np.asarray([AMBIGUOUS if name is None else index[name] for name in columns["person"]], np.int64).reshape(-1),
                np.asarray(columns["boxes_xyxy"], np.float64).reshape(-1, 4))
        return cls(str(data["clip_id"]), int(data["num_frames"]), people, cameras, dict(data["provenance"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path: Path) -> ClipLabels:
        return cls.from_json(json.loads(path.read_text()))


@dataclass(frozen=True)
class ReviewedTrack:
    """The boxes of one reviewed camera-local track."""

    track_id: int
    boxes_xyxy: NDArray[np.float32]  # (T, 4)
    observed: NDArray[np.bool_]  # (T,)


def _segments(track: int, value: Any, num_frames: int) -> list[tuple[int, int, str | None]]:
    if value is None or isinstance(value, str):
        return [(0, num_frames, value)]
    if not isinstance(value, list) or not value:
        raise ValueError(f"Track {track}: expected a person, null or a list of [start, end, person] segments")
    segments = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3 or type(item[0]) is not int or type(item[1]) is not int \
                or not (item[2] is None or isinstance(item[2], str)):
            raise ValueError(f"Track {track}: invalid segment {item!r}")
        segments.append((item[0], item[1], item[2]))
    contiguous = all(previous[1] == following[0] for previous, following in zip(segments, segments[1:], strict=False))
    if segments[0][0] != 0 or segments[-1][1] != num_frames or not contiguous or any(start >= end for start, end, _ in segments):
        raise ValueError(f"Track {track}: segments must be contiguous, nonempty and cover [0, {num_frames})")
    return segments


def materialize(clip_id: str, num_frames: int, review: Mapping[str, Any], tracks: Mapping[str, list[ReviewedTrack]],
                provenance: dict[str, Any]) -> ClipLabels:
    """Turn a reviewed track assignment into box labels.

    ``review`` holds ``people`` ({id: {role, description}}) and ``tracks``
    ({camera: {track id: person | null | [[start, end, person | null], ...]}}).
    Every observed box of every track must be reviewed, and every reviewed
    track must exist: an unreviewed box would silently drop out of the test set.
    """
    people = tuple(LabelledPerson(str(key), str(value["role"]), str(value["description"])) for key, value in review["people"].items())
    index = {person.person_id: row for row, person in enumerate(people)}
    assigned: dict[str, dict[Any, Any]] = review["tracks"]
    if set(assigned) != set(tracks):
        raise ValueError(f"{clip_id}: reviewed cameras {sorted(assigned)} differ from observed cameras {sorted(tracks)}")
    used: set[str] = set()
    cameras = {}
    for camera, camera_tracks in tracks.items():
        review_ids = {int(key) for key in assigned[camera]}
        track_ids = {track.track_id for track in camera_tracks}
        if review_ids != track_ids:
            raise ValueError(f"{clip_id} {camera}: unreviewed tracks {sorted(track_ids - review_ids)}, "
                             f"reviewed tracks that do not exist {sorted(review_ids - track_ids)}")
        rows: list[tuple[int, int, NDArray[np.float32]]] = []
        for track in camera_tracks:
            if track.observed.shape != (num_frames,) or track.boxes_xyxy.shape != (num_frames, 4):
                raise ValueError(f"{clip_id} {camera} t{track.track_id}: track timeline differs from the clip")
            for start, end, person in _segments(track.track_id, assigned[camera][track.track_id], num_frames):
                if person is not None and person not in index:
                    raise ValueError(f"{clip_id} {camera} t{track.track_id}: undeclared person {person!r}")
                if person is not None:
                    used.add(person)
                label = AMBIGUOUS if person is None else index[person]
                rows.extend((int(frame), label, track.boxes_xyxy[frame]) for frame in start + np.flatnonzero(track.observed[start:end]))
        rows.sort(key=lambda row: row[0])
        cameras[camera] = CameraLabels(np.asarray([row[0] for row in rows], np.int64),
                                       np.asarray([row[1] for row in rows], np.int64),
                                       np.asarray([row[2] for row in rows], np.float64).reshape(-1, 4))
    if set(index) - used:
        raise ValueError(f"{clip_id}: declared people without any box {sorted(set(index) - used)}")
    return ClipLabels(clip_id, num_frames, people, cameras, provenance)


def review_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

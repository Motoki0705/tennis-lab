"""Human event supervision and its original-frame scoring regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.data.targets.events import gaussian_event_target
from src.tasks.ball_refiner_3d.events import EVENT_CLASS_BITS, EVENT_CLASSES
from src.tennis_scene.ball_event_schema import EVENT_CLASSES_V4
from src.tennis_scene.schema import SceneResult

MATCH_TOLERANCE = 2
UNCERTAIN_RADIUS = 2
STATUSES = ("certain", "probable", "uncertain")
SERVE_SUBTYPES = frozenset({"serve", "serve_overhead", "serve_underhand"})
VOLLEY_SUBTYPES = frozenset(
    {"volley", "volley_forehand", "volley_backhand", "volley_center"}
)


@dataclass(frozen=True, slots=True)
class Event:
    frame: int
    kind: str
    status: str
    in_point: bool
    subtype: str | None

    @property
    def channel(self) -> int:
        channel: int = EVENT_CLASSES.index(self.kind)
        return channel


@dataclass(frozen=True, slots=True)
class Annotations:
    events: tuple[Event, ...]
    coverage: NDArray[np.bool_]
    unknown_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        seen: set[tuple[int, str]] = set()
        for event in self.events:
            key = (event.frame, event.kind)
            if (
                type(event.frame) is not int
                or not 0 <= event.frame < len(self.coverage)
                or event.kind not in EVENT_CLASSES
                or event.status not in STATUSES
                or type(event.in_point) is not bool
                or key in seen
            ):
                raise ValueError(f"Invalid or duplicate event: {event}")
            seen.add(key)
        for frame in self.unknown_frames:
            if type(frame) is not int or not 0 <= frame < len(self.coverage):
                raise ValueError(f"Invalid unknown event frame: {frame}")
        if self.coverage.ndim != 1 or self.coverage.dtype != np.bool_:
            raise ValueError("Coverage must be a boolean frame vector")

    @classmethod
    def from_scene(cls, scene: SceneResult) -> Annotations:
        meta = scene.metadata.get("ball_events")
        if (
            not isinstance(meta, dict)
            or scene.schema_version not in (3, 4)
            or meta.get("source") != "annotation"
            or meta.get("statuses") != list(STATUSES)
        ):
            raise ValueError(
                "Post-training requires v3 or v4 human event annotations"
            )
        frames, kinds, statuses, subtypes, coverage = (
            scene.ball_event_frame,
            scene.ball_event_kind,
            scene.ball_event_status,
            scene.ball_event_subtype,
            scene.ball_event_coverage,
        )
        if (
            frames is None
            or kinds is None
            or statuses is None
            or subtypes is None
            or coverage is None
        ):
            raise ValueError("Scene has incomplete annotation arrays")

        if scene.schema_version == 3:
            if meta.get("classes") != list(EVENT_CLASSES):
                raise ValueError(
                    "Post-training requires three-class v3 human annotations"
                )
            in_point = scene.ball_event_in_point
            if in_point is None:
                raise ValueError("Scene has incomplete annotation arrays")
            events = tuple(
                Event(
                    int(frame),
                    EVENT_CLASSES[kind],
                    STATUSES[status],
                    bool(inside),
                    meta["subtypes"][subtype],
                )
                for frame, kind, status, inside, subtype in zip(
                    frames, kinds, statuses, in_point, subtypes, strict=True
                )
            )
            return cls(events, coverage)

        if meta.get("classes") != list(EVENT_CLASSES_V4):
            raise ValueError(
                "Post-training requires v4 human annotations with standard classes"
            )
        if scene.ball_event_in_point is not None:
            raise ValueError("SceneResult v4 must not include ball_event_in_point")
        parsed_events: list[Event] = []
        unknown_frames: list[int] = []
        for frame, kind, status, subtype in zip(
            frames, kinds, statuses, subtypes, strict=True
        ):
            kind_name = EVENT_CLASSES_V4[kind]
            if kind_name == "unknown":
                unknown_frames.append(int(frame))
            else:
                parsed_events.append(
                    Event(
                        int(frame),
                        kind_name,
                        STATUSES[status],
                        True,
                        meta["subtypes"][subtype],
                    )
                )
        return cls(tuple(parsed_events), coverage, tuple(unknown_frames))

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> Annotations:
        """Read the independent label document after its digest/grid are verified."""
        coverage = np.ones(doc["frames"], dtype=np.bool_)
        for interval in doc["unlabelable"]:
            start, end = interval["start"], interval["end"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start <= end < len(coverage)
            ):
                raise ValueError("Invalid independent unlabelable interval")
            coverage[start : end + 1] = False
        events: list[Event] = []
        unknown_frames: list[int] = []
        for e in doc["events"]:
            if e["kind"] == "unknown":
                unknown_frames.append(e["frame"])
            else:
                events.append(
                    Event(
                        e["frame"],
                        e["kind"],
                        e["status"],
                        bool(e.get("in_point", True)),
                        e.get("subtype"),
                    )
                )
        return cls(tuple(events), coverage, tuple(unknown_frames))

    def ignored_near_uncertain(self) -> NDArray[np.bool_]:
        ignored: NDArray[np.bool_] = np.zeros(
            (len(self.coverage), len(EVENT_CLASSES)), dtype=np.bool_
        )
        for event in self.events:
            if event.status == "uncertain":
                start = max(0, event.frame - UNCERTAIN_RADIUS)
                ignored[start : event.frame + UNCERTAIN_RADIUS + 1, event.channel] = (
                    True
                )
        for frame in self.unknown_frames:
            start = max(0, frame - UNCERTAIN_RADIUS)
            ignored[start : frame + UNCERTAIN_RADIUS + 1, :] = True
        return ignored

    def targets(self, sigma: float) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Render known events before masking frame/class pairs for BCE."""
        bits: NDArray[np.uint8] = np.zeros(len(self.coverage), dtype=np.uint8)
        valid = self.coverage[:, None] & ~self.ignored_near_uncertain()
        for event in self.events:
            if event.status != "uncertain":
                bits[event.frame] |= EVENT_CLASS_BITS[event.channel]
                valid[event.frame, event.channel] = self.coverage[event.frame]
        return gaussian_event_target(bits, sigma), valid

    def point_frames(self) -> NDArray[np.bool_]:
        """Approximate point spans using the first and last labeled contact."""
        inside = np.zeros_like(self.coverage)
        start: int | None = None
        previous = 0
        for event in sorted(self.events, key=lambda e: (e.frame, e.in_point)):
            if start is not None and (
                not event.in_point
                or (event.subtype in SERVE_SUBTYPES and event.frame > previous)
            ):
                inside[start : previous + 1] = True
                start = None
            if event.in_point:
                if start is None:
                    start = event.frame
                previous = event.frame
        if start is not None:
            inside[start : previous + 1] = True
        return inside & self.coverage

"""Human event supervision and its original-frame scoring regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.data.targets.events import gaussian_event_target
from src.tasks.ball_refiner_3d.events import EVENT_CLASS_BITS, EVENT_CLASSES
from src.tennis_scene.schema import SceneResult

MATCH_TOLERANCE = 2
UNCERTAIN_RADIUS = 2
STATUSES = ("certain", "probable", "uncertain")


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
        if self.coverage.ndim != 1 or self.coverage.dtype != np.bool_:
            raise ValueError("Coverage must be a boolean frame vector")

    @classmethod
    def from_scene(cls, scene: SceneResult) -> Annotations:
        meta = scene.metadata["ball_events"]
        if (
            scene.schema_version != 3
            or meta["source"] != "annotation"
            or meta["classes"] != list(EVENT_CLASSES)
            or meta["statuses"] != list(STATUSES)
        ):
            raise ValueError("Post-training requires three-class v3 human annotations")
        frames, kinds, statuses, in_point, subtypes, coverage = (
            scene.ball_event_frame,
            scene.ball_event_kind,
            scene.ball_event_status,
            scene.ball_event_in_point,
            scene.ball_event_subtype,
            scene.ball_event_coverage,
        )
        if (
            frames is None
            or kinds is None
            or statuses is None
            or in_point is None
            or subtypes is None
            or coverage is None
        ):
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
        return cls(
            tuple(
                Event(
                    e["frame"], e["kind"], e["status"], e["in_point"], e.get("subtype")
                )
                for e in doc["events"]
            ),
            coverage,
        )

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
        return ignored

    def targets(self, sigma: float) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Independent BCE masks. An explicit covered positive retains its center."""
        bits: NDArray[np.uint8] = np.zeros(len(self.coverage), dtype=np.uint8)
        valid = self.coverage[:, None] & ~self.ignored_near_uncertain()
        for event in self.events:
            if event.status != "uncertain" and self.coverage[event.frame]:
                bits[event.frame] |= EVENT_CLASS_BITS[event.channel]
                valid[event.frame, event.channel] = True
        return gaussian_event_target(bits, sigma), valid

    def point_frames(self) -> NDArray[np.bool_]:
        """Approximate point spans using the first and last labeled contact."""
        inside = np.zeros_like(self.coverage)
        start: int | None = None
        previous = 0
        for event in sorted(self.events, key=lambda e: (e.frame, e.in_point)):
            if start is not None and (
                not event.in_point
                or (event.subtype == "serve" and event.frame > previous)
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

"""Source-independent description of one clip to encode into the ball store.

Source adapters parse and validate their own annotation format and return
:class:`ClipSpec` values; the builder only encodes frames and concatenates
tables. Labels stay in **source pixels** here; the builder scales them once
when it decides the stored image size.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    EVENT_CODES,
    POINT_KIND_CODES,
    Source,
    located_codes,
)

# Per-frame label columns a source provides (the builder adds the rest).
LABEL_FRAME_FIELDS: Mapping[str, type[np.generic]] = {
    "pts": np.int64,
    "annotated": np.bool_,
    "is_target": np.bool_,
    "segment_break": np.bool_,
    "event": np.uint8,
    "inst_count": np.int32,
}
LABEL_INSTANCE_FIELDS: Mapping[str, type[np.generic]] = {
    "track_index": np.int32,
    "point_kind": np.uint8,
    "occluded": np.bool_,
}


@dataclass(frozen=True, slots=True)
class BallInstance:
    """One annotated ball of one frame, in source pixels (``None`` = unlocated)."""

    track_id: str
    point_kind: str
    xy: tuple[float, float] | None
    occluded: bool

    def __post_init__(self) -> None:
        if self.point_kind not in POINT_KIND_CODES:
            raise ValueError(f"Unknown point kind {self.point_kind!r}")
        located = POINT_KIND_CODES[self.point_kind] in located_codes()
        if located != (self.xy is not None):
            raise ValueError(f"{self.point_kind} ball must {'' if located else 'not '}have a position")


@dataclass(frozen=True, slots=True)
class FrameLabel:
    """Labels of one frame as a source states them."""

    pts: int
    annotated: bool
    is_target: bool
    segment_break: bool
    event: str
    balls: tuple[BallInstance, ...]

    def __post_init__(self) -> None:
        if self.event not in EVENT_CODES:
            raise ValueError(f"Unknown event {self.event!r}")
        ids = [ball.track_id for ball in self.balls]
        if len(ids) != len(set(ids)):
            raise ValueError("A frame lists the same ball track twice")


@dataclass(frozen=True, slots=True)
class ClipLabels:
    """Columnar labels of every frame of one clip, in source pixels."""

    track_ids: tuple[str, ...]
    frames: dict[str, NDArray[np.generic]]
    instances: dict[str, NDArray[np.generic]]
    xy: NDArray[np.float64]

    @classmethod
    def from_frames(cls, labels: list[FrameLabel], *, width: int, height: int) -> ClipLabels:
        """Validate and pack per-frame labels; located balls must lie in the image."""
        if not labels:
            raise ValueError("A clip needs at least one frame")
        track_ids = tuple(sorted({ball.track_id for label in labels for ball in label.balls}))
        lookup = {track: index for index, track in enumerate(track_ids)}
        frames: dict[str, list[object]] = {name: [] for name in LABEL_FRAME_FIELDS}
        instances: dict[str, list[object]] = {name: [] for name in LABEL_INSTANCE_FIELDS}
        xy: list[tuple[float, float]] = []
        for label in labels:
            for name, value in (
                ("pts", label.pts),
                ("annotated", label.annotated),
                ("is_target", label.is_target),
                ("segment_break", label.segment_break),
                ("event", EVENT_CODES[label.event]),
                ("inst_count", len(label.balls)),
            ):
                frames[name].append(value)
            for ball in label.balls:
                if ball.xy is not None:
                    x, y = ball.xy
                    if not (np.isfinite([x, y]).all() and 0 <= x < width and 0 <= y < height):
                        raise ValueError(f"Ball at {ball.xy} lies outside the {width}x{height} image")
                instances["track_index"].append(lookup[ball.track_id])
                instances["point_kind"].append(POINT_KIND_CODES[ball.point_kind])
                instances["occluded"].append(ball.occluded)
                xy.append((np.nan, np.nan) if ball.xy is None else ball.xy)
        packed_frames = {name: np.asarray(frames[name], dtype=dtype) for name, dtype in LABEL_FRAME_FIELDS.items()}
        if (np.diff(packed_frames["pts"]) <= 0).any():
            raise ValueError("Frame pts must strictly increase")
        return cls(
            track_ids=track_ids,
            frames=packed_frames,
            instances={name: np.asarray(instances[name], dtype=dtype) for name, dtype in LABEL_INSTANCE_FIELDS.items()},
            xy=np.asarray(xy, dtype=np.float64).reshape(-1, 2),
        )

    @property
    def frame_count(self) -> int:
        return int(self.frames["pts"].shape[0])


@dataclass(frozen=True, slots=True)
class JpegSequence:
    """Frames that already exist as JPEG files, stored byte-for-byte when not resized."""

    paths: tuple[Path, ...]

    def iter_frames(self) -> Iterator[bytes | NDArray[np.uint8]]:
        for path in self.paths:
            yield path.read_bytes()


@dataclass(frozen=True, slots=True)
class VideoFrames:
    """A video decoded in presentation order; every frame's PTS is checked.

    ``pts`` (in ``time_base`` units) is the expected presentation time of each
    frame; a decoder that drops, duplicates or reorders frames is an error.
    """

    path: Path
    pts: tuple[int, ...]
    time_base: Fraction

    def iter_frames(self) -> Iterator[bytes | NDArray[np.uint8]]:
        index = 0
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            for frame in container.decode(stream):
                if index >= len(self.pts):
                    raise ValueError(f"{self.path}: more frames than the annotation lists")
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f"{self.path}: decoded frame {index} has no PTS")
                if frame.pts * Fraction(frame.time_base) != self.pts[index] * self.time_base:
                    raise ValueError(f"{self.path}: frame {index} PTS differs from the annotation")
                index += 1
                pixels: NDArray[np.uint8] = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
                yield pixels
        if index != len(self.pts):
            raise ValueError(f"{self.path}: decoded {index} of {len(self.pts)} frames")


FrameMedia = JpegSequence | VideoFrames


@dataclass(frozen=True, slots=True)
class ClipSpec:
    """One clip to encode: identity, media, labels and provenance.

    ``provenance`` is copied verbatim into the clip's ``metadata.json`` entry.
    """

    clip_id: str
    source: Source
    group_id: str
    camera_id: str | None
    width: int
    height: int
    time_base: Fraction
    fps: Fraction
    has_events: bool
    annotation_path: Path
    annotation_sha256: str
    media_sha256: str
    media: FrameMedia
    labels: ClipLabels
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        events = self.labels.frames["event"]
        if not self.has_events and (events != EVENT_CODES["unlabeled"]).any():
            raise ValueError(f"{self.clip_id}: events given for a source without event labels")
        if self.has_events and (events == EVENT_CODES["unlabeled"]).all():
            raise ValueError(f"{self.clip_id}: declared with events but every frame is unlabeled")


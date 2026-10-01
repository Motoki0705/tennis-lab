"""Memory-mapped ball frame store shared by every ball-detection data source.

Layout of one dataset version (``data/ball_detection/<version>/``)::

    shards/clip-00000.bin   # concatenated JPEG bytes of every frame of one clip
    index.npz               # columnar frame and instance tables (below)
    metadata.json           # schema, clips (identity, source, split, tracks), build
    README.md               # generated human summary

The layout follows ``src/tasks/player_detection/data/store.py``. The ball
differs in one respect: **every frame of a clip is stored**, in presentation
order, so that temporal windows and negatives are available. A "clip" is one
camera's continuous frame sequence (a TrackNet ``Clip*`` directory, one camera
of a Meiji multi-camera clip, or one chat-annotation clip), and a stored row is
addressed as ``clip_start[clip] + frame_index``.

Coordinates are in **stored-image pixels** (``ClipRecord.width``/``height``);
a clip decoded taller than the build's ``max_height`` is downscaled and its
labels are scaled by ``width / source_width`` (see ``ClipRecord.scale``).

Label semantics (sources map onto these; the builder never invents labels):

* ``annotated`` is False for a frame the annotator did not review; it carries
  no supervision at all.
* An annotated frame with no instance asserts that no ball in play is visible.
* ``point_kind`` records how an instance's position was obtained:
  ``observed`` (seen in the image), ``interpolated`` (filled across a short
  gap), ``occlusion_estimated`` (hidden, position estimated from context),
  ``unresolved`` (a ball is in play but could not be located; NaN position) and
  ``out_of_frame`` (the ball left the image; NaN position). Training decides
  which kinds are positives and which make a frame unusable; an ``unresolved``
  ball is never a negative.
* ``occluded`` is True when the source says the ball is hidden.
* ``segment_break`` marks a hit, bounce, cut or play boundary that trajectory
  interpolation must not cross (as annotated; not every source provides it).
* ``event`` is the typed hit/bounce label where the source provides one;
  sources without event labels store ``unlabeled`` (never ``none``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION: Final = "ball_detection_frames.v1"
INDEX_FILE: Final = "index.npz"
METADATA_FILE: Final = "metadata.json"
SHARDS_DIR: Final = "shards"

SOURCES: Final = ("tracknet", "meiji", "chat_annotation")
Source = Literal["tracknet", "meiji", "chat_annotation"]
POINT_KIND_CODES: Final[Mapping[str, int]] = {
    "observed": 0,
    "interpolated": 1,
    "occlusion_estimated": 2,
    "unresolved": 3,
    "out_of_frame": 4,
}
POINT_KIND_NAMES: Final[Mapping[int, str]] = {code: name for name, code in POINT_KIND_CODES.items()}
LOCATED_POINT_KINDS: Final = frozenset({"observed", "interpolated", "occlusion_estimated"})
EVENT_CODES: Final[Mapping[str, int]] = {"none": 0, "hit": 1, "bounce": 2, "unlabeled": 3}
EVENT_NAMES: Final[Mapping[int, str]] = {code: name for name, code in EVENT_CODES.items()}
SPLIT_CODES: Final[Mapping[str, int]] = {"train": 0, "val": 1, "test": 2}
Split = Literal["train", "val", "test"]

FRAME_FIELDS: Final[Mapping[str, type[np.generic]]] = {
    "clip": np.int32,
    "frame_index": np.int32,
    "pts": np.int64,
    "annotated": np.bool_,
    "is_target": np.bool_,
    "segment_break": np.bool_,
    "event": np.uint8,
    "offset": np.int64,
    "length": np.int64,
    "inst_start": np.int64,
    "inst_count": np.int32,
}
INSTANCE_FIELDS: Final[Mapping[str, type[np.generic]]] = {
    "track_index": np.int32,
    "point_kind": np.uint8,
    "occluded": np.bool_,
}
INSTANCE_XY: Final = "xy"


def shard_name(clip_index: int) -> str:
    return f"clip-{clip_index:05d}.bin"


def located_codes() -> NDArray[np.uint8]:
    return np.asarray(sorted(POINT_KIND_CODES[name] for name in LOCATED_POINT_KINDS), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """One camera frame sequence; ``index`` is the value stored in the frame table.

    ``group_id`` is the split unit of the source (TrackNet game, Meiji video,
    YouTube source video); ``camera_id`` is set only for multi-camera sources.
    ``time_base`` converts the frame table's ``pts`` to seconds and ``fps`` is
    the nominal frame rate, both as exact fraction strings.
    """

    index: int
    clip_id: str
    source: Source
    group_id: str
    camera_id: str | None
    split: Split
    width: int
    height: int
    source_width: int
    source_height: int
    frame_count: int
    time_base: str
    fps: str
    has_events: bool
    track_ids: tuple[str, ...]
    annotation_sha256: str
    media_sha256: str

    @property
    def scale(self) -> float:
        """Stored-pixel per source-pixel factor (identical on both axes)."""
        return self.width / self.source_width


@dataclass(frozen=True, slots=True)
class FrameInstances:
    """Ball annotations of one stored frame, in stored-image pixels.

    ``xy`` is NaN exactly for ``unresolved`` and ``out_of_frame`` instances.
    """

    track_index: NDArray[np.int32]
    point_kind: NDArray[np.uint8]
    xy: NDArray[np.float32]
    occluded: NDArray[np.bool_]


def _require_columns(
    columns: Mapping[str, NDArray[np.generic]],
    fields: Mapping[str, type[np.generic]],
    *,
    extra: Mapping[str, tuple[type[np.generic], tuple[int, ...]]],
    table: str,
) -> int:
    expected = set(fields) | set(extra)
    lengths = set()
    for name in sorted(expected):
        if name not in columns:
            raise ValueError(f"Ball store {table} table is missing column {name!r}")
        column = columns[name]
        dtype, trailing = extra[name] if name in extra else (fields[name], ())
        if column.dtype != np.dtype(dtype) or column.shape[1:] != trailing:
            raise ValueError(
                f"Ball store column {name!r} must be {np.dtype(dtype)} with "
                f"trailing shape {trailing}, got {column.dtype} {column.shape}"
            )
        lengths.add(int(column.shape[0]))
    if len(lengths) != 1:
        raise ValueError(f"Ball store {table} columns have inconsistent lengths")
    return lengths.pop()


def _clip_record(clip: Mapping[str, object]) -> ClipRecord:
    camera = clip["camera_id"]
    return ClipRecord(
        index=int(cast(int, clip["index"])),
        clip_id=str(clip["clip_id"]),
        source=cast(Source, clip["source"]),
        group_id=str(clip["group_id"]),
        camera_id=None if camera is None else str(camera),
        split=cast(Split, clip["split"]),
        width=int(cast(int, clip["width"])),
        height=int(cast(int, clip["height"])),
        source_width=int(cast(int, clip["source_width"])),
        source_height=int(cast(int, clip["source_height"])),
        frame_count=int(cast(int, clip["frame_count"])),
        time_base=str(clip["time_base"]),
        fps=str(clip["fps"]),
        has_events=bool(clip["has_events"]),
        track_ids=tuple(str(value) for value in cast(list[object], clip["track_ids"])),
        annotation_sha256=str(clip["annotation_sha256"]),
        media_sha256=str(clip["media_sha256"]),
    )


class BallFrameStore:
    """Read-only accessor for one generated ball frame store."""

    def __init__(self, directory: Path) -> None:
        if not directory.is_absolute():
            raise ValueError(f"Ball store directory must be absolute: {directory}")
        self.directory = directory
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ball store schema {metadata['schema_version']!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        self.metadata: dict[str, object] = metadata
        self.clips: tuple[ClipRecord, ...] = tuple(_clip_record(clip) for clip in metadata["clips"])
        with np.load(directory / INDEX_FILE) as data:
            columns = {name: data[name] for name in data.files}
        self.frames = {name: columns[name] for name in FRAME_FIELDS}
        self.instances = {name: columns[name] for name in (*INSTANCE_FIELDS, INSTANCE_XY)}
        self.clip_start: NDArray[np.int64] = self._validate(columns)
        self._shards: dict[int, np.memmap] = {}

    def _validate(self, columns: Mapping[str, NDArray[np.generic]]) -> NDArray[np.int64]:
        frame_count = _require_columns(columns, FRAME_FIELDS, extra={}, table="frame")
        instance_count = _require_columns(
            columns,
            INSTANCE_FIELDS,
            extra={INSTANCE_XY: (np.float32, (2,))},
            table="instance",
        )
        if frame_count == 0:
            raise ValueError("Ball store contains no frames")
        if [clip.index for clip in self.clips] != list(range(len(self.clips))):
            raise ValueError("Ball store clip indices must be dense from zero")
        if len({clip.clip_id for clip in self.clips}) != len(self.clips):
            raise ValueError("Ball store clip ids must be unique")
        for clip in self.clips:
            if clip.source not in SOURCES or clip.split not in SPLIT_CODES:
                raise ValueError(f"Clip {clip.clip_id} has an unknown source or split")
            if clip.width * clip.source_height != clip.height * clip.source_width:
                raise ValueError(f"Clip {clip.clip_id} was resized anisotropically")
            if Fraction(clip.time_base) <= 0 or Fraction(clip.fps) <= 0:
                raise ValueError(f"Clip {clip.clip_id} needs a positive time base and fps")
        # Every frame of every clip is stored contiguously, in clip order.
        lengths = np.asarray([clip.frame_count for clip in self.clips], dtype=np.int64)
        if int(lengths.sum()) != frame_count or (lengths <= 0).any():
            raise ValueError("Ball store frame table does not cover every clip frame")
        clip_start = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        expected_clip: NDArray[np.int32] = np.repeat(np.arange(len(self.clips), dtype=np.int32), lengths)
        if not np.array_equal(self.frames["clip"], expected_clip):
            raise ValueError("Ball store frames are not grouped by clip in clip order")
        expected_index = np.arange(frame_count, dtype=np.int64) - np.repeat(clip_start, lengths)
        if not np.array_equal(self.frames["frame_index"], expected_index):
            raise ValueError("Ball store frame indices are not dense within each clip")
        pts_step = np.diff(self.frames["pts"])
        within_clip = self.frames["clip"][1:] == self.frames["clip"][:-1]
        if (pts_step[within_clip] <= 0).any():
            raise ValueError("Ball store pts must strictly increase within a clip")
        if not np.isin(self.frames["event"], list(EVENT_NAMES)).all():
            raise ValueError("Ball store contains an unknown event code")
        starts = self.frames["inst_start"]
        counts = self.frames["inst_count"]
        if (counts < 0).any():
            raise ValueError("Ball store instance counts must be non-negative")
        expected_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        if not np.array_equal(starts, expected_starts) or int(counts.sum()) != instance_count:
            raise ValueError("Ball store instance ranges are not contiguous")
        kinds = self.instances["point_kind"]
        if not np.isin(kinds, list(POINT_KIND_NAMES)).all():
            raise ValueError("Ball store contains an unknown point_kind")
        xy = self.instances[INSTANCE_XY]
        located = np.isin(kinds, located_codes())
        if not np.isnan(xy[~located]).all() or not np.isfinite(xy[located]).all():
            raise ValueError("Exactly the unresolved/out_of_frame balls must have NaN positions")
        instance_clips: NDArray[np.int32] = np.repeat(self.frames["clip"], counts)
        sizes = np.asarray([[clip.width, clip.height] for clip in self.clips], dtype=np.float32)
        limits = sizes[instance_clips]
        if ((xy[located] < 0) | (xy[located] >= limits[located])).any():
            raise ValueError("Ball store contains a located ball outside its stored image")
        track_limits = np.asarray([len(clip.track_ids) for clip in self.clips])
        track_index = self.instances["track_index"]
        if (track_index < 0).any() or (track_index >= track_limits[instance_clips]).any():
            raise ValueError("Ball store instance references an unknown track")
        for clip in self.clips:
            rows = slice(int(clip_start[clip.index]), int(clip_start[clip.index]) + clip.frame_count)
            if not clip.has_events and (self.frames["event"][rows] != EVENT_CODES["unlabeled"]).any():
                raise ValueError(f"Clip {clip.clip_id} has events but is declared without them")
            shard = self.directory / SHARDS_DIR / shard_name(clip.index)
            offsets = self.frames["offset"][rows]
            ends = offsets + self.frames["length"][rows]
            if offsets[0] != 0 or not np.array_equal(offsets[1:], ends[:-1]):
                raise ValueError(f"Shard of {clip.clip_id} is not contiguous")
            if not shard.is_file() or int(ends[-1]) != shard.stat().st_size:
                raise ValueError(f"Shard size does not match the index: {shard}")
        return clip_start

    def __len__(self) -> int:
        return int(self.frames["clip"].shape[0])

    def clip_of(self, row: int) -> ClipRecord:
        return self.clips[int(self.frames["clip"][row])]

    def clip_by_id(self, clip_id: str) -> ClipRecord:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"Ball store has no clip {clip_id!r}")

    def clip_rows(self, clip: ClipRecord) -> NDArray[np.int64]:
        """Stored rows of one clip in presentation order."""
        start = int(self.clip_start[clip.index])
        return np.arange(start, start + clip.frame_count, dtype=np.int64)

    def row_of(self, clip: ClipRecord, frame_index: int) -> int:
        if not 0 <= frame_index < clip.frame_count:
            raise IndexError(f"{clip.clip_id} has no frame {frame_index}")
        return int(self.clip_start[clip.index]) + frame_index

    def split_clips(
        self, split: Split, sources: Sequence[Source] | None = None
    ) -> tuple[ClipRecord, ...]:
        """Clips of one split, optionally restricted to the given sources."""
        if sources is not None:
            unknown = sorted(set(sources) - set(SOURCES))
            if unknown:
                raise ValueError(f"Unknown ball store source(s): {unknown}")
        return tuple(
            clip
            for clip in self.clips
            if clip.split == split and (sources is None or clip.source in sources)
        )

    def instances_of(self, row: int) -> FrameInstances:
        start = int(self.frames["inst_start"][row])
        stop = start + int(self.frames["inst_count"][row])
        return FrameInstances(
            track_index=self.instances["track_index"][start:stop],
            point_kind=self.instances["point_kind"][start:stop],
            xy=self.instances[INSTANCE_XY][start:stop],
            occluded=self.instances["occluded"][start:stop],
        )

    def _shard(self, clip_index: int) -> np.memmap:
        shard = self._shards.get(clip_index)
        if shard is None:
            path = self.directory / SHARDS_DIR / shard_name(clip_index)
            shard = np.memmap(path, dtype=np.uint8, mode="r")
            self._shards[clip_index] = shard
        return shard

    def read_jpeg(self, row: int) -> NDArray[np.uint8]:
        """The stored JPEG bytes of one frame (a view of the shard)."""
        clip = self.clip_of(row)
        offset = int(self.frames["offset"][row])
        length = int(self.frames["length"][row])
        return np.asarray(self._shard(clip.index)[offset : offset + length])

    def read_bgr(self, row: int) -> NDArray[np.uint8]:
        """Decode one stored frame as a BGR ``uint8 (H,W,3)`` image."""
        clip = self.clip_of(row)
        image = cv2.imdecode(self.read_jpeg(row), cv2.IMREAD_COLOR)
        if image is None or image.shape != (clip.height, clip.width, 3):
            raise ValueError(
                f"Frame {row} of {clip.clip_id} failed to decode to "
                f"{(clip.height, clip.width, 3)}"
            )
        return cast(NDArray[np.uint8], image)

    def frame_key(self, row: int) -> str:
        return f"{self.clip_of(row).clip_id}:{int(self.frames['frame_index'][row])}"

    def __getstate__(self) -> dict[str, object]:
        # Memory maps are per process; workers reopen them lazily.
        state = dict(self.__dict__)
        state["_shards"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)


def split_names(values: Sequence[str]) -> tuple[Split, ...]:
    unknown = sorted(set(values) - set(SPLIT_CODES))
    if unknown:
        raise ValueError(f"Unknown split name(s): {unknown}")
    return tuple(cast(Split, value) for value in values)

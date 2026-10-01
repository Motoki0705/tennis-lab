"""Memory-mapped player frame store built from chat-annotation player labels.

Layout of one dataset version (``data/player_detection/<version>/``)::

    shards/clip-00000.bin   # concatenated JPEG bytes of one clip's player frames
    index.npz               # columnar frame and instance tables (below)
    metadata.json           # schema, clips (identity, split, track names), build
    README.md               # generated human summary

Only frames whose annotation lists at least one player are stored. Frames keep
their clip-local ``frame_index`` and the clip's temporal metadata so that
consumers (detection now, ID tracking later) can recover gaps and ordering.
Per-instance ``track_index`` values index the clip's ``track_ids`` list in
``metadata.json``; they are clip-local identities from the annotation and are
never shared across clips.

Frames are addressed through ``np.memmap`` views of the shard files, opened
lazily per process so the store is safe in forked ``DataLoader`` workers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION: Final = "player_detection_frames.v1"
INDEX_FILE: Final = "index.npz"
METADATA_FILE: Final = "metadata.json"
SHARDS_DIR: Final = "shards"

BBOX_SOURCE_CODES: Final[Mapping[str, int]] = {
    "observed": 0,
    "inferred": 1,
    "unresolved": 2,
}
BBOX_SOURCE_NAMES: Final[Mapping[int, str]] = {
    code: name for name, code in BBOX_SOURCE_CODES.items()
}
SPLIT_CODES: Final[Mapping[str, int]] = {"train": 0, "val": 1, "test": 2}
SPLIT_NAMES: Final[Mapping[int, str]] = {code: name for name, code in SPLIT_CODES.items()}
Split = Literal["train", "val", "test"]

FRAME_FIELDS: Final[Mapping[str, type[np.generic]]] = {
    "clip": np.int32,
    "frame_index": np.int32,
    "source_frame_index": np.int32,
    "clip_pts": np.int64,
    "is_target": np.bool_,
    "reviewed": np.bool_,
    "offset": np.int64,
    "length": np.int64,
    "inst_start": np.int64,
    "inst_count": np.int32,
}
INSTANCE_FIELDS: Final[Mapping[str, type[np.generic]]] = {
    "track_index": np.int32,
    "bbox_source": np.uint8,
    "occluded": np.bool_,
    "truncated": np.bool_,
}


def shard_name(clip_index: int) -> str:
    return f"clip-{clip_index:05d}.bin"


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """One source clip; ``index`` is the value stored in the frame table."""

    index: int
    clip_id: str
    source_id: str
    split: Split
    width: int
    height: int
    frame_count: int
    time_base: str
    nominal_fps: str
    track_ids: tuple[str, ...]
    annotation_status: str
    annotation_sha256: str
    video_sha256: str


@dataclass(frozen=True, slots=True)
class FrameInstances:
    """Player annotations of one stored frame, in original image pixels.

    ``boxes_xyxy`` is NaN for ``unresolved`` players (present, not locatable)
    and may extend outside the image for truncated amodal boxes.
    """

    track_index: NDArray[np.int32]
    boxes_xyxy: NDArray[np.float32]
    bbox_source: NDArray[np.uint8]
    occluded: NDArray[np.bool_]
    truncated: NDArray[np.bool_]


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
            raise ValueError(f"Player store {table} table is missing column {name!r}")
        column = columns[name]
        dtype, trailing = extra[name] if name in extra else (fields[name], ())
        if column.dtype != np.dtype(dtype) or column.shape[1:] != trailing:
            raise ValueError(
                f"Player store column {name!r} must be {np.dtype(dtype)} with "
                f"trailing shape {trailing}, got {column.dtype} {column.shape}"
            )
        lengths.add(int(column.shape[0]))
    if len(lengths) != 1:
        raise ValueError(f"Player store {table} columns have inconsistent lengths")
    return lengths.pop()


class PlayerFrameStore:
    """Read-only accessor for one generated player frame store."""

    def __init__(self, directory: Path) -> None:
        if not directory.is_absolute():
            raise ValueError(f"Player store directory must be absolute: {directory}")
        self.directory = directory
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported player store schema {metadata['schema_version']!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        self.metadata: dict[str, object] = metadata
        self.clips: tuple[ClipRecord, ...] = tuple(
            ClipRecord(
                index=int(clip["index"]),
                clip_id=str(clip["clip_id"]),
                source_id=str(clip["source_id"]),
                split=cast(Split, clip["split"]),
                width=int(clip["width"]),
                height=int(clip["height"]),
                frame_count=int(clip["frame_count"]),
                time_base=str(clip["time_base"]),
                nominal_fps=str(clip["nominal_fps"]),
                track_ids=tuple(str(value) for value in clip["track_ids"]),
                annotation_status=str(clip["annotation_status"]),
                annotation_sha256=str(clip["annotation_sha256"]),
                video_sha256=str(clip["video_sha256"]),
            )
            for clip in metadata["clips"]
        )
        with np.load(directory / INDEX_FILE) as data:
            columns = {name: data[name] for name in data.files}
        self.frames = {name: columns[name] for name in FRAME_FIELDS}
        self.instances = {
            name: columns[name] for name in (*INSTANCE_FIELDS, "bbox_xyxy")
        }
        self._validate(columns)
        self._shards: dict[int, np.memmap] = {}

    def _validate(self, columns: Mapping[str, NDArray[np.generic]]) -> None:
        frame_count = _require_columns(columns, FRAME_FIELDS, extra={}, table="frame")
        instance_count = _require_columns(
            columns,
            INSTANCE_FIELDS,
            extra={"bbox_xyxy": (np.float32, (4,))},
            table="instance",
        )
        if frame_count == 0:
            raise ValueError("Player store contains no frames")
        if [clip.index for clip in self.clips] != list(range(len(self.clips))):
            raise ValueError("Player store clip indices must be dense from zero")
        if any(clip.split not in SPLIT_CODES for clip in self.clips):
            raise ValueError("Player store contains an unknown split")
        clip_column = self.frames["clip"]
        if clip_column.min() < 0 or clip_column.max() >= len(self.clips):
            raise ValueError("Player store frame references an unknown clip")
        starts = self.frames["inst_start"]
        counts = self.frames["inst_count"]
        if (counts <= 0).any():
            raise ValueError("Every stored frame must contain at least one player")
        expected_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        if not np.array_equal(starts, expected_starts) or int(counts.sum()) != instance_count:
            raise ValueError("Player store instance ranges are not contiguous")
        sources = self.instances["bbox_source"]
        if not np.isin(sources, list(BBOX_SOURCE_NAMES)).all():
            raise ValueError("Player store contains an unknown bbox_source")
        boxes = self.instances["bbox_xyxy"]
        unresolved = sources == BBOX_SOURCE_CODES["unresolved"]
        if not np.isnan(boxes[unresolved]).all() or not np.isfinite(boxes[~unresolved]).all():
            raise ValueError("Only unresolved players may (and must) have NaN boxes")
        located = boxes[~unresolved]
        if ((located[:, 2] <= located[:, 0]) | (located[:, 3] <= located[:, 1])).any():
            raise ValueError("Player store contains a non-positive-area box")
        instance_clips: NDArray[np.int32] = np.repeat(clip_column, counts)
        track_limits = np.asarray([len(clip.track_ids) for clip in self.clips])
        track_index = self.instances["track_index"]
        if (track_index < 0).any() or (track_index >= track_limits[instance_clips]).any():
            raise ValueError("Player store instance references an unknown track")
        for clip in self.clips:
            rows = np.flatnonzero(clip_column == clip.index)
            if rows.size == 0:
                raise ValueError(f"Clip {clip.clip_id} has no stored frames")
            frame_index = self.frames["frame_index"][rows]
            if (np.diff(frame_index) <= 0).any() or frame_index[-1] >= clip.frame_count:
                raise ValueError(f"Clip {clip.clip_id} frames are not strictly ordered")
            shard = self.directory / SHARDS_DIR / shard_name(clip.index)
            end = self.frames["offset"][rows] + self.frames["length"][rows]
            if not shard.is_file() or int(end.max()) != shard.stat().st_size:
                raise ValueError(f"Shard size does not match the index: {shard}")

    def __len__(self) -> int:
        return int(self.frames["clip"].shape[0])

    def clip_of(self, frame: int) -> ClipRecord:
        return self.clips[int(self.frames["clip"][frame])]

    def split_frames(self, split: Split) -> NDArray[np.int64]:
        """Stored frame rows of one split, in clip then frame order."""
        clips = [clip.index for clip in self.clips if clip.split == split]
        return np.flatnonzero(np.isin(self.frames["clip"], clips)).astype(np.int64)

    def clip_frames(self, clip: ClipRecord) -> NDArray[np.int64]:
        """Stored rows of one clip in presentation order (tracking sequences)."""
        return np.flatnonzero(self.frames["clip"] == clip.index).astype(np.int64)

    def instances_of(self, frame: int) -> FrameInstances:
        start = int(self.frames["inst_start"][frame])
        stop = start + int(self.frames["inst_count"][frame])
        return FrameInstances(
            track_index=self.instances["track_index"][start:stop],
            boxes_xyxy=self.instances["bbox_xyxy"][start:stop],
            bbox_source=self.instances["bbox_source"][start:stop],
            occluded=self.instances["occluded"][start:stop],
            truncated=self.instances["truncated"][start:stop],
        )

    def _shard(self, clip_index: int) -> np.memmap:
        shard = self._shards.get(clip_index)
        if shard is None:
            path = self.directory / SHARDS_DIR / shard_name(clip_index)
            shard = np.memmap(path, dtype=np.uint8, mode="r")
            self._shards[clip_index] = shard
        return shard

    def read_bgr(self, frame: int) -> NDArray[np.uint8]:
        """Decode one stored frame as a BGR ``uint8 (H,W,3)`` image."""
        clip = self.clip_of(frame)
        offset = int(self.frames["offset"][frame])
        length = int(self.frames["length"][frame])
        encoded = np.asarray(self._shard(clip.index)[offset : offset + length])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.shape != (clip.height, clip.width, 3):
            raise ValueError(
                f"Frame {frame} of {clip.clip_id} failed to decode to "
                f"{(clip.height, clip.width, 3)}"
            )
        return cast(NDArray[np.uint8], image)

    def frame_key(self, frame: int) -> str:
        return f"{self.clip_of(frame).clip_id}:{int(self.frames['frame_index'][frame])}"

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

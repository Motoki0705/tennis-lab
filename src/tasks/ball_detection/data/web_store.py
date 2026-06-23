"""Storage backend for the unified web ball-detection frame dataset.

The web datasets under ``data/tennis/web`` are converted into a single,
IO- and storage-efficient store consumed by :mod:`web_datamodule`.

Layout (under ``data_dir``, e.g. ``data/tennis/web/unified``)::

    unified/
    ├── shards/shard-00000.bin   # concatenated JPEG bytes (video-extracted frames)
    ├── index.npz                # columnar per-sample / per-instance arrays
    ├── index_strings.json       # source names + referenced file paths
    ├── manifest.json            # human-readable summary
    └── README.md

Design rationale:

* Frames extracted from videos are packed into a few large ``shard-*.bin``
  files instead of tens of thousands of loose JPEGs. Random access is a
  memory-mapped slice (``offset:offset+length``) decoded with OpenCV, which
  avoids per-file open() overhead and inode/block padding waste.
* COCO still images already exist on disk, so they are *referenced in place*
  (``store == STORE_FILE``) rather than duplicated into shards.
* Every sample carries a sequence id, split, frame index, and explicit label
  state. Positive and explicitly annotated negative frames are retained;
  frames whose annotation state is unknown are not written to the store.
* Sequence ids are the split unit. Augmented variants and frames from the same
  video therefore cannot leak across train/validation/test.
"""

from __future__ import annotations

import json
import mmap
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.tasks.ball_detection.data.types import FrameLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = "web_ball_frames_v2"

STORE_SHARD = 0
STORE_FILE = 1

LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1
LABEL_UNKNOWN = 2
LABEL_NAMES = {
    LABEL_NEGATIVE: "negative",
    LABEL_POSITIVE: "positive",
    LABEL_UNKNOWN: "unknown",
}

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
SPLIT_NAMES = {code: name for name, code in SPLIT_CODES.items()}

INDEX_FILE = "index.npz"
STRINGS_FILE = "index_strings.json"
MANIFEST_FILE = "manifest.json"
SHARDS_DIR = "shards"

# Per-sample columns persisted in ``index.npz``.
SAMPLE_FIELDS: dict[str, type[np.generic]] = {
    "store": np.uint8,
    "shard": np.int32,
    "offset": np.int64,
    "length": np.int64,
    "path_id": np.int32,
    "orig_w": np.int32,
    "orig_h": np.int32,
    "temporal": np.uint8,
    "split": np.uint8,
    "source_id": np.int32,
    "sequence_id": np.int32,
    "frame_index": np.int32,
    "label_state": np.uint8,
    "inst_start": np.int64,
    "inst_count": np.int32,
}

# Per-instance (per ball) columns persisted in ``index.npz``.
INSTANCE_FIELDS: dict[str, type[np.generic]] = {
    "inst_x": np.float32,
    "inst_y": np.float32,
    "inst_vis": np.uint8,
}


def shard_name(shard_id: int) -> str:
    """Return the file name for a shard id."""
    return f"shard-{shard_id:05d}.bin"


class WebFrameStore:
    """Read-only accessor for a converted web ball-detection store.

    The numeric index is loaded eagerly (a few MB). Shard memory maps and file
    handles are opened lazily on first access *in the current process*, which
    keeps the store safe to share across forked ``DataLoader`` workers.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser()
        index_path = self.data_dir / INDEX_FILE
        strings_path = self.data_dir / STRINGS_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"Web frame index not found: {index_path}. Run "
                "`python -m src.tasks.ball_detection.scripts.convert_web_dataset` "
                "to build the unified store."
            )
        with np.load(index_path) as data:
            self._columns = {key: data[key] for key in data.files}
        strings = json.loads(strings_path.read_text(encoding="utf-8"))
        self.sources: list[str] = list(strings.get("sources", []))
        self.sequences: list[str] = list(strings.get("sequences", []))
        self.paths: list[str] = list(strings.get("paths", []))
        self.schema_version: str = str(strings.get("schema", SCHEMA_VERSION))

        self._validate()

        # Lazily-opened, per-process resources.
        self._mmaps: dict[int, mmap.mmap] = {}
        self._files: dict[int, object] = {}
        self._lock = threading.Lock()
        self._decode_cache: tuple[int, np.ndarray] | None = None

    def _validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported web store schema {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}."
            )
        missing = [name for name in SAMPLE_FIELDS if name not in self._columns]
        missing += [name for name in INSTANCE_FIELDS if name not in self._columns]
        if missing:
            raise ValueError(f"Web store index is missing columns: {missing}")
        sample_count = int(self._columns["store"].shape[0])
        bad_sample_lengths = {
            name: int(self._columns[name].shape[0])
            for name in SAMPLE_FIELDS
            if int(self._columns[name].shape[0]) != sample_count
        }
        if bad_sample_lengths:
            raise ValueError(
                "Web store sample columns have inconsistent lengths: "
                f"expected={sample_count}, actual={bad_sample_lengths}."
            )
        instance_count = int(self._columns["inst_x"].shape[0])
        bad_instance_lengths = {
            name: int(self._columns[name].shape[0])
            for name in INSTANCE_FIELDS
            if int(self._columns[name].shape[0]) != instance_count
        }
        if bad_instance_lengths:
            raise ValueError(
                "Web store instance columns have inconsistent lengths: "
                f"expected={instance_count}, actual={bad_instance_lengths}."
            )
        if sample_count == 0:
            raise ValueError("Web store contains no samples.")
        if np.any(self._columns["source_id"] < 0) or np.any(
            self._columns["source_id"] >= len(self.sources)
        ):
            raise ValueError("Web store contains an invalid source_id.")
        if np.any(self._columns["sequence_id"] < 0) or np.any(
            self._columns["sequence_id"] >= len(self.sequences)
        ):
            raise ValueError("Web store contains an invalid sequence_id.")
        if not np.isin(self._columns["split"], list(SPLIT_NAMES)).all():
            raise ValueError("Web store contains an invalid split code.")
        starts = self._columns["inst_start"]
        counts = self._columns["inst_count"]
        if np.any(starts < 0) or np.any(counts < 0):
            raise ValueError("Web store contains negative instance bounds.")
        if np.any(starts + counts > instance_count):
            raise ValueError("Web store instance bounds exceed the instance table.")
        label_states = self._columns["label_state"]
        if not np.isin(label_states, list(LABEL_NAMES)).all():
            raise ValueError("Web store contains an invalid label_state.")
        positive = label_states == LABEL_POSITIVE
        negative = label_states == LABEL_NEGATIVE
        if np.any(positive & (self._columns["inst_count"] <= 0)):
            raise ValueError("Positive web samples must contain an instance.")
        if np.any(negative & (self._columns["inst_count"] != 0)):
            raise ValueError("Negative web samples must not contain instances.")

        sequence_splits: dict[int, int] = {}
        for sequence_id, split in zip(
            self._columns["sequence_id"].tolist(),
            self._columns["split"].tolist(),
            strict=True,
        ):
            existing = sequence_splits.setdefault(int(sequence_id), int(split))
            if existing != int(split):
                raise ValueError(
                    f"Web sequence_id={sequence_id} spans multiple splits."
                )

    def __len__(self) -> int:
        return int(self._columns["store"].shape[0])

    # -- selection -------------------------------------------------------

    def split_indices(
        self,
        split: str,
        *,
        temporal_only: bool = False,
        sources: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return sample indices for a split with optional filters."""
        if split not in SPLIT_CODES:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {sorted(SPLIT_CODES)}."
            )
        mask = self._columns["split"] == SPLIT_CODES[split]
        if temporal_only:
            mask &= self._columns["temporal"] == 1
        if sources is not None:
            wanted = {
                self.sources.index(name) for name in sources if name in self.sources
            }
            if not wanted:
                raise ValueError(
                    f"None of sources={list(sources)} exist in store "
                    f"sources={self.sources}."
                )
            mask &= np.isin(self._columns["source_id"], list(wanted))
        indices: np.ndarray = np.nonzero(mask)[0].astype(np.int64)
        return indices

    def temporal_windows(
        self,
        split: str,
        *,
        num_frames: int,
        frame_step: int = 1,
        sample_stride: int = 1,
        max_frame_gap: int | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[tuple[int, ...]]:
        """Build ordered windows from explicitly labeled temporal samples.

        ``frame_step`` is measured in stored labeled observations rather than
        source-video FPS. RoPE consumers can therefore use the local order
        ``0..T-1`` independently of the original frame indices.
        """
        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        if frame_step <= 0:
            raise ValueError("frame_step must be positive.")
        if sample_stride <= 0:
            raise ValueError("sample_stride must be positive.")
        if max_frame_gap is not None and max_frame_gap <= 0:
            raise ValueError("max_frame_gap must be positive when set.")

        indices = self.split_indices(
            split,
            temporal_only=True,
            sources=sources,
        )
        by_sequence: dict[int, list[int]] = {}
        for index in indices.tolist():
            sequence_id = int(self._columns["sequence_id"][index])
            by_sequence.setdefault(sequence_id, []).append(index)

        windows: list[tuple[int, ...]] = []
        span = (num_frames - 1) * frame_step + 1
        for sequence_indices in by_sequence.values():
            ordered = sorted(
                sequence_indices,
                key=lambda index: int(self._columns["frame_index"][index]),
            )
            if len(ordered) < span:
                continue
            for start in range(0, len(ordered) - span + 1, sample_stride):
                window = tuple(
                    ordered[start + offset * frame_step] for offset in range(num_frames)
                )
                if max_frame_gap is not None:
                    frame_indices = [
                        int(self._columns["frame_index"][index]) for index in window
                    ]
                    if any(
                        later - earlier > max_frame_gap
                        for earlier, later in zip(
                            frame_indices,
                            frame_indices[1:],
                            strict=False,
                        )
                    ):
                        continue
                windows.append(window)
        return windows

    # -- per-sample metadata --------------------------------------------

    def original_size(self, index: int) -> tuple[int, int]:
        """Return ``(width, height)`` of the original frame."""
        return (
            int(self._columns["orig_w"][index]),
            int(self._columns["orig_h"][index]),
        )

    def temporal(self, index: int) -> bool:
        """Return whether the sample came from a temporally-ordered source."""
        return bool(self._columns["temporal"][index])

    def source_name(self, index: int) -> str:
        """Return the originating dataset name."""
        return self.sources[int(self._columns["source_id"][index])]

    def sequence_name(self, index: int) -> str:
        """Return the split-safe source group or video sequence id."""
        return self.sequences[int(self._columns["sequence_id"][index])]

    def frame_index(self, index: int) -> int:
        """Return the original frame index, or ``-1`` for unordered stills."""
        return int(self._columns["frame_index"][index])

    def label_state(self, index: int) -> int:
        """Return one of ``LABEL_NEGATIVE/POSITIVE/UNKNOWN``."""
        return int(self._columns["label_state"][index])

    def is_positive(self, index: int) -> bool:
        """Return whether this sample has at least one visible ball."""
        return self.label_state(index) == LABEL_POSITIVE

    def labels(self, index: int) -> tuple[FrameLabel, ...]:
        """Return the ball annotations for a sample."""
        start = int(self._columns["inst_start"][index])
        count = int(self._columns["inst_count"][index])
        xs = self._columns["inst_x"]
        ys = self._columns["inst_y"]
        vis = self._columns["inst_vis"]
        return tuple(
            FrameLabel(
                visibility=float(vis[start + offset]),
                x=float(xs[start + offset]),
                y=float(ys[start + offset]),
                instance_id=f"b{offset + 1:03d}",
                role="target",
                state="visible" if vis[start + offset] > 0 else "absent",
            )
            for offset in range(count)
        )

    # -- pixels ----------------------------------------------------------

    def read_jpeg(self, index: int) -> bytes:
        """Return the encoded JPEG bytes for a sample."""
        if int(self._columns["store"][index]) == STORE_SHARD:
            shard_id = int(self._columns["shard"][index])
            offset = int(self._columns["offset"][index])
            length = int(self._columns["length"][index])
            buffer = self._shard_mmap(shard_id)
            return bytes(buffer[offset : offset + length])
        path_id = int(self._columns["path_id"][index])
        path = self.data_dir / self.paths[path_id]
        return path.read_bytes()

    def decode_bgr(self, index: int) -> np.ndarray:
        """Decode a sample to a BGR ``uint8`` array, caching the last decode.

        The caller (a static-clip dataset) reads the same index ``num_frames``
        times in a row, so a one-slot cache removes redundant JPEG decodes.
        """
        cached = self._decode_cache
        if cached is not None and cached[0] == index:
            return cached[1]
        buffer = np.frombuffer(self.read_jpeg(index), dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(f"Failed to decode web frame index={index}.")
        image: np.ndarray = decoded
        self._decode_cache = (index, image)
        return image

    def _shard_mmap(self, shard_id: int) -> mmap.mmap:
        existing = self._mmaps.get(shard_id)
        if existing is not None:
            return existing
        with self._lock:
            existing = self._mmaps.get(shard_id)
            if existing is not None:
                return existing
            path = self.data_dir / SHARDS_DIR / shard_name(shard_id)
            handle = path.open("rb")
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._files[shard_id] = handle
            self._mmaps[shard_id] = mapped
            return mapped

    def close(self) -> None:
        """Release memory maps and file handles held by this process."""
        for mapped in self._mmaps.values():
            mapped.close()
        for handle in self._files.values():
            handle.close()  # type: ignore[attr-defined]
        self._mmaps.clear()
        self._files.clear()
        self._decode_cache = None


__all__ = [
    "INSTANCE_FIELDS",
    "LABEL_NAMES",
    "LABEL_NEGATIVE",
    "LABEL_POSITIVE",
    "LABEL_UNKNOWN",
    "MANIFEST_FILE",
    "SAMPLE_FIELDS",
    "SCHEMA_VERSION",
    "SHARDS_DIR",
    "SPLIT_CODES",
    "SPLIT_NAMES",
    "STORE_FILE",
    "STORE_SHARD",
    "STRINGS_FILE",
    "INDEX_FILE",
    "WebFrameStore",
    "shard_name",
]

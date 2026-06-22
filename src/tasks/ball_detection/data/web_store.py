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
* Every sample carries a ``temporal`` flag (1 if it came from a video with a
  recoverable frame order, 0 for shuffled still images) plus ``frame_index``
  and ``source`` provenance, so a later multi-frame phase can rebuild temporal
  windows without re-deriving the source mapping.
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

SCHEMA_VERSION = "web_ball_frames_v1"

STORE_SHARD = 0
STORE_FILE = 1

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
    "frame_index": np.int32,
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
            wanted = {self.sources.index(name) for name in sources if name in self.sources}
            if not wanted:
                raise ValueError(
                    f"None of sources={list(sources)} exist in store "
                    f"sources={self.sources}."
                )
            mask &= np.isin(self._columns["source_id"], list(wanted))
        indices: np.ndarray = np.nonzero(mask)[0].astype(np.int64)
        return indices

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

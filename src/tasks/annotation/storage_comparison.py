"""Native label meaning stays in adapters; measurement lives in the shared layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.base.annotation.storage.comparison.contracts import StoreSpec
from src.tasks.base.annotation.storage.comparison.sources import StoreReader
from src.tasks.player_detection.data.store import PlayerFrameStore
from src.utils.data.image_record_store import ImageRecordStore


def _array(value: NDArray[np.generic]) -> dict[str, object]:
    # Preserve exact dtype, shape, and NaN payloads without JSON nonfinite values.
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "hex": value.tobytes().hex(),
    }


class NativeFrames:
    def __init__(self, store: BallFrameStore | PlayerFrameStore) -> None:
        self.store = store

    @property
    def metadata(self) -> Mapping[str, object]:
        return self.store.metadata

    @property
    def shards(self) -> tuple[Path, ...]:
        return tuple(
            self.store.directory / "shards" / f"clip-{clip.index:05d}.bin"
            for clip in self.store.clips
        )

    def __len__(self) -> int:
        return len(self.store)

    def key(self, row: int) -> str:
        result: str = self.store.frame_key(row)
        return result

    def labels(self, row: int) -> Mapping[str, object]:
        store = self.store
        start = int(store.frames["inst_start"][row])
        stop = start + int(store.frames["inst_count"][row])
        result: dict[str, object] = {
            "clip": asdict(store.clip_of(row)),
            "frame": {k: _array(v[row : row + 1]) for k, v in store.frames.items()},
            "instances": {k: _array(v[start:stop]) for k, v in store.instances.items()},
        }
        if isinstance(store, BallFrameStore):
            masks = store.teacher_masks
            result["teacher_masks"] = (
                None
                if masks is None
                else {
                    "heatmap": _array(masks.heatmap[row : row + 1]),
                    "coordinate": _array(masks.coordinate[row : row + 1]),
                }
            )
        return result

    def jpeg(self, row: int) -> bytes:
        result: bytes = self.store.read_jpeg(row).tobytes()
        return result

    def rgb(self, row: int) -> NDArray[np.uint8]:
        result: NDArray[np.uint8] = cv2.cvtColor(
            self.store.read_bgr(row), cv2.COLOR_BGR2RGB
        )
        return result


class NativeRecords:
    def __init__(self, store: ImageRecordStore) -> None:
        self.store = store

    @property
    def metadata(self) -> Mapping[str, object]:
        result: Mapping[str, object] = self.store.metadata
        return result

    @property
    def shards(self) -> tuple[Path, ...]:
        return tuple(self.store.paths)

    def __len__(self) -> int:
        return len(self.store)

    def key(self, row: int) -> str:
        return str(row)

    def labels(self, row: int) -> Mapping[str, object]:
        result: Mapping[str, object] = self.store.record(row)
        return result

    def jpeg(self, row: int) -> bytes:
        result: bytes = self.store.jpeg(row)
        return result

    def rgb(self, row: int) -> NDArray[np.uint8]:
        result: NDArray[np.uint8] = self.store.rgb(row)
        return result


def open_store(spec: StoreSpec, root: Path) -> StoreReader:
    if spec.adapter == "image_records":
        if spec.descriptor is None:
            raise ValueError("image_records requires its frozen descriptor")
        return NativeRecords(ImageRecordStore(root, spec.descriptor))
    if spec.descriptor is not None:
        raise ValueError("native frame stores cannot silently ignore a descriptor")
    if spec.adapter == "ball_frames":
        return NativeFrames(BallFrameStore(root))
    if spec.adapter == "player_frames":
        return NativeFrames(PlayerFrameStore(root))
    raise ValueError(f"unknown storage comparison adapter: {spec.adapter}")

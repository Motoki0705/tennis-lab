"""Full-fidelity frame and label access for ball store visualization."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import POINT_KIND_NAMES, BallFrameStore
from src.tasks.ball_detection.data.supervision import (
    OBSERVED_ONLY,
    resolve_frame_supervision,
)
from src.tasks.ball_detection.data.types import FrameLabel


class StoreSceneFrames:
    """Display every label; score only trusted observed-only frames.

    Unknown positions remain invisible and retain their point_kind in ``state``.
    ``annotated`` and ``supervised`` are distinct: a reviewed unresolved frame
    is annotated but cannot be scored as either a positive or a negative.
    """

    mode: Literal["temporal"] = "temporal"

    def __init__(self, store: BallFrameStore, clip_id: str) -> None:
        self.store = store
        self.clip = store.clip_by_id(clip_id)
        self._supervision = resolve_frame_supervision(store, OBSERVED_ONLY)

    @property
    def frames(self) -> int:
        return int(self.clip.frame_count)

    def _row(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Frame index must be an int")
        return int(self.store.row_of(self.clip, index))

    def name(self, index: int) -> str:
        return str(self.store.frame_key(self._row(index)))

    def original_size(self, index: int) -> tuple[int, int]:
        self._row(index)
        return self.clip.width, self.clip.height

    def read_rgb(self, index: int) -> NDArray[np.uint8]:
        # Recheck on access as well as catalog discovery (symlinks can change).
        from src.tasks.ball_detection.data.store import SHARDS_DIR, shard_name

        root = self.store.directory.resolve()
        shard = root / SHARDS_DIR / shard_name(self.clip.index)
        if not shard.resolve().is_relative_to(root):
            raise ValueError(f"Shard {shard} resolves outside the store root")
        return cast(
            NDArray[np.uint8],
            cv2.cvtColor(self.store.read_bgr(self._row(index)), cv2.COLOR_BGR2RGB),
        )

    def labels(self, index: int) -> tuple[FrameLabel, ...]:
        instances = self.store.instances_of(self._row(index))
        labels = []
        for track, kind, xy in zip(
            instances.track_index, instances.point_kind, instances.xy, strict=True
        ):
            located = bool(np.isfinite(xy).all())
            labels.append(
                FrameLabel(
                    x=float(xy[0]) if located else 0.0,
                    y=float(xy[1]) if located else 0.0,
                    visibility=float(located),
                    instance_id=self.clip.track_ids[int(track)],
                    state=POINT_KIND_NAMES[int(kind)],
                )
            )
        return tuple(labels)

    def annotated(self, index: int) -> bool:
        return bool(self.store.frames["annotated"][self._row(index)])

    def supervised(self, index: int) -> bool:
        return bool(self._supervision.supervised[self._row(index)])

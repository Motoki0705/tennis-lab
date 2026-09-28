"""Temporal windows over the unified ball frame store."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.dataset import (
    BallDetectionDataset,
    WindowFrame,
    WindowFrames,
)
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.data.supervision import FrameSupervision

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True, slots=True)
class StoreWindow:
    """``length`` consecutive frames of one store clip starting at ``start``."""

    clip: int
    start: int


@dataclass(frozen=True, slots=True)
class WindowSelection:
    """Windows of one split plus the counts of what was left out, by source."""

    windows: tuple[StoreWindow, ...]
    stats: dict[str, dict[str, int]]


def select_store_windows(
    store: BallFrameStore,
    supervision: FrameSupervision,
    clips: Sequence[ClipRecord],
    *,
    length: int,
    stride: int,
) -> WindowSelection:
    """Every ``stride``-th window of ``length`` frames that has a supervised frame.

    A clip shorter than ``length`` yields no window, and a window whose frames
    are all unsupervised carries no training signal; both are counted in
    ``stats`` so a selection never shrinks silently.
    """
    if length <= 0 or stride <= 0:
        raise ValueError("Window length and stride must be positive")
    windows: list[StoreWindow] = []
    stats: dict[str, dict[str, int]] = {}
    for clip in clips:
        counts = stats.setdefault(
            clip.source,
            {
                "clips": 0,
                "clips_shorter_than_window": 0,
                "frames": 0,
                "supervised_frames": 0,
                "windows": 0,
                "windows_without_supervision": 0,
            },
        )
        counts["clips"] += 1
        rows = store.clip_rows(clip)
        supervised = supervision.supervised[rows]
        counts["frames"] += int(rows.size)
        counts["supervised_frames"] += int(supervised.sum())
        if clip.frame_count < length:
            counts["clips_shorter_than_window"] += 1
            continue
        # Number of supervised frames in each window [start, start + length).
        cumulative = np.concatenate([[0], np.cumsum(supervised, dtype=np.int64)])
        for start in range(0, clip.frame_count - length + 1, stride):
            if cumulative[start + length] - cumulative[start] == 0:
                counts["windows_without_supervision"] += 1
                continue
            windows.append(StoreWindow(clip=clip.index, start=start))
            counts["windows"] += 1
    return WindowSelection(windows=tuple(windows), stats=stats)


class BallStoreDataset(BallDetectionDataset):
    """Samples of ``model.num_frames`` consecutive store frames."""

    def __init__(
        self,
        *,
        store: BallFrameStore,
        supervision: FrameSupervision,
        windows: Sequence[StoreWindow],
        config: DictConfig,
        augmentation: BallDetectionAugmentation | None = None,
    ) -> None:
        super().__init__(config=config, augmentation=augmentation)
        self.store = store
        self.supervision = supervision
        self.windows = tuple(windows)
        if not self.windows:
            raise RuntimeError("No ball store windows were provided.")
        for window in self.windows:
            clip = store.clips[window.clip]
            if window.start < 0 or window.start + self.num_frames > clip.frame_count:
                raise ValueError(
                    f"Window {clip.clip_id}:{window.start} does not hold "
                    f"model.num_frames={self.num_frames} frames"
                )

    def __len__(self) -> int:
        return len(self.windows)

    def source_of(self, index: int) -> str:
        return str(self.store.clips[self.windows[index].clip].source)

    def read_window(self, index: int, num_frames: int) -> WindowFrames:
        window = self.windows[index]
        clip = self.store.clips[window.clip]
        first = self.store.row_of(clip, window.start)
        frames = []
        for row in range(first, first + num_frames):
            supervised = bool(self.supervision.supervised[row])
            points: tuple[tuple[float, float], ...] = ()
            if supervised:
                start = int(self.store.frames["inst_start"][row])
                stop = start + int(self.store.frames["inst_count"][row])
                xy = self.store.instances["xy"][start:stop]
                positive = self.supervision.positive[start:stop]
                points = tuple((float(x), float(y)) for x, y in xy[positive])
            frames.append(WindowFrame(self.store.read_bgr(row), points, supervised))
        return WindowFrames(
            frames=tuple(frames),
            original_size=(clip.width, clip.height),
            window_id=f"{clip.clip_id}:{window.start}",
            source=clip.source,
        )


__all__ = [
    "BallStoreDataset",
    "StoreWindow",
    "WindowSelection",
    "select_store_windows",
]

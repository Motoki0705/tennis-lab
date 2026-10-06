"""Membership and continuity are explicit, independent choices."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.play_intervals import (
    PlayIntervalConfig,
    PlaySelection,
    infer_play_intervals,
    mask_intervals,
)

from .configuration import StatisticsConfig
from .contracts import ClipInput


def selections(data: ClipInput, config: StatisticsConfig) -> dict[int, PlaySelection]:
    s = data.states
    return {stride: infer_play_intervals(s.evidence, s.supervision, s.times, s.target,
                                        replace(PlayIntervalConfig(), window_stride=stride)) for stride in config.strides}


def scope_masks(data: ClipInput, selection: PlaySelection) -> dict[str, NDArray[np.bool_]]:
    return dict(clip=np.ones(data.clip.frame_count, bool), play=selection.play,
                selected=selection.selected, excluded=~selection.selected)


def runs(mask: NDArray[np.bool_], breaks: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    """Half-open runs that never bridge explicitly marked boundaries."""
    result: list[tuple[int, int]] = []
    for a, b in mask_intervals(mask):
        boundaries = [a, *np.flatnonzero(breaks[a + 1:b]) + a + 1, b]
        result.extend((int(x), int(y)) for x, y in zip(boundaries[:-1], boundaries[1:], strict=True))
    return tuple(result)


def edges(data: ClipInput, scope: NDArray[np.bool_], valid: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return scope[1:] & scope[:-1] & valid[1:] & valid[:-1] & ~data.breaks[1:]

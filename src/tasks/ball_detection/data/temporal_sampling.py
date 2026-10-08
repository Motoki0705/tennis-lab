"""Subsample native frames inside approved play spans, before forming MDD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .annotation_states import AnnotationStates
from .play_intervals import PlayIntervalConfig, infer_play_intervals


@dataclass(frozen=True)
class TemporalSamplingConfig:
    frame_steps: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self) -> None:
        if (not self.frame_steps or any(type(step) is not int or step < 1 for step in self.frame_steps)
                or tuple(sorted(set(self.frame_steps))) != self.frame_steps or self.frame_steps[0] != 1):
            raise ValueError("Frame steps must be sorted unique positive integers including native step 1")


@dataclass(frozen=True)
class SampledWindow:
    start: int
    frame_step: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0 or type(self.frame_step) is not int or self.frame_step < 1:
            raise ValueError("Window start and frame step must be nonnegative/positive integers")

    def indices(self, length: int = 32) -> NDArray[np.int64]:
        return self.start + self.frame_step * np.arange(length, dtype=np.int64)


def select_sampled_windows(
    states: AnnotationStates, play_config: PlayIntervalConfig, sampling: TemporalSamplingConfig,
) -> tuple[tuple[tuple[int, int], ...], tuple[SampledWindow, ...]]:
    """Keep 32 inputs at each FPS; start stride is expressed in sampled frames.

    Play evidence and gap bridging are computed once on native PTS. Window
    evidence/teacher counts are then checked on exactly the sampled inputs.
    No skipped frame is used to supply MDD or supervision to a sampled window.
    """
    if play_config.window_length != 32:
        raise ValueError("Coordinate models require 32 sampled frames")
    selection = infer_play_intervals(states.evidence, states.supervision, states.times, states.target, play_config)
    if not selection.intervals:
        return (), ()
    differences = np.diff(states.times)
    cuts = np.flatnonzero(differences > play_config.timestamp_gap_factor * np.median(differences)) + 1
    # A boolean play mask can merge adjacent true spans across a PTS jump.
    # Keep those barriers explicit before selecting a longer, subsampled span.
    spans: list[tuple[int, int]] = []
    for start, stop in selection.intervals:
        bounds = [start, *(int(x) for x in cuts if start < x < stop), stop]
        spans.extend(zip(bounds[:-1], bounds[1:], strict=True))
    windows = []
    for step in sampling.frame_steps:
        span_length = (play_config.window_length - 1) * step + 1
        for start, stop in spans:
            last_start = stop - span_length
            if last_start < start:
                continue
            candidates = list(range(start, last_start + 1, play_config.window_stride * step))
            if candidates[-1] != last_start:
                candidates.append(last_start)
            for first in candidates:
                window = SampledWindow(first, step)
                indices = window.indices(play_config.window_length)
                if (states.evidence[indices].mean() >= play_config.min_presence_fraction
                        and states.supervision[indices].sum() >= play_config.min_observed_frames):
                    windows.append(window)
    return tuple(spans), tuple(windows)

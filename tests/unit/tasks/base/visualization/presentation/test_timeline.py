"""Frame ownership, learning interpolation and timeline input boundaries."""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.tasks.base.visualization.presentation.timeline import Segment, StepClock

pytestmark = pytest.mark.unit


def test_segment_boundaries_and_linear_learning_steps() -> None:
    intro = Segment("intro", 1, 0, 0)
    train = Segment("train", 2, 0, 100)
    handoff = Segment("handoff", 1, 80, 80)
    next_stage = Segment("next", 1, 0, 40)
    clock = StepClock([intro, train, handoff, next_stage], fps=3)
    assert clock.frame_count() == 15
    assert clock.at(0) == (intro, 0.0, 0.0)
    assert clock.at(2) == (intro, 1.0, 0.0)
    assert clock.at(3) == (train, 0.0, 0.0)
    assert clock.at(4) == (train, 0.2, 20.0)
    assert clock.at(8) == (train, 1.0, 100.0)
    assert clock.at(9) == (handoff, 0.0, 80.0)
    assert clock.at(10) == (handoff, 0.5, 80.0)
    assert clock.at(11) == (handoff, 1.0, 80.0)
    assert clock.at(12) == (next_stage, 0.0, 0.0)
    assert clock.at(14) == (next_stage, 1.0, 40.0)


def test_cumulative_rounding_avoids_per_segment_duration_drift() -> None:
    segments = [Segment(str(index), 0.15, 0, 10) for index in range(4)]
    clock = StepClock(segments, fps=10)
    assert clock.frame_count() == 6
    assert [clock.at(index)[0].name for index in range(6)] == [
        "0",
        "0",
        "1",
        "2",
        "2",
        "3",
    ]
    assert clock.at(2) == (segments[1], 1.0, 10.0)
    assert clock.at(5) == (segments[3], 1.0, 10.0)


def test_fractional_fps_and_immutable_segment_sequence() -> None:
    segments = [Segment("train", 1, 0, 100)]
    clock = StepClock(segments, fps=29.97)
    segments.clear()
    assert clock.frame_count() == 30
    assert clock.at(29)[2] == 100


@pytest.mark.parametrize("frame", [-1, 3])
def test_out_of_range_frames_are_rejected(frame: int) -> None:
    clock = StepClock([Segment("stage", 1, 0, 1)], fps=3)
    with pytest.raises(IndexError, match="frame must be in"):
        clock.at(frame)


@pytest.mark.parametrize("frame", [0.5, True, "0"])
def test_non_integer_frames_are_rejected(frame: Any) -> None:
    clock = StepClock([Segment("stage", 1, 0, 1)], fps=3)
    with pytest.raises(TypeError, match="integer"):
        clock.at(frame)


@pytest.mark.parametrize("fps", [0, -1, math.inf, math.nan, True])
def test_invalid_fps_is_rejected(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        StepClock([Segment("stage", 1, 0, 1)], fps=fps)


def test_empty_and_unrepresentable_timelines_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        StepClock([], fps=30)
    with pytest.raises(ValueError, match="zero frames"):
        StepClock([Segment("short", 0.001, 0, 1)], fps=30)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": " "},
        {"video_seconds": 0},
        {"video_seconds": -1},
        {"video_seconds": math.nan},
        {"video_seconds": math.inf},
        {"video_seconds": True},
        {"step_start": -1},
        {"step_start": 1.5},
        {"step_start": True},
        {"step_stop": -1},
        {"step_stop": 0.5},
        {"step_start": 2, "step_stop": 1},
    ],
)
def test_invalid_segments_are_rejected(kwargs: dict[str, Any]) -> None:
    arguments: dict[str, Any] = dict(
        name="train", video_seconds=1, step_start=0, step_stop=1
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        Segment(**arguments)

from __future__ import annotations

import pytest

from src.utils.video.sampling import (
    parse_time_seconds,
    sample_frame_indices_by_time_ranges,
    sample_step_seconds,
    sample_uniform_frame_indices,
)


def test_parse_time_seconds_supports_colon_format() -> None:
    assert parse_time_seconds("01:02:03.5") == 3723.5
    assert parse_time_seconds(12) == 12.0


def test_sample_step_seconds_modes() -> None:
    assert (
        sample_step_seconds(
            sample_mode="interval_seconds",
            fps=30,
            interval_seconds=2,
            target_fps=1,
            every_n_frames=60,
        )
        == 2
    )
    assert (
        sample_step_seconds(
            sample_mode="fps",
            fps=30,
            interval_seconds=2,
            target_fps=2,
            every_n_frames=60,
        )
        == 0.5
    )
    assert (
        sample_step_seconds(
            sample_mode="every_n_frames",
            fps=30,
            interval_seconds=2,
            target_fps=2,
            every_n_frames=15,
        )
        == 0.5
    )


def test_sample_frame_indices_by_time_ranges_deduplicates_and_limits() -> None:
    indices = sample_frame_indices_by_time_ranges(
        [{"start": 0, "end": 1}, {"start": 0.5, "end": 1.5}],
        duration=2,
        fps=2,
        sample_mode="interval_seconds",
        interval_seconds=0.5,
        target_fps=1,
        every_n_frames=1,
        max_frames=4,
    )

    assert indices == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "time_range",
    [
        {"end": 1.0},
        {"start": 0.0},
        {"start": 0.0, "end": 1.0, "legacy_duration": 1.0},
    ],
)
def test_sample_frame_indices_rejects_missing_or_unknown_range_keys(
    time_range: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="time range keys must be exactly"):
        sample_frame_indices_by_time_ranges(
            [time_range],
            duration=2.0,
            fps=2.0,
            sample_mode="interval_seconds",
            interval_seconds=0.5,
            target_fps=1.0,
            every_n_frames=1,
        )


def test_sample_uniform_frame_indices_spans_video() -> None:
    assert sample_uniform_frame_indices(100, 5) == [0, 25, 50, 74, 99]
    assert sample_uniform_frame_indices(3, 10) == [0, 1, 2]


def test_sample_uniform_frame_indices_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        sample_uniform_frame_indices(0, 5)
    with pytest.raises(ValueError):
        sample_uniform_frame_indices(10, 0)

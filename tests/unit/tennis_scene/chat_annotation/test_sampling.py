from __future__ import annotations

from fractions import Fraction

import pytest

from src.tennis_scene.chat_annotation.runtime.contracts import FrameRange
from src.tennis_scene.chat_annotation.runtime.media import Timeline
from src.tennis_scene.chat_annotation.sampling import (
    centered_subrange,
    select_target_ranges,
)


def timeline(count: int, fps: int = 30) -> Timeline:
    return Timeline(
        96, 64, Fraction(1, fps), Fraction(fps), tuple(range(count)), (1,) * count
    )


def test_long_videos_contribute_the_same_five_time_uniform_windows() -> None:
    for seconds in (100, 600, 3600):
        video = timeline(seconds * 30)
        ranges, candidates = select_target_ranges(video, 15, 1, 5, "uniform_midpoints")
        assert candidates > 5 and len(ranges) == 5
        for slot, target in enumerate(ranges):
            center = (video.boundary(target.start) + video.boundary(target.stop)) / 2
            expected = Fraction(seconds * (2 * slot + 1), 10)
            assert abs(center - expected) <= Fraction(1, 30)
            assert video.boundary(target.stop) - video.boundary(target.start) == 13
        assert all(a.stop <= b.start for a, b in zip(ranges, ranges[1:], strict=False))


def test_short_videos_keep_available_targets_without_duplicates() -> None:
    video = timeline(31 * 30)
    ranges, count = select_target_ranges(video, 15, 1, 5, "uniform_midpoints")
    assert count == 3
    assert [(r.start, r.stop) for r in ranges] == [(0, 390), (390, 780), (780, 930)]
    tiny, count = select_target_ranges(timeline(1), 15, 1, 5, "uniform_midpoints")
    assert count == 1 and tiny == [FrameRange(start=0, stop=1)]


def test_single_clip_uses_video_midpoint_and_null_preserves_full_coverage() -> None:
    video = timeline(600 * 30)
    ranges, _ = select_target_ranges(video, 15, 1, 1, "uniform_midpoints")
    assert len(ranges) == 1
    assert (video.boundary(ranges[0].start) + video.boundary(ranges[0].stop)) / 2 == 300
    full, count = select_target_ranges(video, 15, 1, None, "uniform_midpoints")
    assert len(full) == count
    assert [i for r in full for i in range(r.start, r.stop)] == list(range(18000))


def test_vfr_sampling_is_uniform_in_time_not_in_frame_number() -> None:
    stamps = [100000 + p for p in [*range(0, 50000, 1000), *range(50000, 100000, 100)]]
    durations = [b - a for a, b in zip(stamps, stamps[1:], strict=False)] + [100]
    video = Timeline(
        96, 64, Fraction(1, 1000), Fraction(11, 2), tuple(stamps), tuple(durations)
    )
    ranges, _ = select_target_ranges(video, 6, 1, 5, "uniform_midpoints")
    centers = [(video.boundary(r.start) + video.boundary(r.stop)) / 2 for r in ranges]
    assert centers == [10, 30, 50, 70, 90]
    assert [r.stop - r.start for r in ranges] == [4, 4, 22, 40, 40]


def test_capacity_shrinking_stays_on_the_temporal_center() -> None:
    # The center lies inside a long VFR frame, not inside a frame-index median.
    video = Timeline(
        96,
        64,
        Fraction(1),
        Fraction(1),
        (0, 10, 20, 25, 30, 35, 90),
        (10, 10, 5, 5, 5, 55, 10),
    )
    target = FrameRange(start=0, stop=7)
    result = centered_subrange(video, target)
    assert result == FrameRange(start=5, stop=6)
    assert video.boundary(result.start) <= 50 < video.boundary(result.stop)
    with pytest.raises(ValueError, match="one-frame"):
        centered_subrange(video, result)


@pytest.mark.parametrize("maximum", [0, -1, True])
def test_invalid_caps_are_rejected(maximum: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        select_target_ranges(timeline(100), 15, 1, maximum, "uniform_midpoints")

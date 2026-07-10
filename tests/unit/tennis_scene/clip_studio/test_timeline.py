"""Tests for src/tennis_scene/clip_studio/timeline.py."""

import pytest

from src.tennis_scene.clip_studio.timeline import (
    TimelineGeometry,
    format_timecode,
    source_coverage_sec,
    source_frame_index,
    timeline_extent_sec,
)


class TestSourceFrameIndex:
    def test_identity_mapping(self) -> None:
        assert source_frame_index(1.0, offset_sec=0.0, fps=30.0, frame_count=300) == 30

    def test_offset_shifts_local_time(self) -> None:
        # local = global + offset
        assert source_frame_index(1.0, offset_sec=0.5, fps=30.0, frame_count=300) == 45
        assert source_frame_index(1.0, offset_sec=-1.0, fps=30.0, frame_count=300) == 0

    def test_rounds_to_nearest_frame(self) -> None:
        assert source_frame_index(0.016, offset_sec=0.0, fps=30.0, frame_count=10) == 0
        assert source_frame_index(0.017, offset_sec=0.0, fps=30.0, frame_count=10) == 1

    def test_out_of_range_returns_none(self) -> None:
        assert source_frame_index(-0.5, offset_sec=0.0, fps=30.0, frame_count=10) is None
        assert source_frame_index(1.0, offset_sec=0.0, fps=30.0, frame_count=10) is None

    def test_invalid_args_raise(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            source_frame_index(0.0, offset_sec=0.0, fps=0.0, frame_count=10)
        with pytest.raises(ValueError, match="frame_count"):
            source_frame_index(0.0, offset_sec=0.0, fps=30.0, frame_count=-1)


class TestCoverageAndExtent:
    def test_coverage(self) -> None:
        assert source_coverage_sec(offset_sec=-1.0, duration_sec=8.0) == (1.0, 9.0)
        assert source_coverage_sec(offset_sec=2.0, duration_sec=8.0) == (-2.0, 6.0)

    def test_extent_is_union(self) -> None:
        assert timeline_extent_sec([(0.0, 10.0), (1.0, 9.0), (-2.0, 3.0)]) == (-2.0, 10.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            timeline_extent_sec([])


class TestFormatTimecode:
    def test_zero(self) -> None:
        assert format_timecode(0.0) == "0:00:00.000"

    def test_hours_minutes(self) -> None:
        assert format_timecode(3661.5) == "1:01:01.500"

    def test_negative(self) -> None:
        assert format_timecode(-1.25) == "-0:00:01.250"


class TestTimelineGeometry:
    def make(self) -> TimelineGeometry:
        return TimelineGeometry(
            x=10, y=100, width=101, height=40, view_start_sec=0.0, view_end_sec=10.0
        )

    def test_sec_to_x_endpoints(self) -> None:
        geometry = self.make()
        assert geometry.sec_to_x(0.0) == 10
        assert geometry.sec_to_x(10.0) == 110
        assert geometry.sec_to_x(5.0) == 60

    def test_x_to_sec_inverts(self) -> None:
        geometry = self.make()
        for sec in [0.0, 2.5, 9.9]:
            assert geometry.x_to_sec(geometry.sec_to_x(sec)) == pytest.approx(sec, abs=0.1)

    def test_contains(self) -> None:
        geometry = self.make()
        assert geometry.contains(10, 100)
        assert geometry.contains(110, 139)
        assert not geometry.contains(9, 100)
        assert not geometry.contains(10, 140)

    def test_invalid_view_raises(self) -> None:
        with pytest.raises(ValueError, match="view_end_sec"):
            TimelineGeometry(
                x=0, y=0, width=10, height=10, view_start_sec=1.0, view_end_sec=1.0
            )

"""Unit tests for :mod:`src.utils.video.windows`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.utils.video.types import FramePacket
from src.utils.video.windows import iter_temporal_windows


def _frames(n: int) -> list[FramePacket[int]]:
    """Frame packets whose payload equals their index, for easy assertions."""
    return [FramePacket(index=i, frame=i, original_size=(1, 1)) for i in range(n)]


def _starts(windows: Iterator) -> list[int]:
    return [w.start_index for w in windows]


class TestIterTemporalWindows:
    def test_basic_stride_one(self) -> None:
        windows = list(
            iter_temporal_windows(_frames(5), sequence_length=3, stride=1)
        )
        assert [w.start_index for w in windows] == [0, 1, 2]
        assert windows[0].frame_indices == (0, 1, 2)
        assert windows[0].frames == (0, 1, 2)

    def test_stride_two_emits_backfill_tail(self) -> None:
        # 6 frames, seq=3, stride=2 -> regular starts 0,2 then a tail at 3 so the
        # last frame (5) is covered.
        starts = _starts(
            iter_temporal_windows(_frames(6), sequence_length=3, stride=2)
        )
        assert starts == [0, 2, 3]

    def test_drop_policy_skips_tail(self) -> None:
        starts = _starts(
            iter_temporal_windows(
                _frames(6), sequence_length=3, stride=2, tail_policy="drop"
            )
        )
        assert starts == [0, 2]

    def test_no_duplicate_tail_when_aligned(self) -> None:
        # stride=1 already ends on the last frame -> no extra tail window.
        starts = _starts(
            iter_temporal_windows(_frames(5), sequence_length=3, stride=1)
        )
        assert starts == [0, 1, 2]

    def test_fewer_frames_than_sequence_backfills(self) -> None:
        windows = list(
            iter_temporal_windows(_frames(2), sequence_length=3, stride=1)
        )
        assert len(windows) == 1
        assert windows[0].start_index == 0
        # Last frame is repeated to pad to sequence_length.
        assert windows[0].frame_indices == (0, 1, 1)

    def test_fewer_frames_drop_policy_yields_nothing(self) -> None:
        windows = list(
            iter_temporal_windows(
                _frames(2), sequence_length=3, stride=1, tail_policy="drop"
            )
        )
        assert windows == []

    def test_empty_stream_yields_nothing(self) -> None:
        assert list(iter_temporal_windows([], sequence_length=3, stride=1)) == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sequence_length": 0, "stride": 1},
            {"sequence_length": 3, "stride": 0},
            {"sequence_length": 3, "stride": 1, "tail_policy": "nope"},
        ],
    )
    def test_invalid_args_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            list(iter_temporal_windows(_frames(4), **kwargs))

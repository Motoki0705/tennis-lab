import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.play_intervals import (
    PlayIntervalConfig,
    PlaySelection,
    infer_play_intervals,
)


def select(mask: NDArray[np.bool_], *, fps: float = 30, eligible: NDArray[np.bool_] | None = None) -> PlaySelection:
    return infer_play_intervals(mask, mask.copy(), np.arange(len(mask)) / fps,
                                np.ones(len(mask), bool) if eligible is None else eligible,
                                PlayIntervalConfig())


def test_short_gaps_bridge_without_extending_or_changing_evidence() -> None:
    mask: NDArray[np.bool_] = np.zeros(180, bool)
    mask[20:70] = mask[80:150] = True
    original = mask.copy()
    result = select(mask)
    assert result.intervals == ((20, 150),)
    assert result.excluded == ((0, 20), (150, 180))
    assert result.bridged[70:80].all()
    np.testing.assert_array_equal(mask, original)
    assert all(start >= 20 and start + 32 <= 150 for start in result.window_starts)


def test_long_gap_and_sparse_single_detections_do_not_join() -> None:
    mask: NDArray[np.bool_] = np.zeros(200, bool)
    mask[10:50] = mask[90:140] = True
    mask[160::10] = True
    result = select(mask)
    assert result.intervals == ((10, 50), (90, 140))
    assert not result.selected[160:].any()


def test_gap_threshold_uses_seconds_not_frame_count() -> None:
    mask: NDArray[np.bool_] = np.ones(100, bool)
    mask[40:60] = False
    assert select(mask, fps=60).intervals == ((0, 100),)
    assert select(mask, fps=30).intervals == ((0, 40), (60, 100))


def test_reference_frames_and_timestamp_jumps_are_barriers() -> None:
    mask: NDArray[np.bool_] = np.ones(100, bool)
    eligible = mask.copy()
    eligible[45:50] = False
    result = select(mask, eligible=eligible)
    assert result.intervals == ((0, 45), (50, 100))
    times = np.arange(100) / 30
    times[50:] += 5
    result = infer_play_intervals(mask, mask, times, mask, PlayIntervalConfig())
    assert all(s + 32 <= 50 or s >= 50 for s in result.window_starts)


def test_unknown_positions_never_become_coordinate_targets() -> None:
    presence: NDArray[np.bool_] = np.ones(100, bool)
    observed: NDArray[np.bool_] = np.zeros(100, bool)
    result = infer_play_intervals(presence, observed, np.arange(100) / 30,
                                 presence, PlayIntervalConfig())
    assert result.window_starts == ()
    assert result.intervals == ((0, 100),)  # No location GT does not mean non-play.


def test_short_clip_and_empty_evidence_remain_excluded() -> None:
    assert select(np.ones(31, bool)).intervals == ()
    assert select(np.zeros(100, bool)).excluded == ((0, 100),)


def test_invalid_timeline_is_rejected() -> None:
    mask: NDArray[np.bool_] = np.ones(32, bool)
    with pytest.raises(ValueError, match="strictly"):
        infer_play_intervals(mask, mask, np.zeros(32), mask, PlayIntervalConfig())

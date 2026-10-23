"""Native barriers, identical pack occurrences and deterministic frame ownership."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.windows.all_frames import (
    AllFrameWindowConfig,
    child_starts,
    common_intervals,
    decode_label_runs,
    encode_label_runs,
    evaluation_windows,
    owner_starts,
    training_packs,
    window_starts,
)


def test_reference_and_pts_barriers_remain_separate_even_when_adjacent() -> None:
    target: NDArray[np.bool_] = np.ones(512, dtype=np.bool_)
    target[256] = False
    times: NDArray[np.float64] = np.arange(512, dtype=np.float64)
    times[128:] += 20
    times[384:] += 20
    states: Any = SimpleNamespace(target=target, times=times)
    for length in (32, 64, 128):
        assert common_intervals(states, AllFrameWindowConfig(length)) == [(0, 128), (128, 256), (384, 512)]


@pytest.mark.parametrize("count", [128, 129, 145, 256, 270, 573])
@pytest.mark.parametrize("length", [32, 64, 128])
def test_owners_match_bruteforce_and_cover_interval_exactly_once(count: int, length: int) -> None:
    a, b = 5, 5 + count
    starts = window_starts(a, b, length, length // 2)
    expected = [min((s for s in starts if s <= f < s + length),
                    key=lambda s: (abs(2 * f - (2 * s + length - 1)), s)) for f in range(a, b)]
    owners = owner_starts(a, b, length)
    assert owners.tolist() == expected
    windows = evaluation_windows([(a, b)], length)
    assert [w["start"] for w in windows] == starts
    owned = [f for w in windows for f in range(*w["owner_range"])]
    assert owned == list(range(a, b))
    assert owners[0] == a and owners[-1] == b - length
    if count == 129:
        midpoint = (starts[-2] + starts[-1] + length - 1) // 2
        assert owners[midpoint - a] == starts[-2]


def test_shared_packs_keep_masked_children_and_teacher_multiplicities() -> None:
    codes: NDArray[np.uint8] = np.zeros(401, dtype=np.uint8)
    codes[7], codes[130], codes[255] = 1, 2, 3
    intervals = [(0, len(codes))]
    previous = None
    for length in (32, 64, 128):
        cfg = AllFrameWindowConfig(length)
        packs = training_packs(intervals, codes, cfg)
        if previous is not None:
            assert packs == previous
        previous = packs
        assert len(packs) < len(window_starts(0, len(codes), 128, 16))
        if length < 128:
            assert any(not codes[s:s + length].any() for p in packs for s in child_starts(p["start"], cfg))
        for pack in packs:
            start = pack["start"]
            children = child_starts(start, cfg)
            assert [i for s in children for i in range(s, s + length)] == list(range(start, start + 128))
            actual: NDArray[np.int64] = np.sum([np.bincount(codes[s:s + length], minlength=4) for s in children], axis=0)
            np.testing.assert_array_equal(actual, np.bincount(codes[start:start + 128], minlength=4))
    # Nonaligned tail is retained even when the only teacher lies at the last frame.
    codes[:] = 0
    codes[-1] = 2
    assert training_packs(intervals, codes, AllFrameWindowConfig(32)) == [dict(start=273, interval_index=0)]


def test_label_run_encoding_is_complete_and_rejects_ambiguous_runs() -> None:
    codes = np.array([0, 0, 1, 2, 2, 3, 0], np.uint8)
    runs = encode_label_runs(codes)
    np.testing.assert_array_equal(decode_label_runs(runs, len(codes)), codes)
    for invalid in (runs[:-1], [[0, 2, 0], [3, 7, 1]], [[0, 7, 4]], [[0, 2, 0], [2, 7, 0]]):
        with pytest.raises(ValueError, match="runs"):
            decode_label_runs(invalid, len(codes))


@pytest.mark.parametrize("options", [dict(window_length=0), dict(window_length=33), dict(window_length=256),
                                     dict(window_length=48), dict(pack_length=256), dict(frame_step=2),
                                     dict(pack_stride=0), dict(timestamp_gap_factor=float("nan")), dict(window_length=True)])
def test_invalid_or_nonpartitioning_parameters_are_rejected(options: dict) -> None:
    with pytest.raises(ValueError):
        AllFrameWindowConfig(**dict(dict(window_length=32), **options))

"""Pressure timers must tolerate brief dips without hiding sustained exhaustion."""

import pytest

from src.utils.resource_guard import GIB, HostRAMGuard


def test_transient_pressure_recovers_and_resets_timer() -> None:
    guard = HostRAMGuard()
    assert guard.sample(8 * GIB, 0) is None
    assert guard.sample(5 * GIB, 1) is None
    assert guard.sample(5 * GIB, 30) is None
    assert guard.sample(6 * GIB, 31) is None
    assert guard.sample(5 * GIB, 32) is None
    assert guard.sample(5 * GIB, 61.99) is None
    assert "sustained" in str(guard.sample(5 * GIB, 62))


def test_every_sample_below_floor_for_30_seconds_trips() -> None:
    guard = HostRAMGuard()
    for second in range(30):
        assert guard.sample(6 * GIB - 1, second) is None
    assert "30.0s" in str(guard.sample(6 * GIB - 1, 30))


def test_emergency_is_immediate_but_exact_boundary_uses_timer() -> None:
    guard = HostRAMGuard()
    assert guard.sample(3 * GIB, 0) is None
    assert "emergency" in str(guard.sample(3 * GIB - 1, 0.1))


def test_timeline_retains_dips_and_peak_process_memory() -> None:
    guard = HostRAMGuard()
    for available, time, rss in [(8, 0, 1), (5, 1, 3), (7, 9, 2), (8, 10, 1)]:
        guard.sample(available * GIB, time, rss_bytes=rss * GIB)
    report = guard.report()
    assert report["minimum_available_bytes"] == 5 * GIB
    assert report["peak_process_tree_rss_bytes"] == 3 * GIB
    assert report["samples"] == 4
    assert len(guard.timeline) == 2
    assert guard.timeline[0]["min_available_bytes"] == 5 * GIB
    assert guard.timeline[0]["samples"] == 3


@pytest.mark.parametrize("seconds", [-1., float("nan"), float("inf"), 0.])
def test_invalid_or_backwards_clock_stops(seconds: float) -> None:
    guard = HostRAMGuard()
    guard.sample(8 * GIB, 1)
    with pytest.raises(ValueError):
        guard.sample(8 * GIB, seconds)

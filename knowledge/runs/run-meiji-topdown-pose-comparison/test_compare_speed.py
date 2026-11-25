"""Check the comparison against small cases with independent expected results."""

from __future__ import annotations

import pytest
from compare_speed import Point, improvements, pareto_front, validate_samples

from src.utils.inference_benchmark import BenchmarkProtocol
from src.utils.inference_benchmark_inputs import TimingSchedule


def test_pareto_uses_all_three_metrics_within_each_batch() -> None:
    baseline = ("vitpose-h", "TRT16", 1)
    dominated = ("pmpose-l", "TRT16", 1)
    smaller_memory = ("rtmpose-x", "TRT16", 1)
    same = ("pmpose-h", "TRT16", 1)
    different_batch = ("rtmpose-m", "TRT16", 8)
    points = {
        baseline: Point(0.78, 5.0, 500),
        dominated: Point(0.77, 7.0, 600),
        smaller_memory: Point(0.77, 6.0, 400),
        same: Point(0.78, 5.0, 500),
        different_batch: Point(0.72, 8.0, 700),
    }
    assert pareto_front(points) == {baseline, same, smaller_memory, different_batch}


def test_improvements_preserve_accuracy_when_speed_is_infeasible() -> None:
    baseline = {
        "mean_oks": 0.74,
        "pck_0_10": 0.78,
        "p95_px": 8.0,
        "speed_status": "ok",
        "wall_p50_ms_person": 10.0,
        "nvml_peak_mib": 400.0,
        "persons_per_s": 100.0,
    }
    row = {
        **baseline,
        "mean_oks": 0.76,
        "pck_0_10": 0.77,
        "p95_px": 7.0,
        "wall_p50_ms_person": 8.0,
        "persons_per_s": 125.0,
        "nvml_peak_mib": 200.0,
    }
    assert improvements(row, baseline) == pytest.approx(
        {
            "oks_delta_pp": 2.0,
            "pck_0_10_delta_pp": -1.0,
            "p95_px_reduction_pct": 12.5,
            "wall_p50_reduction_pct": 20.0,
            "nvml_peak_reduction_pct": 50.0,
            "throughput_gain_pct": 25.0,
        }
    )
    failure = {
        **row,
        "speed_status": "infeasible",
        "wall_p50_ms_person": None,
        "persons_per_s": None,
        "nvml_peak_mib": None,
    }
    result = improvements(failure, baseline)
    assert result["oks_delta_pp"] == pytest.approx(2.0)
    assert result["p95_px_reduction_pct"] == pytest.approx(12.5)
    assert result["wall_p50_reduction_pct"] is None
    assert result["nvml_peak_reduction_pct"] is None
    assert result["throughput_gain_pct"] is None


def test_timed_samples_require_order_and_count_for_success() -> None:
    protocol = BenchmarkProtocol(
        batch_sizes=(1,), warmup=1, iterations_per_block=2, blocks=2
    )
    schedule = TimingSchedule(
        protocol, (("clip-a",), ("clip-z",), ("clip-b",)), "fixture"
    )
    samples = [
        {"ids": [clip], "items": 1, "processed_items": 1, "block": block}
        for clip, block in (("clip-a", 1), ("clip-z", 1), ("clip-b", 2), ("clip-a", 2))
    ]
    validate_samples(samples, schedule, 1, complete=True)
    validate_samples(samples[:3], schedule, 1, complete=False)
    with pytest.raises(ValueError, match="count of timed iterations"):
        validate_samples(samples[:3], schedule, 1, complete=True)
    with pytest.raises(ValueError, match="departed from the pinned"):
        validate_samples(
            [samples[1], samples[0], *samples[2:]], schedule, 1, complete=True
        )

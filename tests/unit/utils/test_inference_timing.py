"""Check attribution from a hand-calculated wall, CPU and GPU timeline."""

from typing import Any

import pytest

from src.utils.inference_timing import TimingSummary, timing_summary


def event(
    ident: int, stage: str, start: int, end: int, *, thread_cpu: tuple[int, int],
    process_cpu: tuple[int, int], tid: int = 1, parent: int | None = None, **attrs: Any,
) -> dict[str, Any]:
    return {
        "id": ident, "parent_id": parent, "stage": stage, "start_ns": start * 10**9, "end_ns": end * 10**9,
        "native_thread_id": tid, "thread_role": "main" if tid == 1 else "decode_prefetch",
        "thread_cpu_start_ns": thread_cpu[0] * 10**9, "thread_cpu_ns": thread_cpu[1] * 10**9,
        "process_cpu_start_ns": process_cpu[0] * 10**9, "process_cpu_ns": process_cpu[1] * 10**9,
        "attributes": attrs,
    }


def test_gpu_overlap_and_cpu_ranking_discount_nested_work_and_unknown_samples() -> None:
    events = [
        event(0, "postprocess", 0, 10, thread_cpu=(0, 6), process_cpu=(0, 10), operation="geometry"),
        event(1, "preprocess", 2, 4, thread_cpu=(1, 1), process_cpu=(2, 2), parent=0, operation="resize"),
        event(2, "decode", 1, 4, thread_cpu=(0, 2), process_cpu=(1, 3), tid=2, parent=0),
        event(3, "cuda_sync", 6, 7, thread_cpu=(4, 0), process_cpu=(6, 1), parent=0),
    ]
    # The first five seconds are idle, the next three are busy. The gap after
    # t=8 is unknown and must not be credited to the CPU.
    samples = [
        {"at_ns": second * 10**9, "gpu": {} if usage is None else {"cuda:0": {"utilization_percent": usage}}}
        for second, usage in ((0, 0), (5, 0), (8, 75), (10, None))
    ]
    report = timing_summary(events, samples, {})
    assert report["by_stage"]["postprocess"] == {
        "wall_seconds": 10., "thread_cpu_seconds": 6., "process_cpu_seconds": 10.,
        "gpu_observed_wall_seconds": 8., "gpu_active_fraction": 3 / 8,
        "gpu_mean_utilization_percent": 225 / 8,
    }
    ranked = report["cpu_during_gpu_idle"]
    assert [row["stage"] for row in ranked] == ["postprocess", "decode", "preprocess"]
    assert [row["estimated_thread_cpu_seconds"] for row in ranked] == pytest.approx([15 / 7, 2., 1.])
    assert ranked[1]["thread_role"] == "decode_prefetch"
    assert ranked[0]["thread_wall_seconds"] == 3.


def test_shared_batches_charge_each_participating_node_and_use_union_per_group() -> None:
    addresses = [("a", "predict/cam0"), ("b", "predict/cam0"), ("b", "predict/cam1")]
    events = [
        event(0, "batch_dispatch", 0, 10, thread_cpu=(0, 6), process_cpu=(0, 12), model="detector",
              batch_nodes=addresses),
        event(1, "inference", 2, 8, thread_cpu=(1, 4), process_cpu=(2, 8), parent=0,
              model="forward", batch_nodes=addresses),
        event(2, "preprocess", 8, 12, thread_cpu=(0, 3), process_cpu=(10, 4), tid=2, model="detector",
              clip_id="b", node="predict/cam0"),
    ]
    report = timing_summary(events, [], {"forward": "detector"})
    assert report["by_node"]["predict/cam0"] == report["by_model"]["detector"] == {
        "wall_seconds": 12., "thread_cpu_seconds": 9., "process_cpu_seconds": 14.,
        "gpu_observed_wall_seconds": 0., "gpu_active_fraction": None, "gpu_mean_utilization_percent": None,
    }
    assert report["by_clip_node"]["a"]["predict/cam0"]["wall_seconds"] == 10.
    assert report["by_clip_node"]["b"]["predict/cam1"]["wall_seconds"] == 10.
    assert report["by_clip_node"]["b"]["predict/cam0"]["wall_seconds"] == 12.
    assert report["cpu_during_gpu_idle"] == []


def test_disjoint_windows_weight_gpu_coverage_and_keep_idle_cpu_totals() -> None:
    attributes: dict[str, Any] = {"clip_id": "clip", "node": "detection/cam0", "model": "resize", "operation": "resize"}
    first = event(0, "preprocess", 0, 2, thread_cpu=(0, 2), process_cpu=(0, 2), **attributes)
    second = event(1, "preprocess", 3, 7, thread_cpu=(2, 2), process_cpu=(2, 3), **attributes)
    samples = [
        {"at_ns": second * 10**9, "gpu": {} if usage is None else {"cuda:0": {"utilization_percent": usage}}}
        for second, usage in ((0, 0), (2, 0), (4, 20), (6, 80), (7, None))
    ]
    summary = TimingSummary(samples, {})
    summary.add([first])
    summary.add([second])
    result = summary.result()
    assert result == timing_summary([first, second], samples, {})
    assert result["by_node"]["detection/cam0"] == {
        "wall_seconds": 6., "thread_cpu_seconds": 4., "process_cpu_seconds": 5.,
        "gpu_observed_wall_seconds": 5., "gpu_active_fraction": .6, "gpu_mean_utilization_percent": 36.,
    }
    assert result["cpu_during_gpu_idle"][0]["estimated_thread_cpu_seconds"] == 2.

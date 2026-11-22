"""Interval statistics for inference recordings."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import pairwise
from typing import Any

Interval = tuple[int, int]
Event = Mapping[str, Any]
_CPU_WORK = {"decode", "video_read", "preprocess", "postprocess", "artifact_load",
             "save", "publication", "planning", "progress_write"}


def _union(intervals: Iterable[Interval]) -> Iterator[Interval]:
    start = end = 0
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if left > end:
            if start < end:
                yield start, end
            start = left
        end = max(end, right)
    if start < end:
        yield start, end


class _GpuTimeline:
    """Integrate adjacent samples, leaving intervals with missing telemetry unknown."""

    def __init__(self, samples: Sequence[Event]) -> None:
        self._times = [s["at_ns"] for s in samples]
        self._prefix: list[tuple[int, int, int]] = [(0, 0, 0)]
        self._levels: list[tuple[int, int, int]] = []
        for before, after in pairwise(samples):
            left, right = before["gpu"], after["gpu"]
            # NVML utilization measures a preceding window. Attribute the sample at
            # the end of each interval, and require matching devices at both ends.
            utilization = max((gpu["utilization_percent"] for gpu in right.values()), default=0)
            covered = int(bool(right) and right.keys() == left.keys())
            level = (covered, covered * int(utilization > 0), covered * utilization)
            self._levels.append(level)
            ns = after["at_ns"] - before["at_ns"]
            prev_covered, prev_busy, prev_utilization = self._prefix[-1]
            self._prefix.append((prev_covered + covered * ns, prev_busy + level[1] * ns,
                                 prev_utilization + level[2] * ns))

    def _at(self, position: int) -> tuple[int, int, int]:
        index = bisect_right(self._times, position) - 1
        if index < 0:
            return 0, 0, 0
        covered, busy, utilization = self._prefix[index]
        if index < len(self._levels):
            distance = position - self._times[index]
            level = self._levels[index]
            return covered + level[0] * distance, busy + level[1] * distance, utilization + level[2] * distance
        return covered, busy, utilization

    def overlap(self, start: int, end: int) -> tuple[int, int, int]:
        a_covered, a_busy, a_utilization = self._at(start)
        b_covered, b_busy, b_utilization = self._at(end)
        return b_covered - a_covered, b_busy - a_busy, b_utilization - a_utilization


def _clock_interval(event: Event, clock: str) -> Interval:
    start = event[f"{clock}_start_ns"]
    return start, start + event[f"{clock}_ns"]


def _group(events: Sequence[Event], gpu: _GpuTimeline) -> dict[str, Any]:
    wall = covered = busy = utilization = 0
    for start, end in _union((e["start_ns"], e["end_ns"]) for e in events):
        wall += end - start
        coverage, active, weighted = gpu.overlap(start, end)
        covered += coverage
        busy += active
        utilization += weighted
    threads: dict[int, list[Interval]] = defaultdict(list)
    for event in events:
        threads[event["native_thread_id"]].append(_clock_interval(event, "thread_cpu"))
    return {
        "wall_seconds": wall / 1e9,
        "thread_cpu_seconds": sum(end - start for times in threads.values() for start, end in _union(times)) / 1e9,
        "process_cpu_seconds": sum(end - start for start, end in _union(
            _clock_interval(e, "process_cpu") for e in events)) / 1e9,
        "gpu_observed_wall_seconds": covered / 1e9,
        "gpu_active_fraction": busy / covered if covered else None,
        "gpu_mean_utilization_percent": utilization / covered if covered else None,
    }


def _idle_cpu(events: Sequence[Event], gpu: _GpuTimeline, model_aliases: Mapping[str, str]) -> list[dict[str, Any]]:
    by_id = {e["id"]: e for e in events}
    children: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        parent = by_id.get(event["parent_id"])
        if parent is not None and parent["native_thread_id"] == event["native_thread_id"]:
            children[parent["id"]].append(event)

    rows: dict[tuple[str, str, str, str, str], tuple[float, int]] = {}
    for event in events:
        stage, attrs = event["stage"], event["attributes"]
        if stage not in _CPU_WORK or str(attrs.get("device", "")).startswith("cuda"):
            continue
        start, end = event["start_ns"], event["end_ns"]
        cpu = event["thread_cpu_ns"]
        exclusive_wall = end - start
        covered, busy, _ = gpu.overlap(start, end)
        idle = covered - busy
        if not idle or not cpu:
            continue
        for child in children.get(event["id"], ()):
            child_start, child_end = child["start_ns"], child["end_ns"]
            cpu -= child["thread_cpu_ns"]
            exclusive_wall -= child_end - child_start
            child_covered, child_busy, _ = gpu.overlap(child_start, child_end)
            idle -= child_covered - child_busy
        if idle > 0 and cpu > 0 and exclusive_wall > 0:
            model = str(attrs.get("model") or "")
            key = (stage, attrs.get("operation", ""), str(attrs.get("node") or ""),
                   model_aliases.get(model, model), event["thread_role"])
            previous_cpu, previous_wall = rows.get(key, (0., 0))
            rows[key] = previous_cpu + cpu * idle / exclusive_wall, previous_wall + idle
    return [
        {"stage": key[0], "operation": key[1], "node": key[2], "model": key[3], "thread_role": key[4],
         "estimated_thread_cpu_seconds": cpu / 1e9, "thread_wall_seconds": wall / 1e9}
        for key, (cpu, wall) in sorted(rows.items(), key=lambda entry: (-entry[1][0], entry[0]))
    ]


def timing_summary(events: Sequence[Event], samples: Sequence[Event],
                   model_aliases: Mapping[str, str]) -> dict[str, Any]:
    gpu = _GpuTimeline(samples)
    stages: dict[str, list[Event]] = defaultdict(list)
    nodes: dict[str, list[Event]] = defaultdict(list)
    clip_nodes: dict[str, dict[str, list[Event]]] = defaultdict(lambda: defaultdict(list))
    models: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        stages[event["stage"]].append(event)
        attrs = event["attributes"]
        if model := attrs.get("model"):
            models[model_aliases.get(model, model)].append(event)
        addresses = attrs.get("batch_nodes")
        if addresses is None and attrs.get("node"):
            addresses = ((attrs.get("clip_id"), attrs["node"]),)
        if addresses is not None:
            names = set()
            for clip, node in addresses:
                names.add(node)
                if clip is not None:
                    clip_nodes[clip][node].append(event)
            for node in names:
                nodes[node].append(event)

    return {
        "by_stage": {key: _group(group, gpu) for key, group in sorted(stages.items())},
        "by_node": {key: _group(group, gpu) for key, group in sorted(nodes.items())},
        "by_clip_node": {clip: {key: _group(group, gpu) for key, group in sorted(groups.items())}
                         for clip, groups in sorted(clip_nodes.items())},
        "by_model": {key: _group(group, gpu) for key, group in sorted(models.items())},
        "cpu_during_gpu_idle": _idle_cpu(events, gpu, model_aliases),
    }

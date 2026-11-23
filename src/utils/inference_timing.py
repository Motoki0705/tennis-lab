"""Interval statistics for inference recordings."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(slots=True)
class _GroupTimes:
    wall_ns: int
    thread_cpu_ns: int
    process_cpu_ns: int
    gpu_observed_ns: int
    gpu_active_ns: int
    gpu_percent_ns: int

    def add(self, other: _GroupTimes) -> None:
        self.wall_ns += other.wall_ns
        self.thread_cpu_ns += other.thread_cpu_ns
        self.process_cpu_ns += other.process_cpu_ns
        self.gpu_observed_ns += other.gpu_observed_ns
        self.gpu_active_ns += other.gpu_active_ns
        self.gpu_percent_ns += other.gpu_percent_ns

    def result(self) -> dict[str, Any]:
        return {
            "wall_seconds": self.wall_ns / 1e9,
            "thread_cpu_seconds": self.thread_cpu_ns / 1e9,
            "process_cpu_seconds": self.process_cpu_ns / 1e9,
            "gpu_observed_wall_seconds": self.gpu_observed_ns / 1e9,
            "gpu_active_fraction": self.gpu_active_ns / self.gpu_observed_ns if self.gpu_observed_ns else None,
            "gpu_mean_utilization_percent": self.gpu_percent_ns / self.gpu_observed_ns if self.gpu_observed_ns else None,
        }


def _group(events: Sequence[Event], gpu: _GpuTimeline) -> _GroupTimes:
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
    thread_cpu = sum(end - start for times in threads.values() for start, end in _union(times))
    process_cpu = sum(end - start for start, end in _union(_clock_interval(e, "process_cpu") for e in events))
    return _GroupTimes(wall, thread_cpu, process_cpu, covered, busy, utilization)


IdleKey = tuple[str, str, str, str, str]


def _idle_cpu(events: Sequence[Event], gpu: _GpuTimeline, model_aliases: Mapping[str, str],
              rows: dict[IdleKey, tuple[float, int]]) -> None:
    by_id = {e["id"]: e for e in events}
    children: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        parent = by_id.get(event["parent_id"])
        if parent is not None and parent["native_thread_id"] == event["native_thread_id"]:
            children[parent["id"]].append(event)

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


class TimingSummary:
    """Accumulate interval metrics for disjoint windows on the same clock."""

    def __init__(self, samples: Sequence[Event], model_aliases: Mapping[str, str]) -> None:
        self.gpu = _GpuTimeline(samples)
        self.model_aliases = model_aliases
        self.stages: dict[str, _GroupTimes] = {}
        self.nodes: dict[str, _GroupTimes] = {}
        self.clip_nodes: dict[str, dict[str, _GroupTimes]] = defaultdict(dict)
        self.models: dict[str, _GroupTimes] = {}
        self.idle: dict[IdleKey, tuple[float, int]] = {}

    def _add_groups(self, output: dict[str, _GroupTimes], groups: Mapping[str, list[Event]]) -> None:
        for key, group in groups.items():
            new = _group(group, self.gpu)
            old = output.get(key)
            if old is None:
                output[key] = new
            else:
                old.add(new)

    def add(self, events: Sequence[Event]) -> None:
        stages: dict[str, list[Event]] = defaultdict(list)
        nodes: dict[str, list[Event]] = defaultdict(list)
        clip_nodes: dict[str, dict[str, list[Event]]] = defaultdict(lambda: defaultdict(list))
        models: dict[str, list[Event]] = defaultdict(list)
        for event in events:
            stages[event["stage"]].append(event)
            attrs = event["attributes"]
            if model := attrs.get("model"):
                models[self.model_aliases.get(model, model)].append(event)
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
        self._add_groups(self.stages, stages)
        self._add_groups(self.nodes, nodes)
        self._add_groups(self.models, models)
        for clip, groups in clip_nodes.items():
            self._add_groups(self.clip_nodes[clip], groups)
        _idle_cpu(events, self.gpu, self.model_aliases, self.idle)

    def result(self) -> dict[str, Any]:
        ranked = [
            {"stage": key[0], "operation": key[1], "node": key[2], "model": key[3], "thread_role": key[4],
             "estimated_thread_cpu_seconds": cpu / 1e9, "thread_wall_seconds": wall / 1e9}
            for key, (cpu, wall) in sorted(self.idle.items(), key=lambda entry: (-entry[1][0], entry[0]))
        ]
        return {
            "by_stage": {key: value.result() for key, value in sorted(self.stages.items())},
            "by_node": {key: value.result() for key, value in sorted(self.nodes.items())},
            "by_model": {key: value.result() for key, value in sorted(self.models.items())},
            "by_clip_node": {
                clip: {key: value.result() for key, value in sorted(group.items())}
                for clip, group in sorted(self.clip_nodes.items())
            },
            "cpu_during_gpu_idle": ranked,
        }


def timing_summary(events: Sequence[Event], samples: Sequence[Event],
                   model_aliases: Mapping[str, str]) -> dict[str, Any]:
    summary = TimingSummary(samples, model_aliases)
    summary.add(events)
    return summary.result()

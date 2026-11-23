"""Sampling values and thread roles with fixed CPU, RAM and GPU readings."""

from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.utils import inference_resources as resources

CpuTimes = namedtuple("CpuTimes", "user system idle iowait guest")


def test_resource_samples_report_delta_per_core_and_native_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([1_000_000_000, 1_003_000_000, 3_000_000_000, 3_003_000_000])
    before_after = iter([200_000_000, 1_700_000_000])
    monkeypatch.setattr(resources, "time", SimpleNamespace(
        perf_counter_ns=lambda: next(clock), process_time_ns=lambda: next(before_after)))
    threads = iter([
        [SimpleNamespace(id=11, user_time=.25, system_time=0.)],
        [SimpleNamespace(id=11, user_time=1.25, system_time=.25),
         SimpleNamespace(id=47, user_time=.25, system_time=0.)],
    ])
    monkeypatch.setattr(resources.psutil, "Process", lambda: SimpleNamespace(
        threads=lambda: next(threads), memory_info=lambda: SimpleNamespace(rss=8192)))
    cores = iter([
        [CpuTimes(1., 1., 2., 0., 0.), CpuTimes(1., 1., 2., 0., 0.)],
        [CpuTimes(2., 1., 2., 1., .5), CpuTimes(1., 2., 3., 0., 0.)],
    ])
    monkeypatch.setattr(resources.psutil, "cpu_times", lambda *, percpu: next(cores))
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(available=65536, total=131072))
    gpu = {"cuda:0": {"utilization_percent": 83, "used_bytes": 8192, "total_bytes": 16384}}
    sampler = resources.ResourceSampler(0, Mock(sample=lambda: gpu))
    initial = sampler.sample({11: "main"})
    assert initial["at_ns"] == 10**9 and initial["system_core_cpu_percent"] == [None, None]
    assert initial["process_cpu_percent"] is None
    second = sampler.sample({11: "main", 47: "decode_prefetch"})
    assert second == {
        "at_ns": 3 * 10**9, "sample_duration_ns": 3_000_000, "process_cpu_percent": 75.,
        "system_core_cpu_percent": [50., 50.],
        "threads": [{"native_id": 11, "role": "main", "cpu_percent": 62.5},
                    {"native_id": 47, "role": "decode_prefetch", "cpu_percent": None}],
        "gpu": gpu, "rss_bytes": 8192, "system_memory_available_bytes": 65536,
        "system_memory_total_bytes": 131072,
    }

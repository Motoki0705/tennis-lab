"""Process, thread, CPU and NVML samples on the recorder's monotonic clock."""

from __future__ import annotations

import ctypes
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlSampler:
    """Resolve CUDA devices by UUID, including when CUDA_VISIBLE_DEVICES reorders them."""

    def __init__(self, cuda_uuids: Mapping[str, str]) -> None:
        self._nvml = ctypes.CDLL("libnvidia-ml.so.1")
        self._call("nvmlInit_v2")
        self._handles: dict[str, ctypes.c_void_p] = {}
        try:
            for device, cuda_uuid in cuda_uuids.items():
                handle = ctypes.c_void_p()
                uuid = f"GPU-{cuda_uuid.removeprefix('GPU-')}".encode("ascii")
                self._call("nvmlDeviceGetHandleByUUID", uuid, ctypes.byref(handle))
                self._handles[device] = handle
        except BaseException:
            self.close()
            raise

    def _call(self, name: str, *args: Any) -> None:
        result = getattr(self._nvml, name)(*args)
        if result:
            error_string = self._nvml.nvmlErrorString
            error_string.restype = ctypes.c_char_p
            raise RuntimeError(f"NVML {name}: {error_string(result).decode()} ({result})")

    def _memory(self, handle: ctypes.c_void_p) -> _NvmlMemory:
        memory = _NvmlMemory()
        self._call("nvmlDeviceGetMemoryInfo", handle, ctypes.byref(memory))
        return memory

    def memory_used_bytes(self, device: str) -> int:
        return int(self._memory(self._handles[device]).used)

    def device_state(self, device: str) -> dict[str, object]:
        """Read driver, clock and power telemetry once, preserving unavailable fields."""
        driver = ctypes.create_string_buffer(96)
        self._call("nvmlSystemGetDriverVersion", driver, len(driver))
        handle = self._handles[device]
        result: dict[str, object] = {"driver": driver.value.decode("ascii")}
        errors: dict[str, str] = {}
        for field, function, args in (
            ("sm_clock_mhz", "nvmlDeviceGetClockInfo", (1,)),
            ("memory_clock_mhz", "nvmlDeviceGetClockInfo", (2,)),
            ("max_sm_clock_mhz", "nvmlDeviceGetMaxClockInfo", (1,)),
            ("max_memory_clock_mhz", "nvmlDeviceGetMaxClockInfo", (2,)),
            ("power_mw", "nvmlDeviceGetPowerUsage", ()),
            ("power_limit_mw", "nvmlDeviceGetEnforcedPowerLimit", ()),
        ):
            value = ctypes.c_uint()
            try:
                self._call(function, handle, *args, ctypes.byref(value))
            except (AttributeError, RuntimeError) as error:
                errors[field] = str(error)
                result[field] = None
            else:
                result[field] = value.value
        result["telemetry_errors"] = errors
        return result

    def sample(self) -> dict[str, dict[str, int]]:
        result = {}
        for device, handle in self._handles.items():
            memory, utilization = self._memory(handle), _NvmlUtilization()
            self._call("nvmlDeviceGetUtilizationRates", handle, ctypes.byref(utilization))
            result[device] = {
                "utilization_percent": utilization.gpu,
                "used_bytes": memory.used,
                "total_bytes": memory.total,
            }
        return result

    def close(self) -> None:
        self._call("nvmlShutdown")


def _core_busy_percent(previous: Any, current: Any) -> float | None:
    # Linux guest time is already included in user/nice.
    times = {name: max(0., getattr(current, name) - getattr(previous, name))
             for name in current._fields if name not in {"guest", "guest_nice"}}
    total = sum(times.values())
    busy = sum(value for name, value in times.items() if name not in {"idle", "iowait"})
    return None if not total else 100. * busy / total


class ResourceSampler:
    def __init__(self, started_ns: int, nvml: NvmlSampler | None) -> None:
        self._started_ns = started_ns
        self._nvml = nvml
        self._process = psutil.Process()
        self._previous_ns: int | None = None
        self._previous_process_ns = 0
        self._previous_threads: dict[int, float] = {}
        self._previous_cores: Sequence[Any] = ()

    def sample(self, roles: Mapping[int, str]) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        process_ns = time.process_time_ns()
        thread_times = {thread.id: thread.user_time + thread.system_time for thread in self._process.threads()}
        cores = psutil.cpu_times(percpu=True)
        memory = psutil.virtual_memory()
        elapsed = None if self._previous_ns is None else (now_ns - self._previous_ns) / 1e9
        process_percent = None
        if elapsed:
            process_percent = 100. * (process_ns - self._previous_process_ns) / 1e9 / elapsed
        threads = []
        for tid, seconds in thread_times.items():
            previous = self._previous_threads.get(tid)
            percent = None if not elapsed or previous is None else 100. * max(0., seconds - previous) / elapsed
            threads.append({"native_id": tid, "role": roles.get(tid, "native"), "cpu_percent": percent})
        core_percent = [
            _core_busy_percent(self._previous_cores[index], core) if index < len(self._previous_cores) else None
            for index, core in enumerate(cores)
        ]
        result = {
            "at_ns": now_ns - self._started_ns,
            "rss_bytes": self._process.memory_info().rss,
            "system_memory_available_bytes": memory.available,
            "system_memory_total_bytes": memory.total,
            "process_cpu_percent": process_percent,
            "threads": threads,
            "system_core_cpu_percent": core_percent,
            "gpu": {} if self._nvml is None else self._nvml.sample(),
        }
        self._previous_ns = now_ns
        self._previous_process_ns = process_ns
        self._previous_threads = thread_times
        self._previous_cores = cores
        result["sample_duration_ns"] = time.perf_counter_ns() - now_ns
        return result


def temporary_bytes(roots: Sequence[Path]) -> int:
    total = 0
    for root in roots:
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass  # Atomic publication can remove the sampled temporary path.
    return total

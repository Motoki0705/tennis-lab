"""Sampled device memory and renderer-context release, without importing CUDA."""

from __future__ import annotations

import csv
import os
import subprocess
import threading
from typing import Any, Protocol


class CacheTelemetry(Protocol):
    def begin(self) -> None: ...

    def end(self) -> dict[str, Any]: ...

    def released(self) -> dict[str, Any]: ...


def _query(fields: str, *, kind: str = "gpu") -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    )
    return [
        [v.strip() for v in row]
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]


class NvidiaL4Telemetry:
    """Device-wide sampled peak is a lower bound, not a Torch allocator peak."""

    def __init__(self, *, interval_seconds: float = 0.5) -> None:
        if not os.environ.get("COLAB_RELEASE_TAG"):
            raise RuntimeError("Colab rendering requires the Colab VM environment.")
        if interval_seconds <= 0:
            raise ValueError("GPU sampling interval must be positive.")
        self.interval = interval_seconds
        self.before = self._snapshot()
        if _query("pid,used_gpu_memory", kind="compute-apps"):
            raise RuntimeError(
                "L4 must have no existing compute processes before generation."
            )
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.samples: list[int] = []
        self.errors: list[Exception] = []

    @staticmethod
    def _snapshot() -> dict[str, Any]:
        rows = _query("uuid,name,memory.used")
        if len(rows) != 1 or rows[0][1] != "NVIDIA L4":
            raise RuntimeError(
                f"Generation requires exactly one NVIDIA L4; observed={rows}."
            )
        uuid, name, memory = rows[0]
        return {"uuid": uuid, "name": name, "device_used_bytes": int(memory) * 1024**2}

    def begin(self) -> None:
        if self.thread is not None:
            raise RuntimeError("GPU sampling is already active.")
        self.samples = [self._snapshot()["device_used_bytes"]]
        self.errors = []
        self.stop.clear()

        def sample() -> None:
            while not self.stop.wait(self.interval):
                try:
                    self.samples.append(self._snapshot()["device_used_bytes"])
                except Exception as error:
                    self.errors.append(error)
                    return

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def end(self) -> dict[str, Any]:
        if self.thread is None:
            raise RuntimeError("GPU sampling was not started.")
        self.stop.set()
        self.thread.join()
        self.thread = None
        if self.errors:
            raise RuntimeError("GPU memory measurement failed.") from self.errors[0]
        self.samples.append(self._snapshot()["device_used_bytes"])
        return {
            "source": "nvidia-smi_device_memory_used",
            "sampling_interval_seconds": self.interval,
            "sample_count": len(self.samples),
            "sampled_peak_bytes": max(self.samples),
            "peak_is_lower_bound": True,
            "torch_peak_allocated_bytes": None,
            "torch_peak_reserved_bytes": None,
            "allocator_measurement": "not_available_across_public_renderer_subprocess",
        }

    def released(self) -> dict[str, Any]:
        if self.thread is not None:
            self.end()
        after = self._snapshot()
        processes = _query("pid,used_gpu_memory", kind="compute-apps")
        if after["uuid"] != self.before["uuid"] or processes:
            raise RuntimeError(
                f"GPU release verification failed; compute_processes={processes}."
            )
        return {
            "schema": "court_scale_renderer_release_v1",
            "before": self.before,
            "after": after,
            "compute_processes_after": [],
            "renderer_context_released": True,
        }

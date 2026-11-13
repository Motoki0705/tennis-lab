"""Bound CUDA/RAM and stop when the physical D: drive has under 10 GiB free."""
import json
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from rebuild_dataset import write

GPU = "NVIDIA GeForce RTX 5060 Ti"
ALLOCATOR_FRACTION = .80
HOST_AVAILABLE_MIN_BYTES = 4 * 1024**3
D_DRIVE = Path("/mnt/d")
D_AVAILABLE_MIN_BYTES = 10 * 1024**3


def available_ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].strip().split()[0]) * 1024


def d_drive_available():
    stat = os.statvfs(D_DRIVE)
    return stat.f_bavail * stat.f_frsize


def filesystem_used(path):
    stat = os.statvfs(path)
    return (stat.f_blocks - stat.f_bfree) * stat.f_frsize


def violation(ram_available, disk_available):
    if ram_available < HOST_AVAILABLE_MIN_BYTES:
        return "host_available_ram"
    if disk_available < D_AVAILABLE_MIN_BYTES:
        return "d_drive_available_space"
    return None


def configure(root):
    import torch
    if not os.environ.get("TENNIS_RUN_ID"):
        raise ValueError("GPU execution requires shared training queue ownership")
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()).resolve()
    expected = str(common.parent / ".training_queue")
    if os.environ.get("TRAINING_QUEUE_DIR") != expected:
        raise ValueError("Explicit shared queue required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("TENNIS_SOURCE_COMMIT") != commit:
        raise ValueError("Queue execution source commit changed")
    ram, disk = available_ram(), d_drive_available()
    reason = violation(ram, disk)
    if reason:
        raise OSError("Resource guard failed before CUDA work: " + reason)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != GPU:
        raise ValueError("The authorized RTX 5060 Ti is required")
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_FRACTION)
    return dict(gpu=GPU, allocator_fraction=ALLOCATOR_FRACTION,
                allocator_limit_bytes=int(torch.cuda.get_device_properties(0).total_memory * ALLOCATOR_FRACTION),
                host_available_min_bytes=HOST_AVAILABLE_MIN_BYTES,
                d_drive_path=str(D_DRIVE), d_drive_available_bytes=disk,
                d_drive_available_min_bytes=D_AVAILABLE_MIN_BYTES,
                queue_id=os.environ["TENNIS_RUN_ID"], queue_directory=os.environ["TRAINING_QUEUE_DIR"],
                pid=os.getpid(), pgid=os.getpgrp())


@contextmanager
def host_guard(path, *, poll_interval=.25):
    stopped = threading.Event()
    status = dict(schema_version=2, minimum_available_bytes=available_ram(),
                  threshold_bytes=HOST_AVAILABLE_MIN_BYTES,
                  d_drive_path=str(D_DRIVE), minimum_d_drive_available_bytes=d_drive_available(),
                  d_drive_available_threshold_bytes=D_AVAILABLE_MIN_BYTES,
                  triggered=False, trigger_reason=None, filesystem_growth_is_telemetry_only=True)
    root = path.parent
    baseline = json.loads((root / "storage-baseline.json").read_text())["filesystem_used_bytes"]
    status["maximum_filesystem_growth_bytes"] = max(0, filesystem_used(root) - baseline)
    reason = violation(status["minimum_available_bytes"], status["minimum_d_drive_available_bytes"])
    if reason:
        status.update(triggered=True, trigger_reason=reason, timestamp=time.time())
        write(path, status)
        raise OSError("Resource guard failed at observer startup: " + reason)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise MemoryError("Resource guard or termination: " + (status["trigger_reason"] or "external_signal"))

    def observe():
        last_write = 0.
        while not stopped.wait(poll_interval):
            try:
                ram, disk = available_ram(), d_drive_available()
                status["minimum_available_bytes"] = min(status["minimum_available_bytes"], ram)
                status["minimum_d_drive_available_bytes"] = min(status["minimum_d_drive_available_bytes"], disk)
                growth = filesystem_used(root) - baseline
                status["maximum_filesystem_growth_bytes"] = max(status["maximum_filesystem_growth_bytes"], growth)
                reason = violation(ram, disk)
                if reason:
                    status.update(triggered=True, trigger_reason=reason, timestamp=time.time())
                if reason or time.monotonic() - last_write >= 1.:
                    status["updated_at"] = time.time()
                    write(path, status)
                    last_write = time.monotonic()
            except Exception as error:
                status.update(triggered=True, trigger_reason="resource_measurement_failed", error=str(error), timestamp=time.time())
                try:
                    write(path, status)
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
            if reason:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    signal.signal(signal.SIGTERM, interrupted)
    thread = threading.Thread(target=observe, name="host-resource-guard", daemon=True)
    thread.start()
    try:
        yield status
    finally:
        stopped.set()
        thread.join()
        signal.signal(signal.SIGTERM, previous)
        status["finished_at"] = time.time()
        write(path, status)

"""Bound WSL CUDA allocator and stop before exhausting host available RAM."""
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
STORAGE_LIMIT_BYTES = 20000000000


def available_ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].strip().split()[0]) * 1024


def filesystem_used(path):
    stat = os.statvfs(path)
    return (stat.f_blocks - stat.f_bfree) * stat.f_frsize


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
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != GPU:
        raise ValueError("The authorized RTX 5060 Ti is required")
    if available_ram() < HOST_AVAILABLE_MIN_BYTES:
        raise MemoryError("Host RAM guard failed before CUDA work")
    baseline = json.loads((root / "storage-baseline.json").read_text())["filesystem_used_bytes"]
    if filesystem_used(root) - baseline > STORAGE_LIMIT_BYTES - 1000000000:
        raise OSError("WSL growth leaves insufficient headroom under the 20 GB limit")
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_FRACTION)
    return dict(gpu=GPU, allocator_fraction=ALLOCATOR_FRACTION,
                allocator_limit_bytes=int(torch.cuda.get_device_properties(0).total_memory * ALLOCATOR_FRACTION),
                host_available_min_bytes=HOST_AVAILABLE_MIN_BYTES,
                queue_id=os.environ["TENNIS_RUN_ID"], queue_directory=os.environ["TRAINING_QUEUE_DIR"],
                pid=os.getpid(), pgid=os.getpgrp())


@contextmanager
def host_guard(path):
    stopped = threading.Event()
    status = dict(schema_version=1, minimum_available_bytes=available_ram(),
                  threshold_bytes=HOST_AVAILABLE_MIN_BYTES, triggered=False)
    root = path.parent
    baseline = json.loads((root / "storage-baseline.json").read_text())["filesystem_used_bytes"]
    status["maximum_filesystem_growth_bytes"] = max(0, filesystem_used(root) - baseline)
    status["storage_stop_threshold_bytes"] = STORAGE_LIMIT_BYTES - 1000000000
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise MemoryError("Host available RAM guard or termination signal")

    def observe():
        last_write = 0.
        while not stopped.wait(.25):
            value = available_ram()
            status["minimum_available_bytes"] = min(status["minimum_available_bytes"], value)
            growth = filesystem_used(root) - baseline
            status["maximum_filesystem_growth_bytes"] = max(status["maximum_filesystem_growth_bytes"], growth)
            if time.monotonic() - last_write >= 1.:
                status["updated_at"] = time.time()
                write(path, status)
                last_write = time.monotonic()
            if value < HOST_AVAILABLE_MIN_BYTES or growth > STORAGE_LIMIT_BYTES - 1000000000:
                status.update(triggered=True, timestamp=time.time())
                write(path, status)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    signal.signal(signal.SIGTERM, interrupted)
    thread = threading.Thread(target=observe, name="host-memory-guard", daemon=True)
    thread.start()
    try:
        yield status
    finally:
        stopped.set()
        thread.join()
        signal.signal(signal.SIGTERM, previous)
        status["finished_at"] = time.time()
        write(path, status)

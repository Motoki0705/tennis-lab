"""Wait between updates below 6 GiB RAM; stop below 2 GiB or 10 GiB D:."""
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
HOST_AVAILABLE_MIN_BYTES = 2 * 1024**3
RAM_ADMISSION_MIN_BYTES = 6 * 1024**3
LONG_WAIT_SECONDS = 30 * 60
D_DRIVE = Path("/mnt/d")
D_AVAILABLE_MIN_BYTES = 10 * 1024**3


def available_ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].strip().split()[0]) * 1024


def d_drive_available():
    stat = os.statvfs(D_DRIVE)
    return stat.f_bavail * stat.f_frsize


def process_tree_memory(*, proc=Path("/proc"), root_pid=None, page_size=None):
    """Sum RSS, including shared pages, for this process and recursive children."""
    root_pid = os.getpid() if root_pid is None else root_pid
    page_size = os.sysconf("SC_PAGE_SIZE") if page_size is None else page_size
    pending, seen, resident = [(root_pid, None)], set(), {}
    while pending:
        pid, expected_parent = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        directory = proc / str(pid)
        try:
            fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            if pid == root_pid:
                raise
            continue
        if expected_parent is not None and int(fields[1]) != expected_parent:
            continue  # An exited/reparented child or a reused PID is not ours.
        resident[pid] = int(fields[21]) * page_size
        try:
            tasks = list((directory / "task").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        for task in tasks:
            try:
                children = (task / "children").read_text().split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            pending.extend((int(child), pid) for child in children)
    parent = resident[root_pid]
    total = sum(resident.values())
    return dict(process_tree_rss_bytes=total, parent_rss_bytes=parent,
                children_rss_bytes=total - parent, process_tree_process_count=len(resident),
                process_tree_pids=sorted(resident))


def resource_snapshot():
    return dict(host_available_bytes=available_ram(),
                d_drive_available_bytes=d_drive_available(), **process_tree_memory())


class RamAdmission:
    """Wait before updates without consuming draws or imposing a wait deadline."""

    def __init__(self, path, *, poll_interval=1., sample=None, sleep=None, clock=None, wall=None):
        self.path, self.poll_interval = path, poll_interval
        self.sample, self.sleep = sample or resource_snapshot, sleep or time.sleep
        self.clock, self.wall = clock or time.monotonic, wall or time.time
        self.last_wait_seconds = 0.
        self.admitted = None

    def event(self, kind, update, started_at, started, sample, **extra):
        record = dict(schema_version=1, event=kind, update=update,
                      timestamp=self.wall(), started_at=started_at,
                      elapsed_seconds=self.clock() - started,
                      admission_threshold_bytes=RAM_ADMISSION_MIN_BYTES,
                      emergency_threshold_bytes=HOST_AVAILABLE_MIN_BYTES,
                      **sample, **extra)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(dict(notification="ram_admission", **record)), flush=True)

    def before_update(self, update):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.last_wait_seconds = 0.
        self.admitted = None
        waiting, noticed, sample = False, False, {}
        started, started_at = self.clock(), self.wall()
        try:
            while True:
                sample = dict(self.sample(), sampled_at=self.wall())
                reason = violation(sample["host_available_bytes"], sample["d_drive_available_bytes"])
                if reason:
                    raise MemoryError("Resource guard before update: " + reason)
                if sample["host_available_bytes"] >= RAM_ADMISSION_MIN_BYTES:
                    self.admitted = sample
                    if waiting:
                        self.last_wait_seconds = self.clock() - started
                        self.event("ram_wait_resumed", update, started_at, started, sample)
                    return
                if not waiting:
                    self.event("ram_wait_started", update, started_at, started, sample)
                    waiting = True
                if self.clock() - started > LONG_WAIT_SECONDS and not noticed:
                    self.event("ram_wait_over_30_minutes", update, started_at, started, sample)
                    noticed = True
                self.sleep(self.poll_interval)
        except BaseException as error:
            if waiting:
                self.event("ram_wait_interrupted", update, started_at, started, sample, error=str(error))
            raise

    def metrics(self):
        if self.admitted is None:
            raise RuntimeError("Update resource metrics require admission first")
        return dict(self.sample(), sampled_at=self.wall(), ram_wait_seconds=self.last_wait_seconds,
                    admitted_host_available_bytes=self.admitted["host_available_bytes"],
                    admitted_at=self.admitted["sampled_at"])


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
                ram_admission_min_bytes=RAM_ADMISSION_MIN_BYTES,
                d_drive_path=str(D_DRIVE), d_drive_available_bytes=disk,
                d_drive_available_min_bytes=D_AVAILABLE_MIN_BYTES,
                queue_id=os.environ["TENNIS_RUN_ID"], queue_directory=os.environ["TRAINING_QUEUE_DIR"],
                pid=os.getpid(), pgid=os.getpgrp())


@contextmanager
def host_guard(path, *, poll_interval=.25):
    stopped = threading.Event()
    status = dict(schema_version=3, minimum_available_bytes=available_ram(),
                  threshold_bytes=HOST_AVAILABLE_MIN_BYTES,
                  admission_threshold_bytes=RAM_ADMISSION_MIN_BYTES,
                  d_drive_path=str(D_DRIVE), minimum_d_drive_available_bytes=d_drive_available(),
                  d_drive_available_threshold_bytes=D_AVAILABLE_MIN_BYTES,
                  triggered=False, trigger_reason=None, filesystem_growth_is_telemetry_only=True)
    root = path.parent
    baseline = json.loads((root / "storage-baseline.json").read_text())["filesystem_used_bytes"]
    status["maximum_filesystem_growth_bytes"] = max(0, filesystem_used(root) - baseline)
    reason = violation(status["minimum_available_bytes"], status["minimum_d_drive_available_bytes"])
    if reason:
        status.update(triggered=True, trigger_reason=reason, timestamp=time.time(), **process_tree_memory())
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
                    status.update(triggered=True, trigger_reason=reason, timestamp=time.time(), **process_tree_memory())
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

"""Run an acceptance subprocess once and collect its diagnostics on failure."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path


def heartbeat(label):
    path = Path(os.environ["TENNIS_PRECISION_HEALTH"]) / "heartbeat"
    pending = path.with_suffix(".tmp")
    pending.write_text(label + "\n")
    pending.replace(path)


def worker(directory, command):
    os.environ["TENNIS_PRECISION_HEALTH"] = str(directory)
    sys.path.insert(0, str(Path.cwd()))
    sys.argv = command
    with (directory / "threads.log").open("w", buffering=1) as stream:
        faulthandler.enable(stream, all_threads=True)
        faulthandler.register(signal.SIGUSR1, stream, all_threads=True)
        heartbeat("worker import")
        try:
            runpy.run_path(command[0], run_name="__main__")
        except (Exception, KeyboardInterrupt):
            faulthandler.dump_traceback(stream, all_threads=True)
            raise
        finally:
            faulthandler.unregister(signal.SIGUSR1)
            faulthandler.disable()


def capture_gpu(directory):
    with (directory / "nvidia-smi-state.log").open("w") as stream:
        try:
            status = subprocess.run(
                ["nvidia-smi", "-q", "-d", "PERFORMANCE,CLOCK,POWER,TEMPERATURE"],
                stdout=stream, stderr=subprocess.STDOUT, timeout=10, check=False,
            )
            return status.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(f"{type(error).__name__}: {error}\n")
            return None


def supervise(directory, command):
    for key in ("TENNIS_RUN_ID", "TENNIS_REPRO_DIR"):
        if not os.environ.get(key):
            raise RuntimeError(f"Submit this command via the shared training queue ({key} is missing)")
    directory.mkdir(parents=True, exist_ok=False)
    beat = directory / "heartbeat"
    beat.touch()
    last_change = time.monotonic()
    last_stamp = beat.stat().st_mtime_ns
    process = subprocess.Popen([sys.executable, "-u", __file__, "--worker", str(directory), *command])
    failure = None
    try:
        while process.poll() is None:
            stamp = beat.stat().st_mtime_ns
            if stamp != last_stamp:
                last_change = time.monotonic()
                last_stamp = stamp
            elif time.monotonic() - last_change >= 180:
                failure = "heartbeat_timeout_180_seconds"
                break
            time.sleep(1)
        if failure is None and process.returncode == 0:
            (directory / "status.json").write_text('{"status": "complete"}\n')
            return 0
        if process.poll() is None:
            process.send_signal(signal.SIGUSR1)
        status = {"status": "failed", "reason": failure or "worker_failed",
                  "last_heartbeat": beat.read_text().strip(), "worker": process.pid,
                  "gpu_query_returncode": capture_gpu(directory)}
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        stack = directory / "threads.log"
        status.update(returncode=process.returncode, stack_bytes=stack.stat().st_size if stack.exists() else 0)
        (directory / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)
        return 1
    finally:
        if process.poll() is None:
            process.kill()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), sys.argv[3:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--health", required=True, type=Path)
        parser.add_argument("command", nargs=argparse.REMAINDER)
        args = parser.parse_args()
        raise SystemExit(supervise(args.health.resolve(), args.command))

"""CPU-only known immutable-memory audit; default 6 workers x 64 fixed passes.

Use .venv/bin/python; no checkpoint input or GPU/model imports. Logging and
provenance perform I/O, but the hashed input is created and reused in memory.
The independent software implementation is _sha256, not another OpenSSL alias.
No capability-mask intervention, global setting changes, or success retries.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

EXPECTED = "281e519df3077b557c6b03f5da83c4e8d397219259615dd7c3308f89cae8f2a6"
BUFFER_BYTES = 64 * 1024 * 1024
TIMEOUT_SECONDS = 240


def provenance() -> dict[str, Any]:
    import ssl

    return {
        "python": sys.version,
        "executable": sys.executable,
        "openssl": ssl.OPENSSL_VERSION,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "OPENSSL_ia32cap": os.environ.get("OPENSSL_ia32cap"),  # noqa: SIM112 -- OpenSSL spelling
    }


def emit(row: dict[str, Any]) -> None:
    print(json.dumps(row), flush=True)


def child(cpu: int, iterations: int) -> int:
    import _sha256
    import hashlib

    started = time.monotonic()
    os.sched_setaffinity(0, {cpu})
    emit(
        {
            "event": "start",
            "cpu": cpu,
            "iterations": iterations,
            "provenance": provenance(),
            "hashlib_type": str(type(hashlib.sha256())),
            "software_type": str(type(_sha256.sha256())),
            "software_implementation": "independent _sha256 software implementation",
        }
    )
    buffer = bytes(range(256)) * (BUFFER_BYTES // 256)
    passed = True
    for iteration in range(iterations):
        iteration_started = time.monotonic()
        openssl = hashlib.sha256(buffer).hexdigest()
        software = _sha256.sha256(buffer).hexdigest()
        affinity = sorted(os.sched_getaffinity(0))
        match = openssl == software == EXPECTED and affinity == [cpu]
        passed = passed and match
        emit(
            {
                "event": "iteration",
                "cpu": cpu,
                "iteration": iteration,
                "hashlib": openssl,
                "_sha256": software,
                "expected": EXPECTED,
                "match": match,
                "affinity": affinity,
                "iteration_seconds": time.monotonic() - iteration_started,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    emit(
        {
            "event": "complete",
            "iterations_completed": iterations,
            "passed": passed,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return 0 if passed else 1


def validate(log: Path, cpu: int, iterations: int) -> dict[str, Any]:
    try:
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        start = rows[0]
        end = rows[-1]
        samples = rows[1:-1]
        complete = (
            len(rows) == iterations + 2
            and start["event"] == "start"
            and start["cpu"] == cpu
            and start["iterations"] == iterations
            and start["provenance"]["affinity"] == [cpu]
            and end["event"] == "complete"
            and end["iterations_completed"] == iterations
            and all(
                row["event"] == "iteration"
                and row["cpu"] == cpu
                and row["iteration"] == index
                for index, row in enumerate(samples)
            )
        )
        passed = (
            complete
            and end["passed"] is True
            and all(
                row["hashlib"] == row["_sha256"] == row["expected"] == EXPECTED
                and row["affinity"] == [cpu]
                and row["match"] is True
                for row in samples
            )
        )
        return {
            "complete": complete,
            "passed": passed,
            "iteration_records": sum(row.get("event") == "iteration" for row in rows),
        }
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, OSError):
        return {
            "complete": False,
            "passed": False,
            "validation_error": traceback.format_exc(),
        }


def run(output: Path, iterations: int, workers: int) -> int:
    started = time.monotonic()
    # Exclusive directory creation prevents competing writers and reuse of any logs.
    output.mkdir(parents=True, exist_ok=False)
    data: dict[str, Any] = {
        "status": "running",
        "provenance": provenance(),
        "iterations_per_worker": iterations,
        "timeout_seconds": TIMEOUT_SECONDS,
        "buffer_bytes": BUFFER_BYTES,
        "expected": EXPECTED,
        "pattern": "bytes(range(256)) repeated; one immutable bytes per child",
        "workers": [],
    }
    processes: list[subprocess.Popen[bytes]] = []

    def save() -> None:
        data["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "results.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(output / "results.json")

    save()
    try:
        cpus = sorted(os.sched_getaffinity(0))[:workers]
        data["selected_cpus"] = cpus
        for cpu in cpus:
            if time.monotonic() - started >= TIMEOUT_SECONDS:
                raise TimeoutError(
                    "240 second internal deadline exhausted during launch"
                )
            log = output / f"cpu-{cpu}.jsonl"
            error_log = output / f"cpu-{cpu}.stderr.log"
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--child-cpu",
                str(cpu),
                "--iterations",
                str(iterations),
            ]
            with log.open("xb") as stdout, error_log.open("xb") as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            processes.append(process)
            data["workers"].append(
                {
                    "cpu": cpu,
                    "pid": process.pid,
                    "command": command,
                    "log": str(log),
                    "stderr_log": str(error_log),
                    "returncode": None,
                }
            )
            save()
        while any(process.poll() is None for process in processes):
            if time.monotonic() - started >= TIMEOUT_SECONDS:
                raise TimeoutError("240 second internal deadline exhausted")
            for process, row in zip(processes, data["workers"], strict=True):
                row["returncode"] = process.poll()
            save()
            time.sleep(0.1)
        data["status"] = "finished"
    except BaseException:
        data["status"] = "error"
        data["error"] = traceback.format_exc()
    finally:
        # Children retain the queue PGID. Stop only these children, never that group.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        teardown_deadline = time.monotonic() + 3
        for process in processes:
            try:
                process.wait(timeout=max(0, teardown_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for process, row in zip(processes, data["workers"], strict=True):
            row["returncode"] = process.returncode
            row.update(validate(Path(row["log"]), row["cpu"], iterations))
        if data["status"] == "finished":
            data["status"] = (
                "passed"
                if (
                    len(processes) == len(data["selected_cpus"]) > 0
                    and all(
                        row["passed"] and row["returncode"] == 0
                        for row in data["workers"]
                    )
                )
                else "failed"
            )
        save()
    return 0 if data["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--iterations",
        type=int,
        choices=(1, 64),
        default=64,
        help="64 for audit; 1 only for smoke testing",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--child-cpu", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_cpu is not None:
        return child(args.child_cpu, args.iterations)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    return run(args.output_dir.resolve(), args.iterations, args.workers)


if __name__ == "__main__":
    sys.exit(main())

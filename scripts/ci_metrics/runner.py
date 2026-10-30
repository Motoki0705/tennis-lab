"""Run the complete CPU selection with optional coordinated measurement locking."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

IDENTITY_KEYS = {
    "CODEX_SESSION_ID",
    "CODEX_THREAD_ID",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_PARENT_SESSION_ID",
    "CLAUDE_CODE_AGENT_ID",
}
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def test_environment(environment: dict[str, str]) -> dict[str, str]:
    result = {
        key: value
        for key, value in environment.items()
        if key not in IDENTITY_KEYS and not key.startswith(("HERDR_", "PYTEST_XDIST_"))
    }
    # Nested pytest starts must not inherit the outer measurement plugin/options.
    result.pop("PYTEST_ADDOPTS", None)
    result.pop("PYTEST_CURRENT_TEST", None)
    result.update({key: "1" for key in THREAD_KEYS})
    result["CUDA_VISIBLE_DEVICES"] = ""
    return result


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.PIPE)


def environment_metadata(workers: int, command: list[str]) -> dict[str, Any]:
    source: dict[str, Any]
    try:
        untracked = _git("ls-files", "--others", "--exclude-standard", "-z")
        source = {
            "commit": _git("rev-parse", "HEAD").decode().strip(),
            "status": _git("status", "--porcelain").decode(),
            "tracked_diff_sha256": hashlib.sha256(
                _git("diff", "HEAD", "--binary")
            ).hexdigest(),
            "untracked_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in untracked.decode().split("\0")
                if name and Path(name).is_file()
            },
        }
    except subprocess.CalledProcessError:
        source = {"commit": None, "error": "not a git checkout"}
    packages: dict[str, str | None] = {}
    for name in ("pytest", "pytest-xdist", "torch", "numpy", "spin"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    return {
        "schema_version": 1,
        "created_at": time.time(),
        "source": source,
        "command": command,
        "workers": workers,
        "python": sys.version,
        "platform": platform.platform(),
        "os_release": platform.freedesktop_os_release(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu.splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        "memory_total_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "load_average_at_start": list(os.getloadavg()),
        "packages": packages,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_SHA",
                "GITHUB_JOB",
                "RUNNER_OS",
                "RUNNER_ARCH",
                "ImageOS",
                "ImageVersion",
                "UV_CACHE_HIT",
            )
        },
        "test_environment": {
            "CUDA_VISIBLE_DEVICES": "",
            **dict.fromkeys(THREAD_KEYS, "1"),
        },
        "memory_metric": "largest individual pytest process high-water RSS, Linux KiB; not process-tree peak",
        "cache_condition": os.environ.get("UV_CACHE_HIT", "unknown"),
        "elapsed_clock": "time.perf_counter; monotonic on this host",
    }


def run(
    output: Path,
    workers: int,
    extra_args: list[str],
    lock: Path | None,
    shard: tuple[int, int, Path] | None = None,
    lane_plan: Path | None = None,
) -> int:
    from .lanes import load_registry, make_plan, read_plan
    from .report import summarize

    if lane_plan is not None and shard is None:
        raise ValueError("--lane-plan requires an explicit shard configuration")
    if lane_plan is not None and extra_args:
        raise ValueError(
            "lane plans cannot be combined with arbitrary pytest arguments"
        )

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"measurement directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    command = [
        sys.executable,
        "-m",
        "spin",
        "test",
        "-q",
        "--no-cov",
        "--dist=worksteal",
        "-n",
        str(workers),
        "--durations=50",
        "--durations-min=1",
        "-p",
        "scripts.ci_metrics.pytest_plugin",
        f"--ci-metrics-dir={output}",
        f"--junitxml={output / 'junit.xml'}",
        *extra_args,
    ]
    shard_metadata = None
    if shard is not None:
        from .shards import read_timing_profile

        index, count, profile = shard
        if not 0 <= index < count <= 32:
            raise ValueError("invalid shard index/count")
        _, profile_digest = read_timing_profile(profile)
        command.extend(
            [
                "-p",
                "scripts.ci_metrics.shard_plugin",
                "--ci-shard-index",
                str(index),
                "--ci-shard-count",
                str(count),
                "--ci-shard-weights",
                str(profile),
            ]
        )
        shard_metadata = {
            "index": index,
            "count": count,
            "profile_sha256": profile_digest,
        }
    metadata = environment_metadata(workers, command)
    plan = None
    if shard is not None:
        if extra_args:
            raise ValueError("CPU sharding requires the default CPU target")
        if lane_plan is not None:
            plan = read_plan(lane_plan, root=Path.cwd())
        else:
            registry, files = load_registry(Path.cwd())
            plan = make_plan(
                registry,
                files,
                source_sha=metadata["environment"].get("GITHUB_SHA")
                or metadata["source"].get("commit"),
                event="local",
                run_id=metadata["environment"].get("GITHUB_RUN_ID"),
                attempt=metadata["environment"].get("GITHUB_RUN_ATTEMPT"),
            )
        if plan["source_sha"] != (
            metadata["environment"].get("GITHUB_SHA")
            or metadata["source"].get("commit")
        ):
            raise ValueError("lane plan source revision differs from the measurement")
        for key, variable in (
            ("run_id", "GITHUB_RUN_ID"),
            ("attempt", "GITHUB_RUN_ATTEMPT"),
        ):
            if plan[key] != metadata["environment"].get(variable):
                raise ValueError("lane plan run/attempt differs from the measurement")
        write_json(output / "lane-plan.json", plan)
        command.extend(["--ci-lane-plan", str(output / "lane-plan.json")])
    metadata["lane_plan_id"] = plan["plan_id"] if plan else None
    metadata["full_cpu_target"] = not extra_args and shard is None
    metadata["cpu_target_before_sharding"] = not extra_args
    metadata["shard"] = shard_metadata
    metadata["lock_path"] = str(lock) if lock else None
    write_json(output / "environment.json", metadata)
    state: dict[str, Any] = {
        "requested_at": time.time(),
        "requested_monotonic_s": time.perf_counter(),
        "status": "waiting_for_lock",
    }
    process: subprocess.Popen[bytes] | None = None
    received_signal: int | None = None

    def interrupted(signum: int, frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signum)
        elif process is None:
            raise InterruptedError(f"signal {signum} while waiting for lock")

    previous = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    lock_stream = None
    # No test command has an exit status until it actually starts and finishes.
    exitcode: int | None = None
    try:
        write_json(output / "run.json", state)
        if lock is not None:
            if not lock.is_absolute():
                raise ValueError("pytest lock must be absolute")
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock_stream = lock.open("a")
            fcntl.flock(lock_stream, fcntl.LOCK_EX)
        acquired_metadata = environment_metadata(workers, command)
        if acquired_metadata["source"] != metadata["source"]:
            raise RuntimeError(
                "source changed while waiting for pytest lock; start a fresh measurement"
            )
        metadata.update(acquired_metadata)
        write_json(output / "environment.json", metadata)
        state.update(
            started_at=time.time(),
            started_monotonic_s=time.perf_counter(),
            status="running",
        )
        state["lock_wait_s"] = (
            state["started_monotonic_s"] - state["requested_monotonic_s"]
        )
        write_json(output / "run.json", state)
        with (output / "pytest.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                env=test_environment(dict(os.environ)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                start_new_session=True,
            )
            state["process_group_id"] = process.pid
            write_json(output / "run.json", state)
            # Keep the live job log consumed by ci_watch as well as the artifact.
            # An unbuffered pipe forwards available bytes without waiting for a
            # full buffer or newline; cancellation still targets the whole group.
            assert process.stdout is not None
            with process.stdout:
                while chunk := process.stdout.read(65536):
                    log.write(chunk)
                    log.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            exitcode = process.wait()
            state["command_finished_at"] = time.time()
            state["command_finished_monotonic_s"] = time.perf_counter()
    except InterruptedError:
        pass
    except BaseException as error:
        state["launcher_error"] = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if lock_stream is not None:
            lock_stream.close()
        state["source_at_finish"] = environment_metadata(workers, command)["source"]
        state["source_unchanged"] = state["source_at_finish"] == metadata["source"]
        state.update(
            finished_at=time.time(),
            finished_monotonic_s=time.perf_counter(),
            load_average_at_finish=list(os.getloadavg()),
            exitcode=exitcode,
            signal=received_signal,
            status="cancelled"
            if received_signal
            else "launcher_error"
            if state.get("launcher_error")
            else "completed",
        )
        write_json(output / "run.json", state)
        summary = summarize(output)
        print((output / "summary.md").read_text())
    if received_signal:
        return 128 + received_signal
    if exitcode == 0 and not summary["successful_test_baseline"]:
        return 2
    if exitcode is None:
        return 2
    return exitcode if exitcode >= 0 else 128 - exitcode

"""Bounded process primitives and honest, fail-closed Linux capability inspection.

There is deliberately no production CPU-test backend until aggregate memory,
device and credential-file isolation have been implemented and verified.
"""

from __future__ import annotations

import contextlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

TIMEOUT_SECONDS = 15 * 60
MEMORY_BYTES = 4 * 1024**3
LOG_BYTES = 8 * 1024**2


def clean_environment() -> dict[str, str]:
    """Construct an allowlist; never copy tokens, agent configuration or proxies."""
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": "/nonexistent",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int
    reason: str
    elapsed_seconds: float
    log_bytes: int


def run_bounded(
    argv: list[str],
    *,
    cwd: Path,
    log: Path,
    env: dict[str, str],
    timeout: float = TIMEOUT_SECONDS,
    max_log_bytes: int = LOG_BYTES,
) -> ProcessResult:
    """Kill the process group even after a leader exits or closes its output.

    This does not claim to catch descendants that escape into another session.
    A future CPU backend must additionally contain those in a PID namespace or
    cgroup. Only trusted host code builds argv; this is not an agent tool.
    """
    if not 0 < timeout <= TIMEOUT_SECONDS or not 0 < max_log_bytes <= LOG_BYTES:
        raise ValueError("process limits must be positive and within policy")
    started = time.monotonic()
    reason = "exited"
    count = 0
    log_fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with (
        os.fdopen(log_fd, "wb") as output,
        subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        ) as proc,
    ):
        try:
            assert proc.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while selector.get_map() or proc.poll() is None:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        reason = "timeout"
                        break
                    for key, _ in selector.select(min(remaining, 0.05)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        accepted = chunk[: max_log_bytes - count]
                        output.write(accepted)
                        count += len(accepted)
                        if len(accepted) < len(chunk):
                            reason = "log_limit"
                            break
                    if reason != "exited":
                        break
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            output.flush()
            os.fsync(output.fileno())
    return ProcessResult(proc.returncode, reason, time.monotonic() - started, count)


@dataclass(frozen=True)
class TestJob:
    checkout: Path
    nodeids: tuple[str, ...]
    pytest_lock: Path
    log: Path

    @property
    def argv(self) -> list[str]:
        return [
            "/usr/bin/bash",
            str(self.checkout / ".agents/ops/pytest_locked.sh"),
            *self.nodeids,
            "-q",
            "-p",
            "xdist.plugin",
            "-p",
            "no:cacheprovider",
            "-m",
            "not cuda",
        ]

    @property
    def env(self) -> dict[str, str]:
        return {**clean_environment(), "PYTEST_LOCK": str(self.pytest_lock)}


class TestRunner(Protocol):
    def inspect(self) -> dict[str, Any]: ...

    def run(self, job: TestJob) -> ProcessResult: ...


class LinuxRunner:
    """Inspection only. No flag, environment override or CLI can enable execution."""

    def inspect(self) -> dict[str, Any]:
        membership = Path("/proc/self/cgroup").read_text().strip()
        return {
            "enabled": False,
            "backend": "linux-preflight-only",
            "required_limits": {
                "timeout_seconds": TIMEOUT_SECONDS,
                "memory_bytes": MEMORY_BYTES,
                "concurrent_runs": 1,
                "log_bytes": LOG_BYTES,
            },
            "implemented_primitives": [
                "environment_allowlist",
                "bounded_process_group",
            ],
            "unverified": {
                "aggregate_memory": (
                    "No verified per-run cgroup backend. RLIMIT_AS/prlimit alone "
                    "is per process and cannot bound pytest workers and descendants "
                    f"to 4 GiB in aggregate. Current membership: {membership}"
                ),
                "credential_files": "No verified minimal filesystem for test execution",
                "gpu": "Environment masking alone does not deny GPU device access",
                "network": "A namespace probe does not install a CPU-test sandbox",
                "descendants": "No verified PID/cgroup containment for escaping descendants",
            },
        }

    def run(self, job: TestJob) -> ProcessResult:
        raise RuntimeError("CPU-test execution is disabled: isolation is unverified")


def probe(directory: Path) -> dict[str, Any]:
    """Run fixed, small namespace/rlimit probes without cgroup or systemd writes."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    results: dict[str, Any] = {"execution": LinuxRunner().inspect(), "probes": {}}
    scripts = {
        "namespaces": [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--mount",
            "--pid",
            "--fork",
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import json,os,socket; "
            "s=socket.socket(); s.settimeout(0.2); "
            "code=s.connect_ex(('192.0.2.1',9)); "
            "print(json.dumps({'pid':os.getpid(),'network_namespace':"
            "os.readlink('/proc/self/ns/net'),'connect_errno':code})); "
            "assert os.getpid()==1 and code!=0",
        ],
        "per_process_memory": [
            "prlimit",
            "--as=67108864:67108864",
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import resource; print(resource.getrlimit(resource.RLIMIT_AS)); "
            "bytearray(128*1024*1024)",
        ],
    }
    for name, argv in scripts.items():
        log = directory / f"{name}.log"
        if not shutil.which(argv[0], path="/usr/bin:/bin"):
            results["probes"][name] = {"available": False, "reason": "binary missing"}
            continue
        result = run_bounded(
            argv, cwd=directory, log=log, env=clean_environment(), timeout=5
        )
        results["probes"][name] = {
            **asdict(result),
            "output": log.read_text(errors="replace"),
            "scope": "fixed probe only; not a verified CPU-test backend",
        }
    results["host_network_namespace"] = os.readlink("/proc/self/ns/net")
    (directory / "probe.json").write_text(json.dumps(results, indent=2) + "\n")
    return results

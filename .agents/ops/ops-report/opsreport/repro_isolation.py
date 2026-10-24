"""Bounded processes and the proof-gated Linux CPU-test backend."""

from __future__ import annotations

import contextlib
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
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
    cleanup_complete: bool | None = None
    scope: str | None = None
    attestation: dict[str, Any] | None = None


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

    This primitive does not confirm cleanup of descendants in another session.
    The CPU backend adds PID namespace/cgroup containment and confirms cleanup.
    Only trusted host code builds argv; this is not an agent tool.
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
    """Only a matching, successful OS probe can enable the execution backend."""

    def __init__(self, proof: Path | None = None) -> None:
        self.proof = proof

    def inspect(self) -> dict[str, Any]:
        from .repro_linux import inspect_proof

        verified, reason = inspect_proof(self.proof)
        return {
            "enabled": verified,
            "backend": "linux-systemd-namespaces",
            "required_limits": {
                "timeout_seconds": TIMEOUT_SECONDS,
                "memory_bytes": MEMORY_BYTES,
                "concurrent_runs": 1,
                "log_bytes": LOG_BYTES,
            },
            "enforcement": [
                "environment_allowlist",
                "systemd_scope_memory_and_swap",
                "systemd_scope_deadline_and_cleanup",
                "user_mount_network_pid_namespaces",
                "minimal_chroot_and_readonly_runtime",
                "empty_gpu_devices_and_hidden_host_home",
                "capability_drop_and_no_new_privileges",
                "pytest_lock_outside_sandbox",
            ],
            "unverified": {}
            if verified
            else dict.fromkeys(
                (
                    "aggregate_memory",
                    "credential_files",
                    "gpu",
                    "network",
                    "descendants",
                ),
                reason,
            ),
            "proof": str(self.proof) if self.proof else None,
        }

    def run(self, job: TestJob) -> ProcessResult:
        from .repro_linux import run_job

        if self.inspect()["enabled"] is not True:
            raise RuntimeError(
                "CPU-test execution is disabled: isolation is unverified"
            )
        return run_job(job)


def probe(directory: Path) -> dict[str, Any]:
    """Exercise the same backend using fixed, small enforcement challenges."""
    from .repro_linux import probe_backend

    return probe_backend(directory)

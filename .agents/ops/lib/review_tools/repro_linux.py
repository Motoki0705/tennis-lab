"""Host-side scope lifecycle and fixed OS enforcement challenges.

The baseline pytest wrapper runs outside the sandbox. Its interpreter is a
temporary trusted launcher; only that launcher can start the contained pytest.
No scope, namespace, filesystem or limit options come from an agent request.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .repro_isolation import (
    MEMORY_BYTES,
    TIMEOUT_SECONDS,
    ProcessResult,
    TestJob,
    clean_environment,
    run_bounded,
)

BOOTSTRAP = Path(__file__).with_name("repro_sandbox.py").resolve()
ROOT = Path(__file__).resolve().parents[4]
CHALLENGES = ("isolation", "memory", "descendants", "timeout", "closed-output")
PROOF_MAX_AGE = 24 * 60 * 60


def private_json(path: Path, value: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def runtime() -> dict[str, Any]:
    venv = (ROOT / ".venv").resolve(strict=True)
    python = (venv / "bin/python").resolve(strict=True)
    base = python.parent.parent
    # uv's pyvenv.cfg can use an unversioned alias for the same interpreter.
    config = (venv / "pyvenv.cfg").read_text()
    homes = [
        line.split("=", 1)[1].strip()
        for line in config.splitlines()
        if line.startswith("home =")
    ]
    targets = {str(base)}
    for home in homes:
        alias = Path(home).parent
        if alias.resolve(strict=True) != base:
            raise RuntimeError("virtualenv interpreter and pyvenv.cfg disagree")
        targets.add(str(alias))
    if not (base / "lib").is_dir():
        raise RuntimeError("a standalone Python runtime is required")
    return {
        "venv": str(venv),
        "runtime": str(base),
        "runtime_targets": sorted(targets),
        "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "pyvenv_cfg_sha256": hashlib.sha256(config.encode()).hexdigest(),
    }


def fingerprint() -> dict[str, Any]:
    files = [
        Path(__file__),
        BOOTSTRAP,
        Path(__file__).with_name("repro_isolation.py"),
        ROOT / ".agents/ops/pytest_locked.sh",
    ]
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uid": os.getuid(),
        "code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "runtime": runtime(),
        "memory_bytes": MEMORY_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
    }


def inspect_proof(path: Path | None) -> tuple[bool, str]:
    if path is None:
        return False, "No verified OS proof was supplied; run the fixed backend probe"
    try:
        info = path.stat()
        if path.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("OS proof must be a private, host-owned regular file")
        if not path.is_file() or info.st_size > 128 * 1024:
            raise ValueError("invalid OS proof file")
        proof = json.loads(path.read_bytes())
        if proof["schema_version"] != 1 or proof["verified"] is not True:
            raise ValueError("OS enforcement challenges did not all pass")
        if proof["fingerprint"] != fingerprint():
            raise ValueError(
                "OS proof belongs to another boot, user, backend or runtime"
            )
        age = time.time() - proof["verified_at_epoch"]
        if not 0 <= age <= PROOF_MAX_AGE:
            raise ValueError(
                "OS proof is stale or from the future (maximum age: 24 hours)"
            )
        for name in CHALLENGES:
            row = proof["challenges"][name]
            if (
                row["passed"] is not True
                or row["process"]["cleanup_complete"] is not True
            ):
                raise ValueError(f"OS challenge is incomplete: {name}")
        return True, ""
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        return False, str(exc)


def manager_environment() -> dict[str, str]:
    # Only the host launcher sees the user manager socket, never test code.
    directory = f"/run/user/{os.getuid()}"
    return {
        **clean_environment(),
        "XDG_RUNTIME_DIR": directory,
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + directory + "/bus",
    }


def manager(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/systemctl", "--user", *args],
        env=manager_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def stop_scope(unit: str) -> bool:
    """Stop only our fresh transient scope and require no remaining members."""
    if not unit.startswith("tennis-audit-cpu-") or not unit.endswith(".scope"):
        raise ValueError("unexpected transient scope name")
    result = manager("show", unit, "--property=LoadState,ActiveState,ControlGroup")
    properties = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if properties.get("LoadState") == "not-found":
        return True
    if result.returncode != 0 or not properties.get("LoadState"):
        return False
    if (
        properties.get("ActiveState") not in {"inactive", "failed"}
        and manager("stop", unit).returncode != 0
    ):
        return False
    group = properties.get("ControlGroup", "")
    if group:
        events = Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.events"
        if events.exists() and "populated 0" not in events.read_text().splitlines():
            return False
    final = manager("show", unit, "--property=LoadState,ActiveState")
    return "LoadState=not-found" in final.stdout or (
        final.returncode == 0
        and any(
            s in final.stdout.splitlines()
            for s in ("ActiveState=inactive", "ActiveState=failed")
        )
    )


def scope_argv(unit: str, config: Path, *, memory: int, deadline: int) -> list[str]:
    if not 0 < memory <= MEMORY_BYTES or not 0 < deadline <= TIMEOUT_SECONDS:
        raise ValueError("scope limits exceed the fixed policy")
    return [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--unit=" + unit,
        "-p",
        f"MemoryMax={memory}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=continue",
        "-p",
        "KillSignal=SIGKILL",
        "-p",
        "TimeoutStopSec=1s",
        "-p",
        f"RuntimeMaxSec={deadline}s",
        "-p",
        "TasksMax=256",
        "/usr/bin/unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "--net",
        "--pid",
        "--fork",
        "--kill-child=KILL",
        "/usr/bin/python3",
        "-I",
        "-S",
        str(BOOTSTRAP),
        "bootstrap",
        str(config),
    ]


def hidden_paths(extra: list[str]) -> list[str]:
    home = Path.home()
    return [
        str(home / p)
        for p in (
            ".config/gh",
            ".ssh",
            ".codex",
            ".claude",
            ".config/tennis-lab-agents",
            ".aws",
            ".netrc",
        )
    ] + [
        str(ROOT),
        f"/run/user/{os.getuid()}/bus",
        "/mnt/c",
        *extra,
    ]


def execute(
    directory: Path,
    checkout: Path,
    log: Path,
    *,
    mode: str,
    memory: int = MEMORY_BYTES,
    deadline: int = TIMEOUT_SECONDS,
    watchdog: float = TIMEOUT_SECONDS,
    job: TestJob | None = None,
    host_port: int = 9,
    hidden: list[str] | None = None,
) -> tuple[ProcessResult, dict[str, Any]]:
    unit = "tennis-audit-cpu-" + uuid.uuid4().hex + ".scope"
    directory.mkdir(mode=0o700)
    root = directory / "root"
    root.mkdir(mode=0o700)
    attestation = directory / "attestation.json"
    environment = {**clean_environment(), "HOME": "/home/sandbox", "TMPDIR": "/tmp"}
    config = {
        **runtime(),
        "unit": unit,
        "root": str(root),
        "checkout": str(checkout),
        "memory_bytes": memory,
        "mode": mode,
        "attestation": str(attestation),
        "environment": environment,
        "host_network": os.readlink("/proc/self/ns/net"),
        "hidden_paths": hidden_paths(hidden or []),
        "host_port": host_port,
        "pytest_argv": [
            "/checkout/.venv/bin/python",
            "-m",
            "pytest",
            "-n",
            "4",
            *(job.argv[2:] if job else []),
        ],
    }
    config_path = directory / "config.json"
    private_json(config_path, config)
    argv = scope_argv(unit, config_path, memory=memory, deadline=deadline)
    launcher: Path | None = None
    launcher_raw = b""
    try:
        if job:
            launcher = checkout / ".venv"
            launcher.mkdir(mode=0o700)  # Never replace a tracked/preexisting path.
            (launcher / "bin").mkdir(mode=0o700)
            expected = ["-m", "pytest", "-n", "4", *job.argv[2:]]
            launcher_raw = (
                "#!/usr/bin/python3\nimport json,os,sys\n"
                + f"expected=json.loads({json.dumps(expected)!r})\n"
                + "if sys.argv[1:] != expected: raise SystemExit('unexpected pytest-wrapper arguments')\n"
                + f"argv=json.loads({json.dumps(argv)!r})\n"
                + f"env=json.loads({json.dumps(manager_environment())!r})\n"
                + "os.execve(argv[0],argv,env)\n"
            ).encode()
            (launcher / "bin/python").write_bytes(launcher_raw)
            (launcher / "bin/python").chmod(0o700)
            argv = job.argv
        process = run_bounded(
            argv,
            cwd=checkout,
            log=log,
            env=job.env if job else manager_environment(),
            timeout=watchdog,
        )
    finally:
        try:
            complete = stop_scope(unit)
        except (OSError, ValueError, subprocess.SubprocessError):
            complete = False
        if not complete:
            raise RuntimeError(f"scope cleanup could not be verified: {unit}")
        if launcher is not None:
            if (launcher / "bin/python").read_bytes() != launcher_raw or sorted(
                p.relative_to(launcher).as_posix() for p in launcher.rglob("*")
            ) != ["bin", "bin/python"]:
                raise RuntimeError("host pytest launcher changed; checkout retained")
            shutil.rmtree(launcher)
        # Mounts exist only in the now-dead namespace; do not traverse a live root.
        if list(root.iterdir()):
            raise RuntimeError("sandbox mounts survived their namespace")
        root.rmdir()
    observed = json.loads(attestation.read_bytes()) if attestation.exists() else {}
    process = replace(
        process, cleanup_complete=complete, scope=unit, attestation=observed
    )
    if not observed.get("verified"):
        raise RuntimeError(
            "sandbox did not produce its trusted pre-execution attestation; see log"
        )
    return process, observed


def run_job(job: TestJob) -> ProcessResult:
    process, _ = execute(
        job.log.parent / "sandbox", job.checkout, job.log, mode="pytest", job=job
    )
    return process


def probe_backend(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    checkout = directory / "checkout"
    checkout.mkdir(mode=0o700)
    sentinel = directory / "credential-sentinel"
    sentinel.write_text("nonsecret isolation sentinel\n")
    proof: dict[str, Any] = {
        "schema_version": 1,
        "verified": False,
        "fingerprint": fingerprint(),
        "verified_at_epoch": time.time(),
        "challenges": {},
    }
    # Only an ephemeral localhost port; never contact any existing service.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        for mode in CHALLENGES:
            try:
                process, observed = execute(
                    directory / mode,
                    checkout,
                    directory / (mode + ".log"),
                    mode=mode,
                    memory=192 * 1024**2 if mode == "memory" else MEMORY_BYTES,
                    deadline=1 if mode == "timeout" else 30,
                    watchdog=1 if mode == "closed-output" else 20,
                    host_port=port,
                    hidden=[str(sentinel)],
                )
                output = (directory / (mode + ".log")).read_text(errors="replace")
                if mode == "timeout":
                    passed = (
                        process.reason == "exited"
                        and process.exit_code != 0
                        and "detached descendant started" in output
                        and process.elapsed_seconds < 10
                    )
                elif mode == "closed-output":
                    passed = (
                        process.reason == "timeout"
                        and "detached descendant started" in output
                    )
                else:
                    passed = (
                        process.reason == "exited"
                        and process.exit_code == 0
                        and "challenge passed" in output
                    )
                    if mode == "memory":
                        passed = passed and '"aggregate_oom": true' in output
                proof["challenges"][mode] = {
                    "passed": passed,
                    "process": asdict(process),
                    "attestation": observed,
                }
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                proof["challenges"][mode] = {"passed": False, "error": str(exc)}
    proof["verified"] = all(row["passed"] for row in proof["challenges"].values())
    proof["verified_at_epoch"] = time.time()
    private_json(directory / "probe.json", proof)
    return proof

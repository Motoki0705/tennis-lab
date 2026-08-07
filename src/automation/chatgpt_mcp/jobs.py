"""Durable, exact-revision Docker jobs and `.training_queue` admission."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import shlex
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.automation.chatgpt_mcp.settings import GatewaySettings
from src.automation.chatgpt_mcp.storage import SqliteStore
from src.automation.chatgpt_mcp.workspace import RevisionWorkspace, WorkspaceManager

_JOB_ID = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_TRAINING_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_QUEUE_FILE = re.compile(r"^[A-Za-z0-9._-]{1,240}\.job$")
_DIRECT_COMMAND_MAX_SECONDS = 30 * 60
_MAX_CONCURRENT_DIRECT_JOBS = 2
_JOB_METADATA_TTL_SECONDS = 30 * 24 * 3600
_TRAINING_METADATA_TTL_SECONDS = 90 * 24 * 3600
_RUNTIME_ROOT = Path(__file__).resolve().parents[3]

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
)


class ExecutionWorkspaceResolver(Protocol):
    """Minimal exact-revision resolver required by the execution plane."""

    def assert_execution_ready(
        self, *, workspace_id: str, expected_sha: str
    ) -> RevisionWorkspace:
        """Return a clean source workspace bound to the supplied SHA."""
        ...


class JobError(RuntimeError):
    """Raised for invalid job state or a failed sandbox operation."""


class SandboxSpec(BaseModel):
    """Private execution contract for one exact, immutable source revision."""

    job_id: str = Field(pattern=_JOB_ID.pattern)
    command: str = Field(min_length=1, max_length=100_000)
    workspace_id: str
    expected_sha: str = Field(pattern=_SHA.pattern)
    use_gpu: bool = False
    timeout_seconds: int = Field(default=900, ge=1, le=7 * 24 * 3600)


def _redact_secrets(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _command_digest(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _safe_mount(path: Path, target: str, *, read_only: bool) -> str:
    source = path.resolve()
    if "," in str(source) or "," in target:
        raise JobError("Docker bind mount paths may not contain commas")
    value = f"type=bind,src={source},dst={target}"
    if read_only:
        value += ",readonly"
    return value


class DockerSandbox:
    """Run one copied revision without host credentials, Git metadata, or network."""

    def __init__(
        self,
        settings: GatewaySettings,
        workspaces: ExecutionWorkspaceResolver,
    ) -> None:
        self.settings = settings
        self.workspaces = workspaces

    def container_name(self, job_id: str) -> str:
        if not _JOB_ID.fullmatch(job_id):
            raise JobError(f"invalid job id: {job_id}")
        return f"tennis-lab-mcp-{job_id}"

    def _job_directories(self, job_id: str) -> tuple[Path, Path, Path]:
        job_root = (self.settings.sandbox_jobs_dir / job_id).resolve()
        sandbox_root = self.settings.sandbox_jobs_dir.resolve()
        if not job_root.is_relative_to(sandbox_root) or job_root.parent != sandbox_root:
            raise JobError("sandbox job path escaped its configured root")
        if job_root.exists():
            raise JobError(f"sandbox directory already exists for {job_id}")
        workspace_copy = job_root / "workspace"
        artifacts = job_root / "artifacts"
        workspace_copy.mkdir(mode=0o700, parents=True)
        artifacts.mkdir(mode=0o700)
        os.chmod(job_root, 0o700)
        os.chmod(workspace_copy, 0o700)
        os.chmod(artifacts, 0o700)
        return job_root, workspace_copy, artifacts

    def command(self, spec: SandboxSpec, *, detached: bool) -> list[str]:
        source = self.workspaces.assert_execution_ready(
            workspace_id=spec.workspace_id,
            expected_sha=spec.expected_sha,
        )
        self.settings.ensure_state()
        if not self.settings.venv_root.is_dir():
            raise JobError(f"repository virtual environment is missing: {self.settings.venv_root}")
        if not self.settings.uv_python_root.is_dir():
            raise JobError(f"uv Python runtime is missing: {self.settings.uv_python_root}")
        _, workspace_copy, artifacts = self._job_directories(spec.job_id)

        wrapper = (
            "set -euo pipefail; "
            "mkdir -p /tmp/tennis-mcp-home /workspace /artifacts/repro; "
            "cp -a /source/. /workspace/; "
            "rm -rf /workspace/.git; "
            "cd /workspace; "
            f"exec /usr/bin/timeout --signal=TERM --kill-after=30s {spec.timeout_seconds} "
            '/bin/bash -lc "$1"'
        )
        arguments = [
            "docker",
            "run",
            "--name",
            self.container_name(spec.job_id),
            "--label",
            "tennis-lab.mcp=true",
            "--label",
            f"tennis-lab.revision={spec.expected_sha}",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "4096",
            "--memory",
            "48g",
            "--shm-size",
            "8g",
            "--network",
            "none",
            "--ipc",
            "private",
            "--read-only",
            "--pull",
            "never",
            "--workdir",
            "/workspace",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=8g,mode=1777",
            "--env",
            "HOME=/tmp/tennis-mcp-home",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            "GIT_CONFIG_NOSYSTEM=1",
            "--env",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "--env",
            f"VIRTUAL_ENV={self.settings.venv_root}",
            "--env",
            (
                f"PATH={self.settings.venv_root / 'bin'}:"
                "/usr/local/bin:/usr/bin:/bin"
            ),
            "--env",
            f"TENNIS_RUN_ID={spec.job_id}",
            "--env",
            "TENNIS_REPRO_DIR=/artifacts/repro",
            "--mount",
            _safe_mount(source.path, "/source", read_only=True),
            "--mount",
            _safe_mount(workspace_copy, "/workspace", read_only=False),
            "--mount",
            _safe_mount(artifacts, "/artifacts", read_only=False),
            "--mount",
            _safe_mount(
                self.settings.venv_root,
                str(self.settings.venv_root),
                read_only=True,
            ),
            "--mount",
            _safe_mount(
                self.settings.uv_python_root,
                str(self.settings.uv_python_root),
                read_only=True,
            ),
            "--mount",
            _safe_mount(self.settings.git_mask_path, "/source/.git", read_only=True),
        ]
        if detached:
            arguments.append("--detach")
        if spec.use_gpu:
            arguments.extend(["--gpus", "all"])
        arguments.extend(
            [
                self.settings.docker_image,
                "/bin/bash",
                "-lc",
                wrapper,
                "tennis-mcp",
                spec.command,
            ]
        )
        return arguments

    def start(self, spec: SandboxSpec) -> str:
        result = subprocess.run(
            self.command(spec, detached=True),
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            raise JobError(_redact_secrets(result.stderr.strip()) or "docker run failed")
        return result.stdout.strip()

    def run_foreground(self, spec: SandboxSpec) -> int:
        process = subprocess.run(
            self.command(spec, detached=False),
            check=False,
            timeout=spec.timeout_seconds + 180,
        )
        return process.returncode

    def inspect(self, job_id: str) -> dict[str, Any]:
        result = subprocess.run(
            ["docker", "inspect", self.container_name(job_id)],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise JobError("sandbox container was not found")
        document = json.loads(result.stdout)[0]
        state = document["State"]
        return {
            "status": state["Status"],
            "running": bool(state["Running"]),
            "exit_code": state["ExitCode"] if not state["Running"] else None,
            "started_at": state["StartedAt"],
            "finished_at": state["FinishedAt"],
            "error": _redact_secrets(state["Error"]) or None,
        }

    def logs(self, job_id: str, *, tail: int = 400) -> str:
        if not 1 <= tail <= 5000:
            raise JobError("tail must be between 1 and 5000")
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self.container_name(job_id)],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise JobError(_redact_secrets(result.stderr.strip()) or "docker logs failed")
        return _redact_secrets((result.stdout + result.stderr)[-200_000:])

    def stop(self, job_id: str) -> None:
        result = subprocess.run(
            ["docker", "stop", "--time", "10", self.container_name(job_id)],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise JobError(_redact_secrets(result.stderr.strip()) or "docker stop failed")


class JobManager:
    """Create and observe bounded, CPU-only direct validation jobs."""

    def __init__(
        self,
        settings: GatewaySettings,
        store: SqliteStore,
        workspaces: ExecutionWorkspaceResolver,
    ) -> None:
        self.settings = settings
        self.store = store
        self.sandbox = DockerSandbox(settings, workspaces)
        self._admission_lock = threading.Lock()

    @staticmethod
    def new_job_id(prefix: str = "job") -> str:
        return f"{prefix}-{secrets.token_hex(8)}"

    def _running_direct_jobs(self) -> int:
        running = 0
        for payload in self.store.list("jobs", limit=1000):
            job_id = str(payload["job_id"])
            with contextlib.suppress(JobError):
                if self.sandbox.inspect(job_id)["running"]:
                    running += 1
        return running

    def start(
        self,
        *,
        command: str,
        workspace_id: str,
        expected_sha: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if timeout_seconds > _DIRECT_COMMAND_MAX_SECONDS:
            raise JobError(
                f"direct commands are limited to {_DIRECT_COMMAND_MAX_SECONDS} seconds; "
                "use enqueue_training for GPU or long-running work"
            )
        job_id = self.new_job_id()
        spec = SandboxSpec(
            job_id=job_id,
            command=command,
            workspace_id=workspace_id,
            expected_sha=expected_sha.lower(),
            use_gpu=False,
            timeout_seconds=timeout_seconds,
        )
        with self._admission_lock:
            if self._running_direct_jobs() >= _MAX_CONCURRENT_DIRECT_JOBS:
                raise JobError("the maximum number of concurrent direct jobs is running")
            container_id = self.sandbox.start(spec)
        payload = {
            "job_id": job_id,
            "workspace_id": workspace_id,
            "revision": spec.expected_sha,
            "container_id": container_id,
            "command_sha256": _command_digest(command),
            "created_at": time.time(),
        }
        self.store.put(
            "jobs",
            job_id,
            payload,
            expires_at=time.time() + _JOB_METADATA_TTL_SECONDS,
        )
        return {"job_id": job_id, "status": "running", "revision": spec.expected_sha}

    def get(self, job_id: str) -> dict[str, Any]:
        payload = self.store.get("jobs", job_id)
        if payload is None:
            raise JobError("job id was not found")
        return {**payload, **self.sandbox.inspect(job_id)}

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        jobs = self.store.list("jobs", limit=limit)
        summaries: list[dict[str, Any]] = []
        for payload in jobs:
            job_id = str(payload["job_id"])
            try:
                state = self.sandbox.inspect(job_id)
            except JobError:
                state = {"status": "missing", "running": False, "exit_code": None}
            summaries.append({**payload, **state})
        return summaries


class TrainingQueueManager:
    """Queue all GPU experiments through the repository's serial training queue."""

    def __init__(
        self,
        settings: GatewaySettings,
        store: SqliteStore,
        workspaces: ExecutionWorkspaceResolver,
    ) -> None:
        self.settings = settings
        self.store = store
        self.workspaces = workspaces
        self.sandbox = DockerSandbox(settings, workspaces)
        self.queue_script = (
            settings.repo_root
            / ".agents/skills/training-queue/scripts/training_queue.sh"
        )
        self.queue_dir = settings.repo_root / ".training_queue"

    def _queue_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["TRAINING_QUEUE_DIR"] = str(self.queue_dir)
        return environment

    def enqueue(
        self,
        *,
        name: str,
        command: str,
        workspace_id: str,
        expected_sha: str,
        issue: int | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if not _TRAINING_NAME.fullmatch(name):
            raise JobError(
                "training name must start with an alphanumeric character and use "
                "only letters, digits, dot, underscore, or hyphen"
            )
        if issue is not None and issue <= 0:
            raise JobError("issue must be a positive integer")
        workspace = self.workspaces.assert_execution_ready(
            workspace_id=workspace_id,
            expected_sha=expected_sha,
        )

        job_id = JobManager.new_job_id("train")
        spec = SandboxSpec(
            job_id=job_id,
            command=command,
            workspace_id=workspace_id,
            expected_sha=expected_sha.lower(),
            use_gpu=True,
            timeout_seconds=timeout_seconds,
        )
        spec_path = self.settings.job_specs_dir / f"{job_id}.json"
        spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.chmod(spec_path, 0o600)

        bootstrap = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(_RUNTIME_ROOT)!r}); "
            "from src.automation.chatgpt_mcp.sandbox_exec import run_from_spec; "
            "raise SystemExit(run_from_spec(Path(sys.argv[1])))"
        )
        runner_arguments = [
            "env",
            f"TENNIS_MCP_REPO_ROOT={self.settings.repo_root}",
            f"TENNIS_MCP_STATE_DIR={self.settings.state_dir}",
            f"TENNIS_MCP_DOCKER_IMAGE={self.settings.docker_image}",
            f"TENNIS_MCP_UV_PYTHON_ROOT={self.settings.uv_python_root}",
            sys.executable,
            "-I",
            "-c",
            bootstrap,
            str(spec_path),
        ]
        runner = shlex.join(runner_arguments)
        arguments = [
            "bash",
            str(self.queue_script),
            "add",
            runner,
            "--name",
            name,
            "--provider",
            "chatgpt-wsl-mcp",
            "--session",
            job_id,
        ]
        if issue is not None:
            arguments.extend(["--issue", str(issue)])
        queued = subprocess.run(
            arguments,
            cwd=workspace.path,
            env=self._queue_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if queued.returncode != 0:
            spec_path.unlink(missing_ok=True)
            raise JobError(
                _redact_secrets(queued.stderr.strip())
                or "training queue admission failed"
            )
        queue_file = queued.stdout.strip().removeprefix("queued: ")
        if (
            not _QUEUE_FILE.fullmatch(queue_file)
            or Path(queue_file).name != queue_file
        ):
            spec_path.unlink(missing_ok=True)
            raise JobError(
                f"unexpected training queue response: "
                f"{_redact_secrets(queued.stdout.strip())}"
            )

        payload = {
            "job_id": job_id,
            "name": name,
            "issue": issue,
            "workspace_id": workspace_id,
            "revision": spec.expected_sha,
            "queue_file": queue_file,
            "command_sha256": _command_digest(command),
            "created_at": time.time(),
        }
        self.store.put(
            "training_jobs",
            job_id,
            payload,
            expires_at=time.time() + _TRAINING_METADATA_TTL_SECONDS,
        )
        self._start_worker()
        return {
            "job_id": job_id,
            "queue_file": queue_file,
            "status": "queued",
            "revision": spec.expected_sha,
        }

    def _start_worker(self) -> None:
        result = subprocess.run(
            ["bash", str(self.queue_script), "start", "--idle-timeout", "30"],
            cwd=self.settings.repo_root,
            env=self._queue_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        combined = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode != 0 and "already running" not in combined:
            raise JobError(
                _redact_secrets(result.stderr.strip())
                or "training queue worker failed to start"
            )

    def status(self, job_id: str) -> dict[str, Any]:
        payload = self.store.get("training_jobs", job_id)
        if payload is None:
            raise JobError("training job id was not found")
        queue_file = str(payload["queue_file"])
        locations = {
            "queued": self.queue_dir / "jobs" / queue_file,
            "running": self.queue_dir / "running" / queue_file,
            "succeeded": self.queue_dir / "done" / queue_file,
            "failed": self.queue_dir / "failed" / queue_file,
        }
        queue_status = "unknown"
        for candidate_status, path in locations.items():
            if path.exists():
                queue_status = candidate_status
                break
        result: dict[str, Any] = {**payload, "status": queue_status}
        if queue_status in {"running", "succeeded", "failed"}:
            with contextlib.suppress(JobError):
                container = self.sandbox.inspect(job_id)
                result.update(
                    {
                        "container_status": container["status"],
                        "running": container["running"],
                        "exit_code": container["exit_code"],
                        "started_at": container["started_at"],
                        "finished_at": container["finished_at"],
                        "error": container["error"],
                    }
                )
        return result

    def logs(self, job_id: str, *, tail: int = 400) -> str:
        payload = self.store.get("training_jobs", job_id)
        if payload is None:
            raise JobError("training job id was not found")
        queue_file = str(payload["queue_file"])
        log_path = self.queue_dir / "logs" / f"{queue_file.removesuffix('.job')}.log"
        if not log_path.is_file():
            return "training log is not available yet"
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return _redact_secrets("\n".join(lines[-tail:])[-200_000:])


def execute_sandbox_spec(settings: GatewaySettings, spec_path: Path) -> int:
    """Execute one owner-only spec through the exact-revision Docker sandbox."""

    resolved = spec_path.resolve()
    specs_root = settings.job_specs_dir.resolve()
    if (
        spec_path.is_symlink()
        or not resolved.is_relative_to(specs_root)
        or resolved.parent != specs_root
    ):
        raise JobError("sandbox spec must be a regular file in TENNIS_MCP_STATE_DIR")
    file_stat = resolved.stat()
    if file_stat.st_uid != os.getuid() or stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise JobError("sandbox spec ownership or permissions are unsafe")
    spec = SandboxSpec.model_validate_json(resolved.read_text(encoding="utf-8"))
    store = SqliteStore(settings.database_path)
    workspaces = WorkspaceManager(settings.repo_root, settings.state_dir, store)
    return DockerSandbox(settings, workspaces).run_foreground(spec)

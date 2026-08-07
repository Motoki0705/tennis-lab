"""Durable Docker sandbox jobs and `.training_queue` admission."""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.automation.chatgpt_mcp.settings import GatewaySettings
from src.automation.chatgpt_mcp.storage import SqliteStore
from src.automation.chatgpt_mcp.workspace import WorkspaceManager

_JOB_ID = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")


class JobError(RuntimeError):
    """Raised for invalid job state or a failed sandbox operation."""


class SandboxSpec(BaseModel):
    """Serializable command contract executed only inside a Docker sandbox."""

    job_id: str
    command: str = Field(min_length=1, max_length=100_000)
    workspace: str
    use_gpu: bool = False
    network_access: bool = False
    timeout_seconds: int = Field(default=3600, ge=1, le=7 * 24 * 3600)


class DockerSandbox:
    """Run untrusted repository code without mounting host credentials or sockets."""

    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings
        self.workspaces = WorkspaceManager(settings.repo_root)

    def container_name(self, job_id: str) -> str:
        if not _JOB_ID.fullmatch(job_id):
            raise JobError(f"invalid job id: {job_id}")
        return f"tennis-lab-mcp-{job_id}"

    def command(
        self,
        spec: SandboxSpec,
        *,
        detached: bool,
        pass_training_environment: bool = False,
    ) -> list[str]:
        workspace = self.workspaces.resolve_workspace(spec.workspace)
        uid = os.getuid()
        gid = os.getgid()
        arguments = [
            "docker",
            "run",
            "--name",
            self.container_name(spec.job_id),
            "--user",
            f"{uid}:{gid}",
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
            "bridge" if spec.network_access else "none",
            "--workdir",
            str(workspace),
            "--env",
            "HOME=/tmp/tennis-mcp-home",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            f"PATH={workspace / '.venv/bin'}:/usr/local/bin:/usr/bin:/bin",
            "--volume",
            f"{self.settings.repo_root}:{self.settings.repo_root}",
            "--volume",
            f"{self.settings.uv_python_root}:{self.settings.uv_python_root}:ro",
        ]
        if detached:
            arguments.append("--detach")
        if spec.use_gpu:
            arguments.extend(["--gpus", "all"])
        if pass_training_environment:
            arguments.extend(
                [
                    "--env",
                    "TENNIS_RUN_ID",
                    "--env",
                    "TENNIS_REPRO_DIR",
                ]
            )
        arguments.extend(
            [
                self.settings.docker_image,
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=30s",
                str(spec.timeout_seconds),
                "/bin/bash",
                "-lc",
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
            raise JobError(result.stderr.strip() or "docker run failed")
        return result.stdout.strip()

    def run_foreground(self, spec: SandboxSpec) -> int:
        process = subprocess.run(
            self.command(spec, detached=False, pass_training_environment=True),
            check=False,
            timeout=spec.timeout_seconds,
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
            "error": state["Error"] or None,
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
            raise JobError(result.stderr.strip() or "docker logs failed")
        output = result.stdout + result.stderr
        return output[-200_000:]

    def stop(self, job_id: str) -> None:
        result = subprocess.run(
            ["docker", "stop", "--time", "10", self.container_name(job_id)],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise JobError(result.stderr.strip() or "docker stop failed")


class JobManager:
    """Create and observe asynchronous Docker jobs."""

    def __init__(self, settings: GatewaySettings, store: SqliteStore) -> None:
        self.settings = settings
        self.store = store
        self.sandbox = DockerSandbox(settings)

    @staticmethod
    def new_job_id(prefix: str = "job") -> str:
        return f"{prefix}-{secrets.token_hex(8)}"

    def start(
        self,
        *,
        command: str,
        workspace: str,
        use_gpu: bool,
        network_access: bool,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        job_id = self.new_job_id()
        spec = SandboxSpec(
            job_id=job_id,
            command=command,
            workspace=str(self.sandbox.workspaces.resolve_workspace(workspace)),
            use_gpu=use_gpu,
            network_access=network_access,
            timeout_seconds=timeout_seconds,
        )
        container_id = self.sandbox.start(spec)
        payload = {
            **spec.model_dump(),
            "container_id": container_id,
            "created_at": time.time(),
        }
        self.store.put("jobs", job_id, payload)
        return {"job_id": job_id, "status": "running", "workspace": spec.workspace}

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
            summaries.append(
                {
                    "job_id": job_id,
                    "command": payload["command"],
                    "workspace": payload["workspace"],
                    **state,
                }
            )
        return summaries


class TrainingQueueManager:
    """Enqueue sandboxed training commands through the repository FIFO queue."""

    def __init__(self, settings: GatewaySettings, store: SqliteStore) -> None:
        self.settings = settings
        self.store = store
        self.sandbox = DockerSandbox(settings)
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
        workspace: str,
        issue: int | None,
        session: str,
        network_access: bool,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", name):
            raise JobError(
                "training name must use letters, digits, dot, underscore, or hyphen"
            )
        if issue is not None and issue <= 0:
            raise JobError("issue must be a positive integer")
        if not session or len(session) > 160:
            raise JobError("session must contain 1-160 characters")

        job_id = JobManager.new_job_id("train")
        spec = SandboxSpec(
            job_id=job_id,
            command=command,
            workspace=str(self.sandbox.workspaces.resolve_workspace(workspace)),
            use_gpu=True,
            network_access=network_access,
            timeout_seconds=timeout_seconds,
        )
        spec_path = self.settings.job_specs_dir / f"{job_id}.json"
        spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.chmod(spec_path, 0o600)

        runner_arguments = [
            "env",
            f"TENNIS_MCP_REPO_ROOT={self.settings.repo_root}",
            f"TENNIS_MCP_STATE_DIR={self.settings.state_dir}",
            f"TENNIS_MCP_DOCKER_IMAGE={self.settings.docker_image}",
        ]
        if self.settings.public_base_url is not None:
            runner_arguments.append(
                f"TENNIS_MCP_PUBLIC_BASE_URL={self.settings.public_base_url}"
            )
        runner_arguments.extend(
            [
                sys.executable,
                "-m",
                "src.automation.chatgpt_mcp",
                "sandbox-exec",
                "--spec",
                str(spec_path),
            ]
        )
        runner = shlex.join(runner_arguments)
        arguments = [
            "bash",
            str(self.queue_script),
            "add",
            runner,
            "--name",
            name,
            "--provider",
            "chatgpt-web",
            "--session",
            session,
        ]
        if issue is not None:
            arguments.extend(["--issue", str(issue)])
        queued = subprocess.run(
            arguments,
            cwd=spec.workspace,
            env=self._queue_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if queued.returncode != 0:
            spec_path.unlink(missing_ok=True)
            raise JobError(queued.stderr.strip() or "training queue admission failed")
        queue_file = queued.stdout.strip().removeprefix("queued: ")
        if not queue_file.endswith(".job"):
            raise JobError(
                f"unexpected training queue response: {queued.stdout.strip()}"
            )

        payload = {
            **spec.model_dump(),
            "name": name,
            "issue": issue,
            "session": session,
            "queue_file": queue_file,
            "spec_path": str(spec_path),
            "created_at": time.time(),
        }
        self.store.put("training_jobs", job_id, payload)
        self._start_worker()
        return {"job_id": job_id, "queue_file": queue_file, "status": "queued"}

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
                result.stderr.strip() or "training queue worker failed to start"
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
        status = "unknown"
        for candidate_status, path in locations.items():
            if path.exists():
                status = candidate_status
                break
        result: dict[str, Any] = {**payload, "status": status}
        if status in {"running", "succeeded", "failed"}:
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
        return "\n".join(lines[-tail:])[-200_000:]


def execute_sandbox_spec(settings: GatewaySettings, spec_path: Path) -> int:
    """Execute one trusted-on-disk spec for the training queue worker."""

    resolved = spec_path.resolve()
    if not resolved.is_relative_to(settings.job_specs_dir):
        raise JobError("sandbox spec must stay inside TENNIS_MCP_STATE_DIR")
    spec = SandboxSpec.model_validate_json(resolved.read_text(encoding="utf-8"))
    return DockerSandbox(settings).run_foreground(spec)

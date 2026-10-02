"""Launch a fixed worker process without a shell script that can change mid-run."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .common import atomic_write_json, atomic_write_text, load_task, utc_now
from .configuration import paths

MODULE = "src.tennis_scene.chat_annotation.local_agent"


def worker_command() -> str:
    config = paths()
    return shlex.join(
        [
            str(config.python_executable),
            "-m",
            MODULE,
            "--campaign",
            str(config.campaign_dir),
            "worker",
        ]
    )


def worker_environment() -> dict[str, str]:
    config = paths()
    env = os.environ.copy()
    env.pop("CODEX_THREAD_ID", None)
    env["PYTHONPATH"] = str(config.project_root)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "2"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CODEX_HOME"] = str(config.codex_home)
    return env


def codex_command(
    attempt_dir: Path, model: str, effort: str, overrides: list[str]
) -> list[str]:
    binary = shutil.which(paths().codex_binary)
    if binary is None:
        raise FileNotFoundError(f"Codex executable not found: {paths().codex_binary}")
    args = [
        binary,
        "--disable",
        "daemon_auto_start",
        "--disable",
        "multi_agent",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "-m",
        model,
    ]
    for override in overrides:
        args += ["-c", override]
    args += [
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-c",
        'approval_policy="never"',
        "-c",
        "agents.enabled=false",
        "-c",
        "skills.include_instructions=false",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        "sandbox_workspace_write.network_access=false",
        "-s",
        "workspace-write",
        "-C",
        str(attempt_dir),
        "--skip-git-repo-check",
        "--color",
        "never",
        "--json",
        "-o",
        str(attempt_dir / "last_message.md"),
        "-",
    ]
    return args


def supervise(attempt_dir: Path) -> int:
    """The dispatcher owns this supervisor PID; the Codex child inherits its process group."""
    directory = attempt_dir.resolve()
    if not directory.is_relative_to(paths().tasks):
        raise ValueError(
            "worker attempt must be inside the configured campaign/tasks directory"
        )
    task = load_task(directory)
    from .configuration import VariantConfig, json_object

    launch = json_object(directory / "launch.json")
    overrides = VariantConfig(codex_config=launch.get("codex_config", [])).codex_config
    atomic_write_text(directory / "started_at", utc_now() + "\n")
    atomic_write_text(directory / "pid", str(os.getpid()) + "\n")
    try:
        command = codex_command(directory, launch["model"], launch["effort"], overrides)
        with (
            (directory / "prompt.md").open("rb") as prompt,
            (directory / "events.jsonl").open("wb") as events,
            (directory / "stderr.log").open("wb") as errors,
        ):
            child = subprocess.Popen(
                command,
                cwd=directory,
                env=worker_environment(),
                stdin=prompt,
                stdout=events,
                stderr=errors,
            )
            return_code = child.wait()
    except OSError as error:
        atomic_write_text(
            directory / "stderr.log", f"{type(error).__name__}: {error}\n"
        )
        return_code = 127
    ended_at = utc_now()
    atomic_write_json(
        directory / "exit.json",
        {"exit_code": return_code, "ended_at": ended_at, "task_id": task["task_id"]},
    )
    atomic_write_text(directory / "exit_code", str(return_code) + "\n")
    atomic_write_text(directory / "ended_at", ended_at + "\n")
    return return_code

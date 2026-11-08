"""Explicit local Codex invocation; no user config or task-specific imports."""

from __future__ import annotations

import os
from pathlib import Path

from ..plan import ExecutionPlan
from .contracts import Launch


def codex_command(launch: Launch, plan: ExecutionPlan) -> list[str]:
    directory = Path(launch.directory)
    command = [
        launch.codex_binary,
        "--disable",
        "daemon_auto_start",
        "--disable",
        "multi_agent",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "-m",
        plan.agent_model,
    ]
    for setting in (
        f'model_reasoning_effort="{plan.agent_effort}"',
        'approval_policy="never"',
        "agents.enabled=false",
        "skills.include_instructions=false",
        "project_doc_max_bytes=0",
        "sandbox_workspace_write.network_access=false",
        "sandbox_workspace_write.exclude_slash_tmp=true",
        "sandbox_workspace_write.exclude_tmpdir_env_var=true",
    ):
        command.extend(("-c", setting))
    if launch.prepared.isolate_inputs:
        # The entire CLI is already inside the inherited Landlock boundary. A
        # second bwrap mount namespace needs host reads that this boundary denies.
        # Anchor project discovery here so parent repo config is never consulted.
        command.extend(("-c", 'project_root_markers=[".camera-input-root"]'))
    command.extend(
        (
            "-s",
            "danger-full-access"
            if launch.prepared.isolate_inputs
            else "workspace-write",
            "-C",
            str(directory / "work"),
            "--skip-git-repo-check",
            "--color",
            "never",
            "--json",
            "-o",
            str(directory / "work/last_message.md"),
        )
    )
    for relative in launch.prepared.writable_directories:
        if relative != "work":
            command.extend(("--add-dir", str(directory / relative)))
    return [*command, "-"]


def worker_environment(launch: Launch, plan: ExecutionPlan) -> dict[str, str]:
    environment = (
        {
            key: os.environ[key]
            for key in ("LANG", "TZ", "OPENAI_API_KEY")
            if key in os.environ
        }
        if launch.prepared.isolate_inputs
        else os.environ.copy()
    )
    environment.pop("CODEX_THREAD_ID", None)
    if launch.prepared.isolate_inputs:
        environment["HOME"] = str(Path(launch.directory) / "work")
        environment["PATH"] = "/usr/bin:/bin"
    else:
        environment["PYTHONPATH"] = launch.code_root
    environment["CODEX_HOME"] = launch.codex_home
    environment["CUDA_VISIBLE_DEVICES"] = ""
    temporary = Path(launch.directory) / "work" / "tmp"
    temporary.mkdir(exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    stage = next(
        stage for stage in plan.resources.stages if stage.stage == launch.request.stage
    )
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = str(stage.threads_per_process)
    return environment

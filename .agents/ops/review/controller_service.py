"""Pin the merge controller's code for the lifetime of a user service."""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path

from lib.checkout_lock import common_directory
from lib.d_capacity import CONFIG_ENV
from lib.worktree_service import OPS_SOURCE_PATHS, worktree_properties
from parent.local_jobs import LOCAL_JOB_ENV_VAR
from parent.snapshot import git

SOURCE = Path(__file__).resolve().parents[3]
FORWARDED_ENV = (
    "AGENT_AUTO_MERGE",
    "PATH",
    "HOME",
    "CODEX_HOME",
    "CODEX_SQLITE_HOME",
    "OPENAI_API_KEY",
    "GH_CONFIG_DIR",
    "GH_HOST",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "XDG_CONFIG_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "SSH_AUTH_SOCK",
    "GIT_SSH_COMMAND",
    CONFIG_ENV,
)
GIT_CHECKOUT_ENV = ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")


def _checkout(state_path: Path, launch_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", launch_id):
        raise ValueError("invalid merge controller launch ID")
    return state_path.parents[2] / "merge-gate-checkouts" / launch_id


def launch(state_path: Path, repository: Path) -> None:
    launch_id = uuid.uuid4().hex
    checkout = _checkout(state_path, launch_id)
    common = common_directory(SOURCE)
    if not repository.is_absolute() or common_directory(repository) != common:
        raise ValueError("merge controller state belongs to another Git repository")
    git_env = {k: v for k, v in os.environ.items() if k not in GIT_CHECKOUT_ENV}
    head = git("rev-parse", "--verify", "HEAD^{commit}", cwd=SOURCE, env=git_env)
    if git(
        "--no-optional-locks",
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        *OPS_SOURCE_PATHS,
        cwd=SOURCE,
        env=git_env,
    ).strip():
        raise ValueError(
            "commit the merge controller source before starting progression"
        )
    venv = (SOURCE / ".venv").resolve()
    python = venv / "bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"merge controller requires an executable Python in {venv}")
    setup = worktree_properties(
        checkout, common=common, head=head.decode().strip(), venv=venv
    )
    ops = checkout / ".agents/ops"
    log = state_path.with_name("controller.log")
    unit = f"tennis-lab-merge-gate-{launch_id}"
    forward = [f"--setenv={name}" for name in FORWARDED_ENV if name in os.environ]
    # In particular, a manager-wide AGENT_AUTO_MERGE must not enable a launch
    # whose caller left it unset. Checkout lock descriptors are process-local.
    unset = [
        *(name for name in FORWARDED_ENV if name not in os.environ),
        *GIT_CHECKOUT_ENV,
        "PYTHONHOME",
        "TENNIS_LAB_CHECKOUT_ROOT",
        "TENNIS_LAB_CHECKOUT_FD",
        "TENNIS_LAB_CHECKOUT_PARENT_FD",
        "TENNIS_LAB_CHECKOUT_PARENT_DIRECTORY",
    ]
    for name in ("TRAINING_QUEUE_DIR", LOCAL_JOB_ENV_VAR):
        if value := os.environ.get(name):
            forward.append(f"--setenv={name}={Path(value).resolve()}")
        else:
            unset.append(name)
    result = subprocess.run(
        [
            "systemd-run",
            "--user",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            "--service-type=notify",
            "--expand-environment=no",
            "--property=TimeoutStartSec=30s",
            "--property=UMask=0077",
            # Git commands still use the registered repository. Every ops
            # entry point and Python import uses the service's fixed source.
            f"--property=WorkingDirectory=-{repository}",
            f"--property=StandardOutput=append:{log}",
            f"--property=StandardError=append:{log}",
            f"--property=UnsetEnvironment={' '.join(unset)}",
            f"--setenv=PYTHONPATH={ops}{os.pathsep}{checkout}",
            *setup,
            *forward,
            "--",
            str(python),
            str(ops / "review/merge_gate.py"),
            "--state",
            str(state_path),
            "--launch-id",
            launch_id,
        ],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"user service {unit} failed to start; inspect {log}: "
            f"{result.stderr.strip()[:600]}"
        )


def require_checkout(state_path: Path, launch_id: str, repository: Path) -> None:
    if _checkout(state_path, launch_id) != SOURCE:
        raise ValueError("start the merge controller service from its pinned worktree")
    git_dir, common, branch = git(
        "rev-parse",
        "--path-format=absolute",
        "--git-dir",
        "--git-common-dir",
        "--abbrev-ref",
        "HEAD",
        cwd=SOURCE,
    ).splitlines()
    if git_dir == common or branch != b"HEAD":
        raise ValueError("merge controller checkout must be a detached worktree")
    if Path.cwd() != repository.resolve() or common != os.fsencode(
        common_directory(repository)
    ):
        raise ValueError("merge controller repository differs from its saved checkout")
    if not os.environ.get("NOTIFY_SOCKET"):
        raise ValueError("merge controller service needs a systemd notification socket")

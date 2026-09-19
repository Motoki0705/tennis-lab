"""Shared-queue orchestration for the complete real RGB dataset recipe."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.utils.paths import PROJECT_ROOT


def shared_repository_root(project: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    common = Path(result.stdout.strip())
    if not common.is_absolute() or common.name != ".git":
        raise ValueError("Expected an absolute Git common directory for this checkout")
    return common.parent


def build_commands(
    mode: str, overrides: Sequence[str], *, project: Path, shared: Path
) -> list[list[str]]:
    if mode not in {"all", "broadcast", "meiji"}:
        raise ValueError(f"Unknown dataset source: {mode}")
    if mode == "all" and overrides:
        raise ValueError("Source-specific overrides require meiji or broadcast mode")
    python = str(project / ".venv/bin/python")
    prefix = [python, "-m"]
    commands: list[list[str]] = []
    asset = f"paths.external_asset_root={shared / 'third_party'}"
    if mode in {"all", "broadcast"}:
        commands.append([*prefix, "src.tennis_scene.scripts.import_broadcast_ball"])
        commands.append(
            [
                *prefix,
                "src.tennis_scene.scripts.build_slcs_dataset",
                "--config-name",
                "build_broadcast_slcs_dataset",
                asset,
                *overrides,
            ]
        )
    if mode in {"all", "meiji"}:
        commands.append(
            [*prefix, "src.tennis_scene.scripts.build_slcs_dataset", asset, *overrides]
        )
    if mode == "all":
        commands.extend(
            [*prefix, f"src.tennis_scene.scripts.{name}"]
            for name in ("report_slcs_dataset_quality", "assemble_slcs_dataset")
        )
    return commands


def worker_alive(queue: Path) -> bool:
    try:
        raw = (queue / "worker.pid").read_text().strip()
    except FileNotFoundError:
        return False
    if not raw.isdecimal() or int(raw) <= 0:
        return False
    try:
        os.kill(int(raw), 0)
    except ProcessLookupError:
        return False
    return True


def run_build(
    mode: str,
    overrides: Sequence[str],
    *,
    execute: bool,
    project: Path = PROJECT_ROOT,
    environment: Mapping[str, str] | None = None,
) -> None:
    env = dict(os.environ if environment is None else environment)
    shared = shared_repository_root(project)
    commands = build_commands(mode, overrides, project=project, shared=shared)
    queue = shared / ".training_queue"
    env["TRAINING_QUEUE_DIR"] = str(queue)
    if execute:
        if not env.get("TENNIS_RUN_ID") or not env.get("TENNIS_REPRO_DIR"):
            raise ValueError("--execute requires the shared training queue reservation")
        env["CUDA_VISIBLE_DEVICES"] = env.get("TENNIS_RGB_GPU", "0")
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        for command in commands:
            subprocess.run(command, cwd=project, env=env, check=True)
        return
    script = project / ".agents/skills/training-queue/scripts/training_queue.sh"
    queued_command = shlex.join(
        [
            "env",
            f"TENNIS_RGB_GPU={env.get('TENNIS_RGB_GPU', '0')}",
            str(project / ".venv/bin/python"),
            "-m",
            "src.tennis_scene.scripts.build_real_rgb",
            "--execute",
            mode,
            *overrides,
        ]
    )
    subprocess.run(
        [
            "bash",
            str(script),
            "add",
            queued_command,
            "--name",
            f"slcs-real-rgb-build-{mode}",
            "--provider",
            env.get("TENNIS_QUEUE_PROVIDER", "human"),
            "--session",
            env.get("TENNIS_QUEUE_SESSION", "manual"),
            "--resource",
            "all",
        ],
        cwd=project,
        env=env,
        check=True,
    )
    if not worker_alive(queue):
        result = subprocess.run(
            ["bash", str(script), "start"], cwd=project, env=env, check=False
        )
        # A concurrent submitter may have started the singleton worker first.
        if result.returncode and not worker_alive(queue):
            result.check_returncode()

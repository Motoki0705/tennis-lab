"""Shared argparse/path/uvicorn helpers for the image detection task CLIs.

Each task script owns its ``argparse.ArgumentParser`` and validates its own
module-level ``NonHydraPathBoundary`` before any side effect.  This module only
contributes the shared argument schema, the role-root resolution, and the local
uvicorn launch from already validated values.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Literal

import uvicorn

from src.tasks.base.visualization.detection.web import (
    DetectionTask,
    create_detection_app,
)
from src.tasks.base.visualization.inference_queue import shared_repository_root
from src.utils.configuration import PathResolver, RuntimePathRoots


def add_path_arguments(
    parser: argparse.ArgumentParser,
    *,
    task: DetectionTask,
    port: int,
) -> None:
    """Declare the shared path and port options on a caller-owned parser."""
    parser.add_argument("--project-root", type=Path, default=shared_repository_root())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--checkpoints-root", type=Path)
    parser.add_argument("--port", type=int, default=port)


def resolve_paths(
    args: argparse.Namespace,
    *,
    task: DetectionTask,
) -> tuple[dict[str, Path], PathResolver]:
    """Resolve parsed options into absolute values and their role resolver."""
    project = args.project_root.expanduser().resolve()
    values = {
        "project_root": project,
        "data_root": (
            args.data_root if args.data_root is not None else project / "data"
        )
        .expanduser()
        .resolve(),
        "outputs_root": (
            args.outputs_root
            if args.outputs_root is not None
            else project / "outputs" / task
        )
        .expanduser()
        .resolve(),
        "checkpoints_root": (
            args.checkpoints_root
            if args.checkpoints_root is not None
            else project / "ckpt" / task
        )
        .expanduser()
        .resolve(),
    }
    roots = RuntimePathRoots(
        project_root=project,
        data_root=values["data_root"],
        output_root=values["outputs_root"],
        checkpoint_root=values["checkpoints_root"],
        artifact_root=project,
        cache_root=project / ".cache",
        external_asset_root=project / "third_party",
    )
    return values, PathResolver(roots)


def serve(
    task: DetectionTask,
    mode: Literal["review", "inference"],
    *,
    port: int,
    values: Mapping[str, Path],
) -> None:
    """Import the task service and serve the local UI from validated values."""
    module = import_module(f"src.tasks.{task}.visualization.inference.service")
    service = module.DetectionService(**values)
    config = {name: str(value) for name, value in values.items()}
    app = create_detection_app(service, task=task, mode=mode, service_config=config)
    print(f"{task} {mode}: http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)

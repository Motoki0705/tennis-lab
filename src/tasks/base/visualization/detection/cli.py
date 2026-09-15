"""Common local-server CLI for the two image detection tasks."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Literal

import uvicorn

from src.tasks.base.visualization.detection.web import (
    DetectionTask,
    create_detection_app,
)
from src.tasks.base.visualization.inference_queue import shared_repository_root
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)


def path_boundary(name: str) -> NonHydraPathBoundary:
    return NonHydraPathBoundary(
        name=name,
        fields=tuple(
            BoundaryPathField(
                field,
                role,
                PathDirection.INPUT,
                PathKind.DIRECTORY,
                must_exist=field in {"project_root", "data_root"},
                allow_role_root=True,
            )
            for field, role in (
                ("project_root", PathRole.PROJECT),
                ("data_root", PathRole.DATA),
                ("outputs_root", PathRole.OUTPUT),
                ("checkpoints_root", PathRole.CHECKPOINT),
            )
        ),
    )


def serve(
    task: DetectionTask,
    mode: Literal["review", "inference"],
    *,
    port: int,
    boundary: NonHydraPathBoundary,
) -> None:
    parser = argparse.ArgumentParser(description=f"Local {task} {mode} UI")
    parser.add_argument("--project-root", type=Path, default=shared_repository_root())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--checkpoints-root", type=Path)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
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
    boundary.validate(values, resolver=PathResolver(roots))
    module = import_module(f"src.tasks.{task}.visualization.inference.service")
    service = module.DetectionService(**values)
    config = {name: str(value) for name, value in values.items()}
    app = create_detection_app(service, task=task, mode=mode, service_config=config)
    print(f"{task} {mode}: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)

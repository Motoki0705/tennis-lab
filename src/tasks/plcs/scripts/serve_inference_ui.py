"""Serve the local PLCS scene inference UI.

Example:

    .venv/bin/python -m src.tasks.plcs.scripts.serve_inference_ui --port 8771

The browser app suggests PLCS checkpoints found under the checkpoint root,
narrows the scene families each checkpoint can consume, runs one scene window
through the model on ``--device``, and draws ground truth and prediction
together on a freely orbitable 3D court.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.tasks.plcs.visualization.inference.service import InferenceService
from src.tasks.plcs.visualization.inference.web import create_app
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT as REPO_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="plcs.inference_ui",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "checkpoint_root",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
    ),
)


def _absolute(value: Path) -> Path:
    return value.expanduser().resolve()


def main() -> None:
    """Parse arguments, build the service, and run the local server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="PLCS dataset root holding the scene families. Defaults to "
        "<repo>/data/plcs.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Directory scanned for *.ckpt suggestions. Defaults to "
        "<repo>/outputs/plcs.",
    )
    parser.add_argument(
        "--extra-checkpoint-root",
        type=Path,
        action="append",
        default=None,
        help="Additional directory scanned for *.ckpt suggestions (repeatable). "
        "Results are labelled with a parent/name prefix, e.g. ckpt/plcs/.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root used for path boundaries. Defaults to <repo>.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Default inference device offered by the UI (cuda/cpu).",
    )
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()

    project_root = _absolute(args.project_root or REPO_ROOT)
    data_root = _absolute(args.data_root or project_root / "data" / "plcs")
    checkpoint_root = _absolute(
        args.checkpoint_root or project_root / "outputs" / "plcs"
    )
    roots = RuntimePathRoots(
        project_root=project_root,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        artifact_root=checkpoint_root,
        output_root=checkpoint_root,
        cache_root=project_root / ".cache",
        external_asset_root=project_root,
    )
    paths = PATH_BOUNDARY.validate(
        {"data_root": data_root, "checkpoint_root": checkpoint_root},
        resolver=PathResolver(roots),
    )
    service = InferenceService(
        data_root=paths.declared("data_root").path,
        checkpoint_root=paths.declared("checkpoint_root").path,
        device=args.device,
        project_root=project_root,
        checkpoint_roots=tuple(args.extra_checkpoint_root or ()),
    )
    print(
        f"PLCS Inference UI · http://127.0.0.1:{args.port} · device={args.device}",
        flush=True,
    )
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

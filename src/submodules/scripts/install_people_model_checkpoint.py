"""Obtain or verify one official scene people model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.submodules.people_model_assets import (
    PEOPLE_MODEL_CHECKPOINTS,
    install_checkpoint,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="submodules.install_people_model_checkpoint",
    fields=(
        BoundaryPathField(
            "checkpoint_root", PathRole.CHECKPOINT, PathDirection.OUTPUT, PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(PEOPLE_MODEL_CHECKPOINTS), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Absolute checkpoint root")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--list", action="store_true", help="Report the official URL, SHA-256 and destination")
    operation.add_argument("--verify-only", action="store_true", help="Verify the existing checkpoint")
    args = parser.parse_args()
    if not args.checkpoint_root.is_absolute():
        parser.error("checkpoint-root must be an absolute path")
    root = args.checkpoint_root.resolve()
    roots = RuntimePathRoots(
        project_root=root, data_root=root, external_asset_root=root, checkpoint_root=root,
        output_root=root, artifact_root=root, cache_root=root,
    )
    resolver = PathResolver(roots)
    PATH_BOUNDARY.validate({"checkpoint_root": args.checkpoint_root}, resolver=resolver)
    checkpoint = PEOPLE_MODEL_CHECKPOINTS[args.model]
    path = resolver.resolve(PathRole.CHECKPOINT, checkpoint.relative_path)
    if args.list:
        print(f"{checkpoint.sha256}  {path}\n{checkpoint.source_url}")
        return
    install_checkpoint(checkpoint, path, verify_only=args.verify_only)
    print(path)


if __name__ == "__main__":
    main()

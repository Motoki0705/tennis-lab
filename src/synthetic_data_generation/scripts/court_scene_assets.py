"""Package or verify immutable Court scene inputs without starting a GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.synthetic_data_generation.dataset.court.assets import (
    pack_scene_assets,
    verify_scene_assets,
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
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.court_scene_assets",
    fields=(
        BoundaryPathField(
            "source_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "destination",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            required=False,
        ),
        BoundaryPathField(
            "root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--source-root", type=Path, required=True)
    pack.add_argument("--destination", type=Path, required=True)
    pack.add_argument("--scene-ids", nargs="+", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    names = ("source_root", "destination") if args.command == "pack" else ("root",)
    arguments = {name: getattr(args, name).expanduser().absolute() for name in names}
    # Preserve the asset reader/writer's final-entry symlink checks before resolution.
    if args.command == "verify" and arguments["root"].is_symlink():
        raise ValueError("Asset root must be an ordinary directory.")
    if args.command == "pack" and arguments["destination"].is_symlink():
        raise FileExistsError(
            f"Asset destination already exists: {arguments['destination']}"
        )
    data_root = arguments["source_root" if args.command == "pack" else "root"].resolve()
    output_root = (
        arguments["destination"].parent.resolve()
        if args.command == "pack"
        else PROJECT_ROOT
    )
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=output_root,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(arguments, resolver=PathResolver(roots))
    if args.command == "pack":
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
        )
        if status.strip():
            raise RuntimeError(
                "Commit the worktree before pinning an asset producer version."
            )

        def revision(ref: str) -> str:
            return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()

        result = pack_scene_assets(
            paths.declared("source_root").path,
            paths.declared("destination").path,
            scene_ids=args.scene_ids,
            repository_commit=revision("HEAD"),
            nht_commit=revision("HEAD:third_party/nht"),
        )
    else:
        result = verify_scene_assets(
            paths.declared("root").path, manifest_sha256=args.manifest_sha256
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

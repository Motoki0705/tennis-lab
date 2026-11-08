"""CPU export of a saved scene for player-view rendering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

from .bundle import PlayerViewBundleConfig, export_player_view_bundle

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.player_view_bundle",
    fields=(
        BoundaryPathField(
            "input",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.ANY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "smpl_faces",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "smpl_joint_regressor",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "smplx_model",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="SceneResult v2 NPZ, store directory, or scene.json",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New bundle directory"
    )
    parser.add_argument("--smpl-faces", required=True, type=Path)
    parser.add_argument("--smpl-joint-regressor", required=True, type=Path)
    parser.add_argument(
        "--smplx-model",
        required=True,
        type=Path,
        help="Neutral SMPL-X NPZ supplying the body skeleton",
    )
    parser.add_argument("--frames", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument(
        "--player-ids",
        type=int,
        nargs="+",
        help="IDs in desired bundle order; default: all scene IDs",
    )
    args = parser.parse_args()
    config = PlayerViewBundleConfig(
        frames=None if args.frames is None else tuple(args.frames),
        player_ids=None if args.player_ids is None else tuple(args.player_ids),
    )
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    faces = args.smpl_faces.expanduser().resolve()
    regressor = args.smpl_joint_regressor.expanduser().resolve()
    model = args.smplx_model.expanduser().resolve()
    # Each file is explicitly supplied. Their containing directories grant
    # the checkpoint read scope, as in the existing non-Hydra audit CLIs.
    asset_root = Path(
        os.path.commonpath([faces.parent, regressor.parent, model.parent])
    )
    roots = RuntimePathRoots(
        project_root=source.parent,
        data_root=source.parent,
        checkpoint_root=asset_root,
        artifact_root=source.parent,
        output_root=output,
        cache_root=output,
        external_asset_root=asset_root,
    )
    paths = PATH_BOUNDARY.validate(
        {
            "input": source,
            "output": output,
            "smpl_faces": faces,
            "smpl_joint_regressor": regressor,
            "smplx_model": model,
        },
        resolver=PathResolver(roots),
    )
    print(
        export_player_view_bundle(
            scene_path=paths.declared("input").path,
            output_dir=paths.declared("output").path,
            smpl_faces_path=paths.declared("smpl_faces").path,
            smpl_joint_regressor_path=paths.declared("smpl_joint_regressor").path,
            smplx_model_path=paths.declared("smplx_model").path,
            config=config,
        )
    )


if __name__ == "__main__":
    main()

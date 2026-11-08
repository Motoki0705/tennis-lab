"""CPU export of a saved scene for player-view rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bundle import PlayerViewBundleConfig, export_player_view_bundle


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
    print(
        export_player_view_bundle(
            scene_path=args.input,
            output_dir=args.output,
            smpl_faces_path=args.smpl_faces,
            smpl_joint_regressor_path=args.smpl_joint_regressor,
            smplx_model_path=args.smplx_model,
            config=config,
        )
    )


if __name__ == "__main__":
    main()

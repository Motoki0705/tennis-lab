"""CPU RGB review. Automatic raw/refined mode: --dataset-root PATH --run-root
PATH --output-root PATH --output tennis_scene/visualize/EXPERIMENT/RUN --clip VIDEO/CLIP
(repeat --clip for multiple clips). Add --dataset-root PATH --help for details.
The legacy explicit-frame mode uses --data-root/--output-root below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tennis_scene.dataset_pipeline.review import render_review
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
)
from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.reconstruction_review",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)
RUN_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.reconstruction_review.run",
    fields=(
        BoundaryPathField(
            "run_root",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def resolve_review_output(resolver: PathResolver, fragment: str) -> Path:
    parts = fragment.split("/")
    if (
        len(parts) != 4
        or parts[:2] != ["tennis_scene", "visualize"]
        or any(part in {"", ".", ".."} or "\\" in part for part in parts)
    ):
        raise ValueError(
            "--output must be tennis_scene/visualize/<experiment>/<run-id>"
        )
    output: Path = resolver.resolve(PathRole.OUTPUT, fragment)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("teachers", "frames"))
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument(
        "--dataset-root", type=Path, help="Teachers mode: absolute dataset root"
    )
    roots.add_argument("--data-root", type=Path, help="Frames mode: absolute data root")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--clip", action="append", required=True)
    parser.add_argument("--min-player-confidence", type=float)
    parser.add_argument("--min-ball-cameras", type=int)
    parser.add_argument("--label-weight-power", type=float)
    parser.add_argument("--dataset", help="Frames mode: DATA-relative dataset fragment")
    parser.add_argument("--video")
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="OUTPUT-relative tennis_scene/visualize/experiment/run-id",
    )
    args = parser.parse_args(argv)
    automatic = args.dataset_root is not None
    mode = "teachers" if automatic else "frames"
    if args.mode is not None and args.mode != mode:
        parser.error("--mode conflicts with the selected root argument")
    if automatic:
        if args.run_root is None:
            parser.error("teachers mode requires --run-root")
        if any(
            value is not None
            for value in (args.dataset, args.video, args.frames, args.cameras)
        ):
            parser.error("teachers mode forbids --dataset/--video/--frames/--cameras")
    else:
        if (
            any(value is None for value in (args.dataset, args.video, args.frames))
            or len(args.clip) != 1
        ):
            parser.error(
                "frames mode requires --dataset/--video/--frames and exactly one --clip"
            )
        if any(
            value is not None
            for value in (
                args.run_root,
                args.min_player_confidence,
                args.min_ball_cameras,
                args.label_weight_power,
            )
        ):
            parser.error("frames mode forbids teacher quality options and --run-root")
        if min(args.frames) < 0 or len(set(args.frames)) != len(args.frames):
            parser.error("--frames must contain unique nonnegative indices")
    data_root = args.dataset_root if automatic else args.data_root
    if (
        not data_root.is_absolute()
        or not args.output_root.is_absolute()
        or (automatic and not args.run_root.is_absolute())
    ):
        parser.error("root arguments must be absolute paths")
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=PROJECT_ROOT,
            data_root=data_root.resolve(),
            output_root=args.output_root.resolve(),
            checkpoint_root=PROJECT_ROOT,
            artifact_root=args.run_root.resolve()
            if automatic
            else args.output_root.resolve(),
            cache_root=PROJECT_ROOT,
            external_asset_root=PROJECT_ROOT,
        )
    )
    try:
        output = resolve_review_output(resolver, args.output)
        PATH_BOUNDARY.validate(
            {"data_root": data_root, "output": output}, resolver=resolver
        )
        if output.exists():
            raise FileExistsError(f"Choose a new review run: {output}")
    except ValueError as exc:
        parser.error(str(exc))
    if automatic:
        from src.tasks.slcs.data.quality import QualityConfig
        from src.tennis_scene.dataset_pipeline.teacher_review import review

        run_paths = RUN_BOUNDARY.validate(
            {"run_root": args.run_root}, resolver=resolver
        )

        review(
            resolver.roots.data_root,
            run_paths.declared("run_root").path,
            output,
            args.clip,
            QualityConfig(
                0.3
                if args.min_player_confidence is None
                else args.min_player_confidence,
                1 if args.min_ball_cameras is None else args.min_ball_cameras,
                1.0 if args.label_weight_power is None else args.label_weight_power,
                0.5,
            ),
        )
        return
    dataset = SLCSDataIndex.load(resolver.resolve(PathRole.DATA, args.dataset))
    matches = [
        record
        for record in dataset.clips
        if record.video_id == args.video and record.clip_id == args.clip[0]
    ]
    if len(matches) != 1:
        parser.error("--video/--clip must select exactly one dataset manifest record")
    image, sidecar = render_review(
        dataset.clip_dir(matches[0]),
        output,
        frames=args.frames,
        cameras=args.cameras,
    )
    print(image)
    print(sidecar)


if __name__ == "__main__":
    main()

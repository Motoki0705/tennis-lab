"""CPU RGB review. Automatic raw/refined mode: --dataset-root PATH --run-root
PATH --output-root PATH --output tennis_scene/visualize/EXPERIMENT/RUN --clip VIDEO/CLIP
(repeat --clip for multiple clips). Add --dataset-root PATH --help for details.
The legacy explicit-frame mode uses --data-root/--output-root below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tennis_scene.dataset_pipeline.review import render_review
from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots


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


def main() -> None:
    # The explicit run-root mode compares raw and published teachers and selects
    # diagnostic frames automatically; retain the existing manual review CLI.
    automatic = any(arg.split("=", 1)[0] == "--dataset-root" for arg in sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    if automatic:
        parser.add_argument("--dataset-root", type=Path, required=True)
        parser.add_argument("--run-root", type=Path, required=True)
        parser.add_argument("--clip", action="append", required=True)
        parser.add_argument("--min-player-confidence", type=float, default=0.3)
        parser.add_argument("--min-ball-cameras", type=int, default=1)
        parser.add_argument("--label-weight-power", type=float, default=1.0)
    else:
        parser.add_argument("--data-root", type=Path, required=True)
        parser.add_argument(
            "--dataset", required=True, help="DATA-relative dataset fragment"
        )
        parser.add_argument("--video", required=True)
        parser.add_argument(
            "--clip",
            required=True,
            help="Full canonical clip ID, e.g. video_000/clip_000",
        )
        parser.add_argument("--frames", type=int, nargs="+", required=True)
        parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="OUTPUT-relative tennis_scene/visualize/experiment/run-id",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=root,
            data_root=(args.dataset_root if automatic else args.data_root).resolve(),
            output_root=args.output_root.resolve(),
            checkpoint_root=(root / "ckpt").resolve(),
            artifact_root=args.output_root.resolve(),
            cache_root=(root / ".cache").resolve(),
            external_asset_root=(root / "third_party").resolve(),
        )
    )
    try:
        output = resolve_review_output(resolver, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    if automatic:
        from src.tasks.slcs.data.quality import QualityConfig
        from src.tennis_scene.dataset_pipeline.teacher_review import review

        review(
            args.dataset_root,
            args.run_root,
            output,
            args.clip,
            QualityConfig(
                args.min_player_confidence,
                args.min_ball_cameras,
                args.label_weight_power,
                0.5,
            ),
        )
        return
    dataset = SLCSDataIndex.load(resolver.resolve(PathRole.DATA, args.dataset))
    matches = [
        record
        for record in dataset.clips
        if record.video_id == args.video and record.clip_id == args.clip
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

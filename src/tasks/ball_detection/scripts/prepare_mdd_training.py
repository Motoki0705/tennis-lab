"""Freeze MDD coordinate datasets and 36 configurations, without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.ball_detection.data.play_intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.temporal_sampling import TemporalSamplingConfig
from src.tasks.ball_detection.training.coordinate_preparation import (
    prepare_coordinate_training,
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
    name="ball_detection.prepare_mdd_training",
    fields=(
        BoundaryPathField("ball_store", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("poses", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("model_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ball-store", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-steps", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--window-stride", type=int, default=16, help="Start stride in sampled frames")
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.ball_store, args.poses, args.model_config, args.output)):
        parser.error("Data, model configuration and output paths must be absolute")
    roots = RuntimePathRoots(project_root=args.model_config.parent, data_root=args.ball_store.parent,
                             output_root=args.output.parent, artifact_root=args.output.parent,
                             checkpoint_root=args.output.parent, cache_root=args.output.parent,
                             external_asset_root=args.output.parent)
    resolved = PATH_BOUNDARY.validate({key: getattr(args, key) for key in ("ball_store", "poses", "model_config", "output")},
                                      resolver=PathResolver(roots))
    paths = {key: resolved.declared(key).path for key in ("ball_store", "poses", "model_config", "output")}

    def progress(stage: str, completed: int, total: int) -> None:
        if completed == 1 or completed % 25 == 0 or completed == total:
            print(json.dumps(dict(stage=stage, completed=completed, total=total)), flush=True)

    plan = prepare_coordinate_training(
        paths["ball_store"], paths["poses"], paths["model_config"], paths["output"],
        play_config=PlayIntervalConfig(window_stride=args.window_stride),
        sampling=TemporalSamplingConfig(tuple(args.frame_steps)), progress=progress,
    )
    print(json.dumps(dict(output=str(paths["output"]), models=len(plan["models"]),
                          datasets=plan["datasets"], status=plan["status"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

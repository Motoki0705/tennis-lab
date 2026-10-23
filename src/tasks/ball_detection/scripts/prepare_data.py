"""Freeze coordinate windows or shared all-frame packs and evaluation owners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.ball_detection.data.distribution import source_payloads
from src.tasks.ball_detection.data.windows.all_frame_preparation import (
    prepare_all_frame_training,
)
from src.tasks.ball_detection.data.windows.all_frames import AllFrameWindowConfig
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.preparation import (
    prepare_coordinate_training,
)
from src.tasks.ball_detection.data.windows.temporal import TemporalSamplingConfig
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

ALL_FRAME_PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_detection.prepare_all_frame_training",
    fields=(
        BoundaryPathField("ball_store", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
        BoundaryPathField("rgb_manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
        BoundaryPathField("distribution_source_store", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True, required=False),
    ),
)


def _prepare_all_frames(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if any(value is not None for value in (args.poses, args.model_config, args.frame_steps, args.window_stride)):
        parser.error("all_frames uses native frames and shared packs; coordinate/pose options are incompatible")
    if args.distribution_source_store is not None and args.rgb_manifest is not None:
        parser.error("distribution-source-store and rgb-manifest cannot be combined")
    declarations = dict(ball_store=args.ball_store, output=args.output)
    if args.rgb_manifest is not None:
        declarations["rgb_manifest"] = args.rgb_manifest
    if args.distribution_source_store is not None:
        declarations["distribution_source_store"] = args.distribution_source_store
    if not all(path.is_absolute() for path in declarations.values()):
        parser.error("Snapshot, output and RGB manifest paths must be absolute")
    artifact_root = args.output.parent
    if args.rgb_manifest is not None:
        artifact_root = args.rgb_manifest.parent
    if args.distribution_source_store is not None:
        artifact_root = args.distribution_source_store.parent
    roots = RuntimePathRoots(
        project_root=args.ball_store.parent, data_root=args.ball_store.parent, output_root=args.output.parent,
        artifact_root=artifact_root,
        checkpoint_root=args.output.parent, cache_root=args.output.parent, external_asset_root=args.output.parent,
    )
    resolved = ALL_FRAME_PATH_BOUNDARY.validate(declarations, resolver=PathResolver(roots))
    configs = [AllFrameWindowConfig(
        window_length=length, max_window_length=args.max_window_length if args.max_window_length is not None else 128,
        pack_length=args.pack_length if args.pack_length is not None else 128,
        pack_stride=args.pack_stride if args.pack_stride is not None else 16,
        timestamp_gap_factor=args.timestamp_gap_factor if args.timestamp_gap_factor is not None else 2.5,
    ) for length in args.window_lengths or (32, 64, 128)]

    def progress(stage: str, completed: int, total: int) -> None:
        if completed == 1 or completed % 300 == 0 or completed == total:
            print(json.dumps(dict(stage=stage, completed=completed, total=total)), flush=True)

    plan = prepare_all_frame_training(
        resolved.declared("ball_store").path, resolved.declared("output").path, configs=configs,
        rgb_manifest=resolved.declared("rgb_manifest").path if args.rgb_manifest is not None else None,
        external_payloads=source_payloads(
            resolved.declared("ball_store").path, resolved.declared("distribution_source_store").path,
        ) if args.distribution_source_store is not None else None,
        progress=progress,
    )
    print(json.dumps(dict(output=str(args.output), status=plan["status"], payload_mode=plan["payload_mode"],
                          manifests={length: {mode: entry["path"] for mode, entry in modes.items()}
                                     for length, modes in plan["datasets"].items()})), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("coordinates", "all_frames"), default="coordinates")
    parser.add_argument("--ball-store", type=Path, required=True)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-steps", nargs="+", type=int)
    parser.add_argument("--window-stride", type=int, help="Coordinate start stride in sampled frames (default: 16)")
    parser.add_argument("--window-lengths", nargs="+", type=int, help="All-frame input lengths (default: 32 64 128)")
    parser.add_argument("--max-window-length", type=int, help="Common-interval minimum length (default: 128)")
    parser.add_argument("--pack-length", type=int, help="Shared pack length (default: 128)")
    parser.add_argument("--pack-stride", type=int, help="Shared pack start stride (default: 16)")
    parser.add_argument("--timestamp-gap-factor", type=float, help="Native median-PTS gap factor (default: 2.5)")
    parser.add_argument("--rgb-manifest", type=Path, help="Reuse RGB hashes from this exact snapshot's coordinate manifest")
    parser.add_argument("--distribution-source-store", type=Path,
                        help="Explicitly use the pinned original payloads of an unmaterialized distribution for local verification")
    args = parser.parse_args()
    if args.mode == "all_frames":
        _prepare_all_frames(args, parser)
        return
    if args.poses is None or args.model_config is None:
        parser.error("coordinates requires --poses and --model-config")
    if any(value is not None for value in (args.window_lengths, args.max_window_length, args.pack_length,
                                           args.pack_stride, args.timestamp_gap_factor, args.rgb_manifest,
                                           args.distribution_source_store)):
        parser.error("All-frame options require --mode all_frames")
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
        play_config=PlayIntervalConfig(window_stride=args.window_stride if args.window_stride is not None else 16),
        sampling=TemporalSamplingConfig(tuple(args.frame_steps) if args.frame_steps is not None else (1, 2, 4)), progress=progress,
    )
    print(json.dumps(dict(output=str(paths["output"]), models=len(plan["models"]),
                          datasets=plan["datasets"], status=plan["status"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

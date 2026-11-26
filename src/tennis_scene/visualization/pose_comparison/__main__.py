"""Render the saved Meiji 4cam pose comparison on CPU."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2

from src.tennis_scene.visualization.pose_comparison.inputs import (
    ClipPlan,
    ComparisonInputs,
    load_comparison,
    select_clips,
)
from src.tennis_scene.visualization.pose_comparison.rendering import (
    VIDEO_H,
    VIDEO_W,
    FrameRenderer,
)
from src.tennis_scene.visualization.pose_comparison.report import (
    clip_metrics,
    write_page,
)
from src.tennis_scene.visualization.rendering.multiview import sampled_frame_indices
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
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
from src.utils.video.reader import RandomAccessVideoReader
from src.utils.video.writer import VideoWriter

VIDEO_LIMIT = 15_000_000
BUNDLE_LIMIT = 200_000_000

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.pose_comparison",
    fields=(
        BoundaryPathField(
            "dataset",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "progress",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "candidates",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "pose_record",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
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
    ),
)


def _render_clip(
    data: ComparisonInputs,
    plan: ClipPlan,
    video: Path,
    poster: Path,
    step: int,
    crf: int,
) -> dict[str, Any]:
    frames = sampled_frame_indices(
        data.source.num_frames, start=0, stop=data.source.num_frames, step=step
    )
    fps = data.source.fps / step
    renderer = FrameRenderer(data, plan)
    with ExitStack() as resources:
        readers = [
            resources.enter_context(RandomAccessVideoReader(path))
            for path in data.source.paths
        ]
        writer = resources.enter_context(VideoWriter(video, fps=fps, crf=crf))
        for index in frames:
            bgr = renderer.render(
                index, tuple(reader.read(index) for reader in readers)
            )
            writer.write_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if index == frames[len(frames) // 3] and not cv2.imwrite(
                str(poster), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88]
            ):
                raise OSError(f"Could not write poster: {poster}")
    size = video.stat().st_size
    if size > VIDEO_LIMIT:
        raise ValueError(
            f"{video.name} is {size:,} bytes. Increase --crf or --frame-step for the 15 MB limit."
        )
    return {
        "video": video.relative_to(video.parents[1]).as_posix(),
        "poster": poster.relative_to(poster.parents[1]).as_posix(),
        "video_bytes": size,
        "video_sha256": dual_sha256(video),
        "video_frames": len(frames),
        "poster_source_frame": frames[len(frames) // 3],
        "source_fps": data.source.fps,
        "video_fps": fps,
    }


def _read_record(path: Path) -> tuple[dict[str, Any], str]:
    digest = dual_sha256(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if dual_sha256(path) != digest:
        raise ValueError(f"Input record changed during loading: {path}")
    return document, digest


def build(args: argparse.Namespace) -> None:
    if args.frame_step < 1 or not 0 <= args.crf <= 51:
        raise ValueError("Require --frame-step >= 1 and --crf in 0..51")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    dataset = args.dataset.resolve()
    if output.resolve().is_relative_to(dataset) or output.resolve().is_relative_to(
        args.candidates.resolve()
    ):
        raise ValueError("Use an output outside the input stores")
    progress, progress_digest = _read_record(args.progress)
    record, record_digest = _read_record(args.pose_record)
    plans, selection = select_clips(progress, args.seed)
    if args.clip:
        plans = [p for p in plans if p.clip_id in args.clip or p.path.name in args.clip]
        selected_keys = {key for p in plans for key in (p.clip_id, p.path.name)}
        if set(args.clip) - selected_keys:
            raise ValueError("--clip must name one of the clips in the selected cohort")
    selection["included"] = [plan.clip_id for plan in plans]

    cv2.setNumThreads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pose-comparison-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "page"
        videos = bundle / "stage1"
        videos.mkdir(parents=True)
        clips = []
        for plan in plans:
            old = StoreReader(
                dataset / plan.path / "annotations/tennis_scene/scene.json"
            )
            new = StoreReader(args.candidates / plan.clip_id / "scene.json")
            data = load_comparison(old, new, plan, record)
            seconds = record["clips"][plan.clip_id]["reconstruction_seconds"]
            media = _render_clip(
                data,
                plan,
                videos / f"{plan.path.name}.mp4",
                videos / f"{plan.path.name}.jpg",
                args.frame_step,
                args.crf,
            )
            clips.append(
                {
                    "id": plan.clip_id,
                    **media,
                    "metrics": clip_metrics(data, plan, seconds),
                    "provenance": data.provenance,
                }
            )
            print(
                f"{plan.clip_id}: {media['video_frames']} frames, {media['video_bytes']:,} bytes",
                flush=True,
            )
            del data
        write_page(
            bundle,
            {
                "schema_version": 1,
                "dataset_id": progress["dataset_id"],
                "selection": selection,
                "legacy_progress": {
                    "sha256": progress_digest,
                    "batch": dataset.parent.parent.name,
                },
                "stages": [
                    {
                        "id": "stage1",
                        "label": "段階1 · フレームごとの3D骨格と観測割当",
                        "description": "新経路のIDと色はフレーム内だけで有効です。コート内外の全人物候補を表示します。",
                        "pose_run": {
                            "name": args.pose_record.parent.name,
                            "sha256": record_digest,
                        },
                        "encoding": {
                            "codec": "h264",
                            "width": VIDEO_W,
                            "height": VIDEO_H,
                            "crf": args.crf,
                            "frame_step": args.frame_step,
                        },
                        "clips": clips,
                    }
                ],
            },
        )
        total = sum(file.stat().st_size for file in bundle.rglob("*") if file.is_file())
        if total > BUNDLE_LIMIT:
            raise ValueError(
                f"The bundle is {total:,} bytes. Increase --crf or --frame-step for the 200 MB limit."
            )
        bundle.rename(output)
    print(f"{output}: {len(clips)} clips, {total:,} bytes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="i2363 dataset root"
    )
    parser.add_argument(
        "--progress", required=True, type=Path, help="Original batch progress.json"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="Root of frame_pose_candidates stores",
    )
    parser.add_argument(
        "--pose-record",
        required=True,
        type=Path,
        help="Run record pinning the candidates and timings",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory for the offline page"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--crf", type=int, default=21)
    parser.add_argument(
        "--clip",
        action="append",
        help="Render a selected clip only. Repeat to render a subset.",
    )
    args = parser.parse_args()
    arguments = {
        field.name: getattr(args, field.name).expanduser().absolute()
        for field in PATH_BOUNDARY.fields
    }
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=arguments["output"].resolve(),
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        arguments, resolver=PathResolver(roots), independent_artifact_inputs=True
    )
    for name in ("dataset", "progress", "candidates", "pose_record"):
        setattr(args, name, paths.declared(name).path)
    args.output = arguments["output"]
    build(args)


if __name__ == "__main__":
    main()

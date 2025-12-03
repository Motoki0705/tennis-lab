#!/usr/bin/env python
"""Generate tennis dataset from video files.

This script runs the annotation pipeline to convert raw tennis match videos
into the tennis dataset format used by WASB-SBDT.

Supports:
- Single video processing
- Batch processing of multiple videos in a directory
- Resume capability with progress tracking (meta.json)
- Automatic detection of new videos

Usage:
    # Single video
    python -m src.wasb.scripts.generate_game \
        --video data/samples/test.mp4 \
        --output data/tennis/game11

    # Batch processing (multiple videos)
    python -m src.wasb.scripts.generate_game \
        --video-dir data/tennis/raw \
        --output-dir data/tennis/ \
        --resume

    # Check status
    python -m src.wasb.scripts.generate_game \
        --output-dir data/tennis/ \
        --status

    # Reset failed videos
    python -m src.wasb.scripts.generate_game \
        --output-dir data/tennis/ \
        --reset-failed

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import shutil

import cv2

# Add project root to path (src/wasb/scripts -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.wasb.inference import HRCNetWASBPredictor, WASBPredictor
from src.wasb.pipeline import (
    AnnotationPipeline,
    BatchProcessor,
    PipelineConfig,
)
from src.wasb.tennis_format import load_label_csv

PREDICTOR_REGISTRY = {
    "wasb": WASBPredictor,
    "hrcnet": HRCNetWASBPredictor,
}


def process_single_video(args: argparse.Namespace) -> int:
    """Process a single video file."""
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    # Create config
    config = PipelineConfig(
        score_threshold=args.score_threshold,
        min_clip_length=args.min_clip_length,
        min_detection_rate=args.min_detection_rate,
        max_gap=args.max_gap,
    )

    # Load predictor
    if not args.quiet:
        print(f"Loading {args.model} model from {args.checkpoint}...")

    predictor_cls = PREDICTOR_REGISTRY[args.model]
    predictor = predictor_cls.load_from_checkpoint(
        checkpoint_path,
        device="cuda",
        score_threshold=args.score_threshold,
    )

    # Run pipeline
    pipeline = AnnotationPipeline(predictor, config=config)
    result = pipeline.run(
        video_path=video_path,
        output_dir=args.output,
        max_frames=args.max_frames,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print(f"\nDone! Generated {len(result.clips)} clips in {result.output_dir}")

    return 0


def process_video_directory(args: argparse.Namespace) -> int:
    """Process all videos in a directory."""
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}", file=sys.stderr)
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)

    # Create config
    config = PipelineConfig(
        score_threshold=args.score_threshold,
        min_clip_length=args.min_clip_length,
        min_detection_rate=args.min_detection_rate,
        max_gap=args.max_gap,
    )

    # Load predictor
    if not args.quiet:
        print(f"Loading {args.model} model from {args.checkpoint}...")

    predictor_cls = PREDICTOR_REGISTRY[args.model]
    predictor = predictor_cls.load_from_checkpoint(
        checkpoint_path,
        device="cuda",
        score_threshold=args.score_threshold,
    )

    # Create pipeline and batch processor
    pipeline = AnnotationPipeline(predictor, config=config)
    batch_processor = BatchProcessor(
        pipeline=pipeline,
        output_dir=output_dir,
        start_game_id=args.start_game_id,
    )

    # Run batch processing
    result = batch_processor.process_directory(
        video_dir=video_dir,
        resume=args.resume,
        max_frames=args.max_frames,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\nBatch processing complete!")

    return 0 if result.failed == 0 else 1


def show_status(args: argparse.Namespace) -> int:
    """Show current processing status."""
    output_dir = Path(args.output_dir)
    meta_path = output_dir / "meta.json"

    if not meta_path.exists():
        print(f"No meta.json found in {output_dir}")
        return 1

    with meta_path.open("r") as f:
        meta = json.load(f)

    print(f"Meta version: {meta.get('version', 'unknown')}")
    print(f"Created: {meta.get('created_at', 'unknown')}")
    print(f"Updated: {meta.get('updated_at', 'unknown')}")
    print(f"Next game ID: {meta.get('next_game_id', 'unknown')}")
    print()

    videos = meta.get("videos", {})
    if not videos:
        print("No videos registered.")
        return 0

    # Count by status
    status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
    for status in videos.values():
        status_counts[status["status"]] = status_counts.get(status["status"], 0) + 1

    print(f"Videos: {len(videos)} total")
    print(f"  - Completed: {status_counts['completed']}")
    print(f"  - Pending: {status_counts['pending']}")
    print(f"  - In progress: {status_counts['in_progress']}")
    print(f"  - Failed: {status_counts['failed']}")
    print()

    # Show details
    print("Video details:")
    for name, status in sorted(videos.items()):
        status_str = status["status"].upper()
        game = status["output_game"]
        clips = status.get("num_clips", "-")
        print(f"  {name}: [{status_str}] -> {game} ({clips} clips)")

        if status["status"] == "failed" and status.get("error_message"):
            print(f"    Error: {status['error_message']}")

    return 0


def reset_videos(args: argparse.Namespace) -> int:
    """Reset video processing status."""
    output_dir = Path(args.output_dir)
    meta_path = output_dir / "meta.json"

    if not meta_path.exists():
        print(f"No meta.json found in {output_dir}")
        return 1

    with meta_path.open("r") as f:
        meta = json.load(f)

    videos = meta.get("videos", {})
    reset_count = 0

    for name, status in videos.items():
        should_reset = False

        if (
            args.reset_failed
            and status["status"] == "failed"
            or args.reset_all
            or args.reset_video
            and name in args.reset_video
        ):
            should_reset = True

        if should_reset:
            status["status"] = "pending"
            status["num_clips"] = None
            status["processed_at"] = None
            status["error_message"] = None
            reset_count += 1

    if reset_count > 0:
        from datetime import datetime

        meta["updated_at"] = datetime.now().isoformat()

        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        print(f"Reset {reset_count} video(s) to pending status.")
    else:
        print("No videos to reset.")

    return 0


def _iter_clip_dirs(game_dir: Path) -> list[tuple[int, Path]]:
    clips: list[tuple[int, Path]] = []
    if not game_dir.exists():
        return clips
    for child in game_dir.iterdir():
        if child.is_dir() and child.name.startswith("Clip"):
            suffix = child.name[4:]
            if suffix.isdigit():
                clips.append((int(suffix), child))
    clips.sort(key=lambda x: x[0])
    return clips


def _make_clip_preview_video(clip_dir: Path, output_path: Path, fps: int) -> None:
    label_path = clip_dir / "Label.csv"
    if not label_path.exists():
        return
    rows = load_label_csv(label_path)
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r.file_name)

    first_frame_path = clip_dir / f"frame_{rows_sorted[0].file_name}"
    frame = cv2.imread(str(first_frame_path))
    if frame is None:
        return

    height, width = frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for row in rows_sorted:
            frame_path = clip_dir / f"frame_{row.file_name}"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            if row.visibility != 0:
                x = int(round(row.x))
                y = int(round(row.y))
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), 2, cv2.LINE_AA)

            writer.write(frame)
    finally:
        writer.release()


def _resolve_games(output_dir: Path, games: list[str]) -> list[str]:
    if len(games) == 1 and games[0].lower() == "all":
        resolved: list[str] = []
        if output_dir.exists():
            for path in sorted(output_dir.iterdir()):
                if path.is_dir() and path.name.startswith("game"):
                    resolved.append(path.name)
        return resolved
    return games


def _generate_samples_for_game(output_dir: Path, game_name: str, fps: int = 15) -> None:
    game_dir = output_dir / game_name
    if not game_dir.exists():
        print(f"Game directory not found: {game_dir}")
        return

    samples_dir = output_dir / "samples" / game_name

    for index, clip_dir in _iter_clip_dirs(game_dir):
        output_path = samples_dir / f"Clip_{index}.mp4"
        _make_clip_preview_video(clip_dir, output_path, fps)


def generate_samples(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    games = _resolve_games(output_dir, args.generate_samples)
    if not games:
        print("No games to generate samples for.")
        return 1

    for game_name in games:
        print(f"Generating samples for {game_name}...")
        _generate_samples_for_game(output_dir, game_name)

    return 0


def _collect_kept_indices(samples_dir: Path) -> set[int]:
    kept: set[int] = set()
    if not samples_dir.exists():
        return kept

    for path in samples_dir.glob("Clip_*.mp4"):
        stem = path.stem
        if not stem.startswith("Clip_"):
            continue
        suffix = stem[5:]
        if suffix.isdigit():
            kept.add(int(suffix))

    return kept


def _delete_clip_dirs(game_dir: Path, indices: set[int]) -> None:
    for index in sorted(indices):
        clip_dir = game_dir / f"Clip{index}"
        if clip_dir.exists() and clip_dir.is_dir():
            shutil.rmtree(clip_dir)


def _reindex_clip_dirs(game_dir: Path) -> None:
    clips = _iter_clip_dirs(game_dir)
    if not clips:
        return

    temp_paths: list[Path] = []
    for index, path in clips:
        temp_path = path.with_name(f"__tmp_Clip{index}")
        path.rename(temp_path)
        temp_paths.append(temp_path)

    for new_index, temp_path in enumerate(temp_paths, start=1):
        final_path = game_dir / f"Clip{new_index}"
        temp_path.rename(final_path)


def _update_meta_num_clips(output_dir: Path, game_name: str) -> None:
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        return

    with meta_path.open("r") as f:
        meta = json.load(f)

    videos = meta.get("videos", {})
    game_dir = output_dir / game_name
    clip_count = len(_iter_clip_dirs(game_dir))

    updated = False
    for status in videos.values():
        if status.get("output_game") == game_name:
            status["num_clips"] = clip_count
            updated = True

    if updated:
        from datetime import datetime

        meta["updated_at"] = datetime.now().isoformat()
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)


def _apply_clip_selection_for_game(output_dir: Path, game_name: str) -> None:
    game_dir = output_dir / game_name
    samples_dir = output_dir / "samples" / game_name

    if not game_dir.exists():
        print(f"Game directory not found: {game_dir}")
        return

    if not samples_dir.exists():
        print(f"No samples found for {game_name} at {samples_dir}")
        return

    clips = _iter_clip_dirs(game_dir)
    if not clips:
        print(f"No clips found in {game_dir}")
        return

    original_indices = {index for index, _ in clips}
    kept_indices = _collect_kept_indices(samples_dir)

    to_drop = original_indices - kept_indices
    if to_drop:
        print(f"Dropping clips for {game_name}: {sorted(to_drop)}")
        _delete_clip_dirs(game_dir, to_drop)

    _reindex_clip_dirs(game_dir)
    _update_meta_num_clips(output_dir, game_name)

    # Cleanup samples directory after applying selection
    shutil.rmtree(samples_dir, ignore_errors=True)


def apply_clip_selection(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    games = _resolve_games(output_dir, args.apply_clip_selection)
    if not games:
        print("No games to apply clip selection for.")
        return 1

    for game_name in games:
        print(f"Applying clip selection for {game_name}...")
        _apply_clip_selection_for_game(output_dir, game_name)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate tennis dataset from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--video",
        type=str,
        help="Path to single video file",
    )
    mode_group.add_argument(
        "--video-dir",
        type=str,
        help="Directory containing video files for batch processing",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show processing status from meta.json",
    )
    mode_group.add_argument(
        "--reset-failed",
        action="store_true",
        help="Reset failed videos to pending status",
    )
    mode_group.add_argument(
        "--reset-all",
        action="store_true",
        help="Reset all videos to pending status",
    )
    mode_group.add_argument(
        "--reset-video",
        type=str,
        nargs="+",
        help="Reset specific video(s) by name",
    )
    mode_group.add_argument(
        "--generate-samples",
        type=str,
        nargs="+",
        help="Generate preview videos with ball overlay for specified game(s)",
    )
    mode_group.add_argument(
        "--apply-clip-selection",
        type=str,
        nargs="+",
        help="Apply clip selection based on samples for specified game(s)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for single video (e.g., data/tennis/game11)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tennis",
        help="Base output directory for batch processing (default: data/tennis)",
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="third_party/WASB-SBDT/pretrained/wasb_tennis_best.pth.tar",
        help="Path to WASB/HRCNet checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wasb",
        choices=list(PREDICTOR_REGISTRY.keys()),
        help="Model type to use (default: wasb)",
    )

    # Pipeline config
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Detection score threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-clip-length",
        type=int,
        default=30,
        help="Minimum frames per clip (default: 30)",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.5,
        help="Minimum detection rate per clip (default: 0.5)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=10,
        help="Maximum gap to bridge in segmentation (default: 10)",
    )

    # Batch processing options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from meta.json state (default for batch processing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing meta.json",
    )
    parser.add_argument(
        "--start-game-id",
        type=int,
        help="Starting game ID for new videos (auto-detect if not specified)",
    )

    # General options
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process per video (for testing)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Handle different modes
    if args.status:
        return show_status(args)

    if args.reset_failed or args.reset_all or args.reset_video:
        return reset_videos(args)

    if args.generate_samples:
        return generate_samples(args)

    if args.apply_clip_selection:
        return apply_clip_selection(args)

    if args.video:
        if not args.output:
            print(
                "Error: --output is required for single video processing",
                file=sys.stderr,
            )
            return 1
        return process_single_video(args)

    if args.video_dir:
        # Default to resume for batch processing
        args.resume = not args.no_resume
        return process_video_directory(args)

    # No mode specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

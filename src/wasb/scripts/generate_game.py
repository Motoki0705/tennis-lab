#!/usr/bin/env python
"""Generate tennis dataset from video files.

This script runs the annotation pipeline to convert raw tennis match videos
into the tennis dataset format used by WASB-SBDT.

Configuration is managed via Hydra and OmegaConf using the YAML file
``src/wasb/configs/generate_game.yaml``. Script-specific CLI flags are not
used; instead you edit the config or pass Hydra-style overrides.

Usage (basic):

    # Use defaults from configs/generate_game.yaml (mode: batch, etc.)
    python -m src.wasb.scripts.generate_game

    # Single video
    python -m src.wasb.scripts.generate_game \
        mode=single_video \
        video=data/samples/test.mp4 \
        output=data/tennis/game11

    # Batch processing (multiple videos)
    python -m src.wasb.scripts.generate_game \
        mode=batch \
        video_dir=data/tennis/raw \
        output_dir=data/tennis

    # Status / reset
    python -m src.wasb.scripts.generate_game mode=status
    python -m src.wasb.scripts.generate_game mode=reset_failed
    python -m src.wasb.scripts.generate_game mode=reset_all
    python -m src.wasb.scripts.generate_game \
        mode=reset_video reset_video=[match1.mp4,match2.mp4]

    # Clip sampling workflow
    python -m src.wasb.scripts.generate_game \
        mode=generate_samples generate_samples=[game11]
    python -m src.wasb.scripts.generate_game \
        mode=apply_clip_selection apply_clip_selection=[game11]

    # Trajectory completion (post-processing)
    python -m src.wasb.scripts.generate_game \
        mode=apply_completion apply_completion=[game11,game12] \
        pipeline.completion_method=hybrid

See ``src/wasb/configs/generate_game.yaml`` for the full list of options
and default values.

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import shutil

import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# Add project root to path (src/wasb/scripts -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.wasb.inference import HRCNetWASBPredictor, WASBPredictor
from src.wasb.pipeline import (
    AnnotationPipeline,
    BatchProcessor,
    PipelineConfig,
)
from src.wasb.models.trajectory_completer import create_completer
from src.wasb.tennis_format import load_label_csv, save_label_csv, TennisLabelRow

PREDICTOR_REGISTRY = {
    "wasb": WASBPredictor,
    "hrcnet": HRCNetWASBPredictor,
}


def _resolve_path(path_str: str | None) -> Path | None:
    """Resolve a possibly relative path string against the project root."""
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _create_pipeline_config_from_cfg(cfg: DictConfig) -> PipelineConfig:
    pipeline_cfg = getattr(cfg, "pipeline", None)
    if pipeline_cfg is None:
        # `pipeline` section is required for WASB pipeline configuration.
        raise ValueError(
            "Config is missing required `pipeline` section for WASB pipeline."
        )

    return PipelineConfig(**pipeline_cfg)  # type: ignore[arg-type]


def run_from_config(cfg: DictConfig) -> int:
    """Dispatch execution based on the configuration mode."""
    mode = str(getattr(cfg, "mode", "batch"))

    if mode == "status":
        return show_status(cfg)
    if mode in {"reset_failed", "reset_all", "reset_video"}:
        return reset_videos(cfg)
    if mode == "generate_samples":
        return generate_samples(cfg)
    if mode == "apply_clip_selection":
        return apply_clip_selection(cfg)
    if mode == "apply_completion":
        return apply_completion(cfg)
    if mode == "single_video":
        return process_single_video(cfg)
    if mode == "batch":
        return process_video_directory(cfg)

    raise ValueError(f"Unknown mode: {mode}")


def process_single_video(cfg: DictConfig) -> int:
    """Process a single video file."""
    video_str = getattr(cfg, "video", None)
    if video_str is None:
        print("Error: 'video' is not set for single_video mode.", file=sys.stderr)
        return 1

    video_path = _resolve_path(str(video_str))
    if video_path is None or not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        return 1

    checkpoint_str = getattr(
        cfg,
        "checkpoint",
        "third_party/WASB-SBDT/pretrained/wasb_tennis_best.pth.tar",
    )
    checkpoint_path = _resolve_path(str(checkpoint_str))
    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    # Create config
    config = _create_pipeline_config_from_cfg(cfg)

    # Load predictor
    quiet = bool(getattr(cfg, "quiet", False))
    model_name = str(getattr(cfg, "model", "wasb"))
    if not quiet:
        print(f"Loading {model_name} model from {checkpoint_path}...")

    predictor_cls = PREDICTOR_REGISTRY[model_name]
    predictor = predictor_cls.load_from_checkpoint(
        checkpoint_path,
        device="cuda",
        score_threshold=config.score_threshold,
    )

    output_str = getattr(cfg, "output", None)
    if output_str is None:
        print("Error: 'output' is not set for single_video mode.", file=sys.stderr)
        return 1

    output_path = _resolve_path(str(output_str))
    if output_path is None:
        print("Error: Failed to resolve 'output' path.", file=sys.stderr)
        return 1

    # Run pipeline
    pipeline = AnnotationPipeline(predictor, config=config)
    result = pipeline.run(
        video_path=video_path,
        output_dir=str(output_path),
        max_frames=getattr(cfg, "max_frames", None),
        verbose=not quiet,
    )

    if not quiet:
        print(f"\nDone! Generated {len(result.clips)} clips in {result.output_dir}")

    return 0


def process_video_directory(cfg: DictConfig) -> int:
    """Process all videos in a directory."""
    video_dir_str = getattr(cfg, "video_dir", None)
    if video_dir_str is None:
        print("Error: 'video_dir' is not set for batch mode.", file=sys.stderr)
        return 1

    video_dir = _resolve_path(str(video_dir_str))
    if video_dir is None or not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}", file=sys.stderr)
        return 1

    checkpoint_str = getattr(
        cfg,
        "checkpoint",
        "third_party/WASB-SBDT/pretrained/wasb_tennis_best.pth.tar",
    )
    checkpoint_path = _resolve_path(str(checkpoint_str))
    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1

    # Create config
    config = _create_pipeline_config_from_cfg(cfg)

    # Load predictor
    quiet = bool(getattr(cfg, "quiet", False))
    model_name = str(getattr(cfg, "model", "wasb"))
    if not quiet:
        print(f"Loading {model_name} model from {checkpoint_path}...")

    predictor_cls = PREDICTOR_REGISTRY[model_name]
    predictor = predictor_cls.load_from_checkpoint(
        checkpoint_path,
        device="cuda",
        score_threshold=config.score_threshold,
    )

    # Create pipeline and batch processor
    pipeline = AnnotationPipeline(predictor, config=config)
    batch_processor = BatchProcessor(
        pipeline=pipeline,
        output_dir=output_dir,
        start_game_id=getattr(cfg, "start_game_id", None),
    )

    # Determine resume behaviour (no_resume overrides resume)
    resume_flag = getattr(cfg, "resume", True)
    no_resume_flag = getattr(cfg, "no_resume", False)
    resume = bool(resume_flag) and not bool(no_resume_flag)

    # Run batch processing
    result = batch_processor.process_directory(
        video_dir=video_dir,
        resume=resume,
        max_frames=getattr(cfg, "max_frames", None),
        verbose=not quiet,
    )

    if not quiet:
        print("\nBatch processing complete!")

    return 0 if result.failed == 0 else 1


def show_status(cfg: DictConfig) -> int:
    """Show current processing status."""
    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
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

        # Show completion info if available
        completion = status.get("completion")
        if completion:
            method = completion.get("method", "unknown")
            applied_at = completion.get("applied_at", "unknown")
            completed = completion.get("frames_completed", 0)
            detected = completion.get("frames_detected", 0)
            print(f"    Completion: {method} applied at {applied_at[:10]}")
            print(f"      Detected: {detected}, Completed: {completed}")

    return 0


def reset_videos(cfg: DictConfig) -> int:
    """Reset video processing status."""
    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    meta_path = output_dir / "meta.json"

    if not meta_path.exists():
        print(f"No meta.json found in {output_dir}")
        return 1

    with meta_path.open("r") as f:
        meta = json.load(f)

    videos = meta.get("videos", {})
    reset_count = 0

    mode = str(getattr(cfg, "mode", ""))
    reset_video_list = list(getattr(cfg, "reset_video", []))

    for name, status in videos.items():
        should_reset = False

        if mode == "reset_failed" and status["status"] == "failed":
            should_reset = True
        elif mode == "reset_all":
            should_reset = True
        elif mode == "reset_video" and name in reset_video_list:
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

        # Prefer meta.json so we only target games that are actually
        # registered in the processing metadata.
        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)

            videos = meta.get("videos", {})
            for status in videos.values():
                game_name = status.get("output_game")
                if isinstance(game_name, str) and game_name not in resolved:
                    resolved.append(game_name)
        else:
            # Fallback: infer game directories directly from filesystem
            if output_dir.exists():
                for path in sorted(output_dir.iterdir()):
                    if path.is_dir() and path.name.startswith("game"):
                        resolved.append(path.name)

        return resolved

    return games


def _generate_samples_for_game(
    output_dir: Path, game_name: str, fps: int = 50, show_progress: bool = True
) -> None:
    game_dir = output_dir / game_name
    if not game_dir.exists():
        print(f"Game directory not found: {game_dir}")
        return

    samples_dir = output_dir / "samples" / game_name

    clips = _iter_clip_dirs(game_dir)
    if show_progress:
        iterator = tqdm(clips, desc=f"{game_name}", unit="clip")
    else:
        iterator = clips

    for index, clip_dir in iterator:
        output_path = samples_dir / f"Clip_{index}.mp4"
        _make_clip_preview_video(clip_dir, output_path, fps)


def generate_samples(cfg: DictConfig) -> int:
    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1

    games_cfg = list(getattr(cfg, "generate_samples", []))
    games = _resolve_games(output_dir, games_cfg)
    if not games:
        print("No games to generate samples for.")
        return 1

    iterable = games
    quiet = bool(getattr(cfg, "quiet", False))
    if not quiet:
        iterable = tqdm(games, desc="Generating samples", unit="game")

    for game_name in iterable:
        print(f"Generating samples for {game_name}...")
        _generate_samples_for_game(
            output_dir,
            game_name,
            show_progress=not quiet,
        )

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


def apply_clip_selection(cfg: DictConfig) -> int:
    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1

    games_cfg = list(getattr(cfg, "apply_clip_selection", []))
    games = _resolve_games(output_dir, games_cfg)
    if not games:
        print("No games to apply clip selection for.")
        return 1

    for game_name in games:
        print(f"Applying clip selection for {game_name}...")
        _apply_clip_selection_for_game(output_dir, game_name)

    return 0


# =============================================================================
# Trajectory Completion (Post-processing)
# =============================================================================


def _apply_completion_to_clip(
    clip_dir: Path,
    completer,
    score_threshold: float,
    verbose: bool = False,
) -> dict:
    """Apply trajectory completion to a single clip.

    Returns:
        Dictionary with completion statistics.
    """
    import numpy as np

    label_path = clip_dir / "Label.csv"
    if not label_path.exists():
        return {"frames": 0, "detected": 0, "completed": 0, "missing": 0}

    rows = load_label_csv(label_path)
    if not rows:
        return {"frames": 0, "detected": 0, "completed": 0, "missing": 0}

    rows_sorted = sorted(rows, key=lambda r: r.file_name)
    T = len(rows_sorted)

    # Extract trajectory data
    xy = np.zeros((T, 2), dtype=np.float32)
    visibility = np.zeros(T, dtype=bool)
    score = np.zeros(T, dtype=np.float32)

    for i, row in enumerate(rows_sorted):
        xy[i] = [row.x, row.y]
        visibility[i] = row.visibility == 1
        score[i] = row.score

    # Apply completion
    result = completer.complete(xy, visibility, score)

    # Build new label rows
    new_rows = []
    stats = {"frames": T, "detected": 0, "completed": 0, "missing": 0}

    for i, row in enumerate(rows_sorted):
        vis = int(result.visibility[i])
        x, y = result.xy[i]

        if vis == 1:
            # Original detection - keep original score
            new_score = row.score
            stats["detected"] += 1
        elif vis == 2:
            # Completed by model
            new_score = 0.0
            stats["completed"] += 1
        else:
            # Missing
            x, y = 0.0, 0.0
            new_score = 0.0
            stats["missing"] += 1

        new_rows.append(
            TennisLabelRow(
                file_name=row.file_name,
                visibility=vis,
                x=float(x),
                y=float(y),
                status=row.status,
                score=new_score,
            )
        )

    # Save updated labels
    save_label_csv(label_path, new_rows)

    return stats


def _apply_completion_to_game(
    output_dir: Path,
    game_name: str,
    completer,
    score_threshold: float,
    verbose: bool = True,
) -> dict:
    """Apply trajectory completion to all clips in a game.

    Returns:
        Dictionary with aggregated completion statistics.
    """
    game_dir = output_dir / game_name
    if not game_dir.exists():
        print(f"Game directory not found: {game_dir}")
        return {"clips": 0, "frames": 0, "detected": 0, "completed": 0, "missing": 0}

    clips = _iter_clip_dirs(game_dir)
    if not clips:
        print(f"No clips found in {game_dir}")
        return {"clips": 0, "frames": 0, "detected": 0, "completed": 0, "missing": 0}

    total_stats = {"clips": len(clips), "frames": 0, "detected": 0, "completed": 0, "missing": 0}

    iterator = clips
    if verbose:
        iterator = tqdm(clips, desc=f"{game_name}", unit="clip")

    for _, clip_dir in iterator:
        stats = _apply_completion_to_clip(
            clip_dir, completer, score_threshold, verbose=False
        )
        total_stats["frames"] += stats["frames"]
        total_stats["detected"] += stats["detected"]
        total_stats["completed"] += stats["completed"]
        total_stats["missing"] += stats["missing"]

    return total_stats


def _update_meta_completion(
    output_dir: Path,
    game_name: str,
    completion_method: str,
    stats: dict,
) -> None:
    """Update meta.json with completion information."""
    from datetime import datetime

    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        return

    with meta_path.open("r") as f:
        meta = json.load(f)

    videos = meta.get("videos", {})

    # Find the video entry for this game
    for video_name, status in videos.items():
        if status.get("output_game") == game_name:
            # Add completion info
            status["completion"] = {
                "applied_at": datetime.now().isoformat(),
                "method": completion_method,
                "frames_completed": stats.get("completed", 0),
                "frames_detected": stats.get("detected", 0),
                "frames_missing": stats.get("missing", 0),
            }

    meta["updated_at"] = datetime.now().isoformat()

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def apply_completion(cfg: DictConfig) -> int:
    """Apply trajectory completion to existing game(s)."""
    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1

    games_cfg = list(getattr(cfg, "apply_completion", []))
    games = _resolve_games(output_dir, games_cfg)

    if not games:
        print("No games to apply completion for.")
        return 1

    pipeline_cfg = getattr(cfg, "pipeline", None)
    if pipeline_cfg is None:
        raise ValueError(
            "Config is missing required `pipeline` section for apply_completion."
        )

    method = str(pipeline_cfg.get("completion_method", "hybrid"))
    checkpoint = pipeline_cfg.get("completion_checkpoint", None)
    physics_gap = int(pipeline_cfg.get("physics_gap_threshold", 5))
    max_gap = int(pipeline_cfg.get("max_completion_gap", 15))
    score_threshold = float(pipeline_cfg.get("score_threshold", 0.5))

    # Create completer
    completer = create_completer(
        method=method,
        checkpoint_path=checkpoint,
        physics_gap_threshold=physics_gap,
        max_gap=max_gap,
        score_threshold=score_threshold,
    )

    quiet = bool(getattr(cfg, "quiet", False))
    if not quiet:
        print(f"Using completion method: {method}")
        print(f"Physics gap threshold: {physics_gap}")
        print(f"Max completion gap: {max_gap}")
        print()

    total_completed = 0
    total_frames = 0

    for game_name in games:
        if not quiet:
            print(f"Applying completion to {game_name}...")

        stats = _apply_completion_to_game(
            output_dir=output_dir,
            game_name=game_name,
            completer=completer,
            score_threshold=score_threshold,
            verbose=not quiet,
        )

        total_completed += stats["completed"]
        total_frames += stats["frames"]

        # Update meta.json
        _update_meta_completion(
            output_dir=output_dir,
            game_name=game_name,
            completion_method=method,
            stats=stats,
        )

        if not quiet:
            det_pct = 100 * stats["detected"] / stats["frames"] if stats["frames"] > 0 else 0
            comp_pct = 100 * stats["completed"] / stats["frames"] if stats["frames"] > 0 else 0
            print(
                f"  {stats['clips']} clips, {stats['frames']} frames: "
                f"{stats['detected']} detected ({det_pct:.1f}%), "
                f"{stats['completed']} completed ({comp_pct:.1f}%)"
            )

    if not quiet:
        print()
        print("=" * 50)
        print("Completion Summary")
        print("=" * 50)
        print(f"Games processed: {len(games)}")
        print(f"Total frames: {total_frames}")
        print(f"Total completed: {total_completed}")
        if total_frames > 0:
            print(f"Completion rate: {100 * total_completed / total_frames:.1f}%")

    return 0


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="generate_game",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    exit_code = run_from_config(cfg)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

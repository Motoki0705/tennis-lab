#!/usr/bin/env python
"""Generate WASB tennis dataset in batch mode.

This script runs the WASB annotation pipeline to convert raw tennis match videos
into the tennis dataset format.

Configuration is managed via Hydra using `src/tasks/wasb/configs/generate_dataset.yaml`.
This entrypoint is intentionally limited to:

- `batch`: process all videos under `video_dir` into `output_dir` (with resume)
- `status`: show current `meta.json` processing state
- `reset_*`: reset processing state in `meta.json`

Clip sampling has been split into `src.tasks.wasb.scripts.generate_dataset.clip_sampling`.

Usage:
    uv run python -m src.tasks.wasb.scripts.generate_dataset
    uv run python -m src.tasks.wasb.scripts.generate_dataset mode=batch video_dir=data/tennis/raw
    uv run python -m src.tasks.wasb.scripts.generate_dataset mode=status output_dir=data/tennis
    uv run python -m src.tasks.wasb.scripts.generate_dataset mode=reset_failed output_dir=data/tennis

Hydra overrides:
    - `model=wasb|hrcnet`
    - `checkpoint=...`
    - `device=cpu|cuda`
    - `pipeline.*` (see `src/tasks/wasb/configs/pipeline/default.yaml`)

"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import hydra
from omegaconf import DictConfig

from src.tasks.wasb.pipeline import AnnotationPipeline, PipelineConfig

if TYPE_CHECKING:
    from src.tasks.wasb.pipeline import PipelineResult


def _get_predictor_cls(model_name: str):
    """Resolve predictor class by name (lazy import to keep module lightweight)."""
    from src.tasks.wasb.inference import HRCNetWASBPredictor, WASBPredictor

    if model_name == "wasb":
        return WASBPredictor
    if model_name == "hrcnet":
        return HRCNetWASBPredictor
    raise ValueError(f"Unknown model: {model_name}")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
META_FILENAME = "meta.json"
META_VERSION = "1.0"


def _resolve_path(path_str: str | None) -> Path | None:
    """Resolve a path from config (relative to current working directory)."""
    if path_str is None:
        return None
    return Path(path_str)


def _create_pipeline_config_from_cfg(cfg: DictConfig) -> PipelineConfig:
    pipeline_cfg = getattr(cfg, "pipeline", None)
    if pipeline_cfg is None:
        raise ValueError("Config is missing required `pipeline` section for WASB.")
    return PipelineConfig(**pipeline_cfg)  # type: ignore[arg-type]


def run_from_config(cfg: DictConfig) -> int:
    """Dispatch execution based on the configuration mode."""
    mode = str(getattr(cfg, "mode", "batch"))

    if mode == "batch":
        return process_video_directory(cfg)
    if mode == "status":
        return show_status(cfg)
    if mode in {"reset_failed", "reset_all", "reset_video"}:
        return reset_videos(cfg)

    raise ValueError(f"Unknown mode: {mode}")


@dataclass
class VideoStatus:
    """Status of a single video processing."""

    status: Literal["pending", "in_progress", "completed", "failed"]
    output_game: str
    num_clips: int | None = None
    processed_at: str | None = None
    file_hash: str = ""
    error_message: str | None = None


@dataclass
class BatchMeta:
    """Metadata for batch processing state."""

    version: str = META_VERSION
    created_at: str = ""
    updated_at: str = ""
    config: dict = field(default_factory=dict)
    videos: dict[str, VideoStatus] = field(default_factory=dict)
    next_game_id: int = 11  # Default start from game11

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config,
            "videos": {name: asdict(status) for name, status in self.videos.items()},
            "next_game_id": self.next_game_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BatchMeta:
        """Create from dictionary."""
        videos = {}
        for name, status_dict in data.get("videos", {}).items():
            videos[name] = VideoStatus(**status_dict)

        return cls(
            version=data.get("version", META_VERSION),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            config=data.get("config", {}),
            videos=videos,
            next_game_id=data.get("next_game_id", 11),
        )


@dataclass
class BatchResult:
    """Result of batch processing."""

    total_videos: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    new_detected: int = 0
    results: dict[str, PipelineResult | str] = field(default_factory=dict)


class BatchProcessor:
    """Batch processor for multiple video files with resume support."""

    def __init__(
        self,
        pipeline: AnnotationPipeline,
        output_dir: str | Path,
        start_game_id: int | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._meta_path = self.output_dir / META_FILENAME
        self._meta: BatchMeta | None = None
        self._start_game_id = start_game_id

    def process_directory(
        self,
        video_dir: str | Path,
        resume: bool = True,
        video_extensions: tuple[str, ...] = VIDEO_EXTENSIONS,
        max_frames: int | None = None,
        verbose: bool = True,
    ) -> BatchResult:
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        if resume and self._meta_path.exists():
            self._load_meta()
            if verbose:
                print(f"Resuming from {self._meta_path}")
        else:
            self._create_meta()
            if verbose:
                print(f"Created new meta at {self._meta_path}")

        videos = self._scan_videos(video_dir, video_extensions)
        if verbose:
            print(f"Found {len(videos)} video(s) in {video_dir}")

        new_videos = self._update_video_status(videos)
        if verbose and new_videos:
            print(f"Detected {len(new_videos)} new video(s)")

        queue = self._build_queue_from_dir(video_dir)
        if verbose:
            print(f"Processing queue: {len(queue)} video(s)")

        result = BatchResult(
            total_videos=len(videos),
            new_detected=len(new_videos),
        )

        for video_path in queue:
            video_name = video_path.name
            status = self._meta.videos[video_name]

            if verbose:
                print(f"\nProcessing: {video_name} -> {status.output_game}")

            status.status = "in_progress"
            self._save_meta()

            try:
                game_output = self.output_dir / status.output_game
                pipeline_result = self.pipeline.run(
                    video_path=video_path,
                    output_dir=game_output,
                    game_name=status.output_game,
                    max_frames=max_frames,
                    verbose=verbose,
                )

                status.status = "completed"
                status.num_clips = len(pipeline_result.clips)
                status.processed_at = datetime.now().isoformat()
                status.error_message = None
                result.processed += 1
                result.results[video_name] = pipeline_result

            except Exception as e:
                status.status = "failed"
                status.error_message = str(e)
                result.failed += 1
                result.results[video_name] = str(e)
                if verbose:
                    print(f"Error processing {video_name}: {e}")

            self._save_meta()

        for video_name, status in self._meta.videos.items():
            if status.status == "completed" and video_name not in result.results:
                result.skipped += 1

        if verbose:
            self._print_summary(result)

        return result

    def _scan_videos(
        self,
        video_dir: Path,
        extensions: tuple[str, ...],
    ) -> list[Path]:
        videos = []
        for ext in extensions:
            videos.extend(video_dir.glob(f"*{ext}"))
            videos.extend(video_dir.glob(f"*{ext.upper()}"))
        return sorted(set(videos))

    def _compute_file_hash(self, path: Path, chunk_size: int = 8192) -> str:
        sha256 = hashlib.sha256()
        bytes_read = 0
        max_bytes = 1024 * 1024  # 1MB

        with path.open("rb") as f:
            while bytes_read < max_bytes:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
                bytes_read += len(chunk)

        return f"sha256:{sha256.hexdigest()[:16]}"

    def _load_meta(self) -> None:
        with self._meta_path.open("r") as f:
            data = json.load(f)
        self._meta = BatchMeta.from_dict(data)

    def _save_meta(self) -> None:
        self._meta.updated_at = datetime.now().isoformat()
        with self._meta_path.open("w") as f:
            json.dump(self._meta.to_dict(), f, indent=2)

    def _create_meta(self) -> None:
        if self._start_game_id is not None:
            next_id = self._start_game_id
        else:
            next_id = self._detect_next_game_id()

        now = datetime.now().isoformat()
        self._meta = BatchMeta(
            version=META_VERSION,
            created_at=now,
            updated_at=now,
            config={
                "score_threshold": self.pipeline.config.score_threshold,
                "min_clip_length": self.pipeline.config.min_clip_length,
                "min_detection_rate": self.pipeline.config.min_detection_rate,
            },
            videos={},
            next_game_id=next_id,
        )
        self._save_meta()

    def _detect_next_game_id(self) -> int:
        max_id = 10
        for path in self.output_dir.iterdir():
            if path.is_dir() and path.name.startswith("game"):
                try:
                    game_id = int(path.name[4:])
                    max_id = max(max_id, game_id)
                except ValueError:
                    continue
        return max_id + 1

    def _update_video_status(self, videos: list[Path]) -> list[Path]:
        new_videos = []
        for video_path in videos:
            video_name = video_path.name
            file_hash = self._compute_file_hash(video_path)

            if video_name not in self._meta.videos:
                game_name = f"game{self._meta.next_game_id}"
                self._meta.videos[video_name] = VideoStatus(
                    status="pending",
                    output_game=game_name,
                    file_hash=file_hash,
                )
                self._meta.next_game_id += 1
                new_videos.append(video_path)
            elif self._meta.videos[video_name].file_hash != file_hash:
                status = self._meta.videos[video_name]
                status.file_hash = file_hash
                status.status = "pending"
                status.num_clips = None
                status.processed_at = None
                new_videos.append(video_path)

        self._save_meta()
        return new_videos

    def _build_queue_from_dir(self, video_dir: Path) -> list[Path]:
        queue = []
        for video_name, status in self._meta.videos.items():
            if status.status in ("pending", "in_progress"):
                video_path = video_dir / video_name
                if video_path.exists():
                    queue.append(video_path)
        return queue

    def _print_summary(self, result: BatchResult) -> None:
        print("\n" + "=" * 50)
        print("Batch Processing Summary")
        print("=" * 50)
        print(f"Total videos: {result.total_videos}")
        print(f"New detected: {result.new_detected}")
        print(f"Processed: {result.processed}")
        print(f"Failed: {result.failed}")
        print(f"Skipped (already done): {result.skipped}")

        if result.failed > 0:
            print("\nFailed videos:")
            for name, error in result.results.items():
                if isinstance(error, str):
                    print(f"  {name}: {error}")


def process_video_directory(cfg: DictConfig) -> int:
    """Process all videos in a directory (batch mode)."""
    video_dir_str = getattr(cfg, "video_dir", None)
    if video_dir_str is None:
        print("Error: 'video_dir' is not set for batch mode.", file=sys.stderr)
        return 1

    video_dir = _resolve_path(str(video_dir_str))
    if video_dir is None or not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}", file=sys.stderr)
        return 1

    checkpoint_str = getattr(cfg, "checkpoint", None)
    if checkpoint_str is None:
        print("Error: 'checkpoint' is not set for batch mode.", file=sys.stderr)
        return 1

    checkpoint_path = _resolve_path(str(checkpoint_str))
    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1

    config = _create_pipeline_config_from_cfg(cfg)

    quiet = bool(getattr(cfg, "quiet", False))
    model_name = str(getattr(cfg, "model", "wasb"))
    if not quiet:
        print(f"Loading {model_name} model from {checkpoint_path}...")

    predictor_cls = _get_predictor_cls(model_name)
    device = str(getattr(cfg, "device", "cpu"))
    predictor = predictor_cls.load_from_checkpoint(
        checkpoint_path,
        device=device,
        score_threshold=config.score_threshold,
    )

    pipeline = AnnotationPipeline(predictor, config=config)
    batch_processor = BatchProcessor(
        pipeline=pipeline,
        output_dir=output_dir,
        start_game_id=getattr(cfg, "start_game_id", None),
    )

    resume_flag = getattr(cfg, "resume", True)
    no_resume_flag = getattr(cfg, "no_resume", False)
    resume = bool(resume_flag) and not bool(no_resume_flag)

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
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1
    meta_path = output_dir / META_FILENAME

    if not meta_path.exists():
        print(f"No {META_FILENAME} found in {output_dir}")
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

    status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
    for status in videos.values():
        status_name = status.get("status", "")
        status_counts[status_name] = status_counts.get(status_name, 0) + 1

    print(f"Videos: {len(videos)} total")
    print(f"  - Completed: {status_counts['completed']}")
    print(f"  - Pending: {status_counts['pending']}")
    print(f"  - In progress: {status_counts['in_progress']}")
    print(f"  - Failed: {status_counts['failed']}")
    print()

    print("Video details:")
    for name, status in sorted(videos.items()):
        status_str = str(status.get("status", "unknown")).upper()
        game = status.get("output_game", "-")
        clips = status.get("num_clips", "-")
        print(f"  {name}: [{status_str}] -> {game} ({clips} clips)")

        if status.get("status") == "failed" and status.get("error_message"):
            print(f"    Error: {status['error_message']}")

    return 0


def reset_videos(cfg: DictConfig) -> int:
    """Reset video processing status."""
    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1
    meta_path = output_dir / META_FILENAME

    if not meta_path.exists():
        print(f"No {META_FILENAME} found in {output_dir}")
        return 1

    with meta_path.open("r") as f:
        meta = json.load(f)

    videos = meta.get("videos", {})
    reset_count = 0

    mode = str(getattr(cfg, "mode", ""))
    reset_video_list = list(getattr(cfg, "reset_video", []))

    for name, status in videos.items():
        should_reset = False

        if mode == "reset_all" or mode == "reset_failed" and status.get("status") == "failed" or mode == "reset_video" and name in reset_video_list:
            should_reset = True

        if should_reset:
            status["status"] = "pending"
            status["num_clips"] = None
            status["processed_at"] = None
            status["error_message"] = None
            reset_count += 1

    if reset_count > 0:
        meta["updated_at"] = datetime.now().isoformat()

        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        print(f"Reset {reset_count} video(s) to pending status.")
    else:
        print("No videos to reset.")

    return 0


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="generate_dataset",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    exit_code = run_from_config(cfg)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

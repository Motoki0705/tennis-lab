"""Batch processing for tennis dataset generation with resume support.

This module provides batch processing of multiple videos with:
- Progress tracking via meta.json
- Resume capability for interrupted processing
- Automatic detection of new videos
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .annotation_pipeline import AnnotationPipeline, PipelineResult


# Supported video extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# Meta file name
META_FILENAME = "meta.json"

# Current meta version
META_VERSION = "1.0"


@dataclass
class VideoStatus:
    """Status of a single video processing.

    Attributes:
        status: Processing status.
        output_game: Output game directory name (e.g., "game11").
        num_clips: Number of clips generated (None if not completed).
        processed_at: ISO timestamp of completion (None if not completed).
        file_hash: SHA256 hash of video file for change detection.
        error_message: Error message if status is "failed".

    """

    status: Literal["pending", "in_progress", "completed", "failed"]
    output_game: str
    num_clips: int | None = None
    processed_at: str | None = None
    file_hash: str = ""
    error_message: str | None = None


@dataclass
class BatchMeta:
    """Metadata for batch processing state.

    Attributes:
        version: Meta file format version.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
        config: Pipeline configuration used.
        videos: Dictionary mapping video filename to VideoStatus.
        next_game_id: Next game ID to use for new videos.

    """

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
    """Result of batch processing.

    Attributes:
        total_videos: Total number of videos found.
        processed: Number of videos successfully processed.
        failed: Number of videos that failed.
        skipped: Number of videos skipped (already completed).
        new_detected: Number of new videos detected.
        results: Dictionary mapping video name to PipelineResult or error.

    """

    total_videos: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    new_detected: int = 0
    results: dict[str, PipelineResult | str] = field(default_factory=dict)


class BatchProcessor:
    """Batch processor for multiple video files with resume support.

    This processor:
    - Scans a directory for video files
    - Tracks progress in meta.json
    - Supports resuming interrupted processing
    - Detects new videos added to the directory

    Example:
        >>> from src.wasb.inference import WASBPredictor
        >>> from src.wasb.pipeline import AnnotationPipeline, PipelineConfig
        >>>
        >>> predictor = WASBPredictor.load_from_checkpoint("checkpoint.pth.tar")
        >>> pipeline = AnnotationPipeline(predictor)
        >>> batch = BatchProcessor(pipeline, output_dir="data/tennis")
        >>> result = batch.process_directory("videos/", resume=True)

    """

    def __init__(
        self,
        pipeline: AnnotationPipeline,
        output_dir: str | Path,
        start_game_id: int | None = None,
    ) -> None:
        """Initialize batch processor.

        Args:
            pipeline: Annotation pipeline instance.
            output_dir: Base output directory for generated games.
            start_game_id: Starting game ID. If None, auto-detect from
                existing games or meta.json.

        """
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
        """Process all videos in a directory.

        Args:
            video_dir: Directory containing video files.
            resume: If True, resume from meta.json state.
            video_extensions: Tuple of video file extensions to process.
            max_frames: Maximum frames per video (for testing).
            verbose: Show progress messages.

        Returns:
            BatchResult with processing statistics.

        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        # Load or create meta
        if resume and self._meta_path.exists():
            self._load_meta()
            if verbose:
                print(f"Resuming from {self._meta_path}")
        else:
            self._create_meta()
            if verbose:
                print(f"Created new meta at {self._meta_path}")

        # Scan for videos
        videos = self._scan_videos(video_dir, video_extensions)
        if verbose:
            print(f"Found {len(videos)} video(s) in {video_dir}")

        # Detect new videos and update meta
        new_videos = self._update_video_status(videos)
        if verbose and new_videos:
            print(f"Detected {len(new_videos)} new video(s)")

        # Build processing queue
        queue = self._build_queue_from_dir(video_dir)
        if verbose:
            print(f"Processing queue: {len(queue)} video(s)")

        # Process videos
        result = BatchResult(
            total_videos=len(videos),
            new_detected=len(new_videos),
        )

        for video_path in queue:
            video_name = video_path.name
            status = self._meta.videos[video_name]

            if verbose:
                print(f"\nProcessing: {video_name} -> {status.output_game}")

            # Mark as in progress
            status.status = "in_progress"
            self._save_meta()

            try:
                # Run pipeline
                game_output = self.output_dir / status.output_game
                pipeline_result = self.pipeline.run(
                    video_path=video_path,
                    output_dir=game_output,
                    game_name=status.output_game,
                    max_frames=max_frames,
                    verbose=verbose,
                )

                # Update status
                status.status = "completed"
                status.num_clips = len(pipeline_result.clips)
                status.processed_at = datetime.now().isoformat()
                status.error_message = None
                result.processed += 1
                result.results[video_name] = pipeline_result

            except Exception as e:
                # Handle failure
                status.status = "failed"
                status.error_message = str(e)
                result.failed += 1
                result.results[video_name] = str(e)

                if verbose:
                    print(f"Error processing {video_name}: {e}")

            # Save after each video
            self._save_meta()

        # Count skipped
        for video_name, status in self._meta.videos.items():
            if status.status == "completed" and video_name not in result.results:
                result.skipped += 1

        if verbose:
            self._print_summary(result)

        return result

    def get_status(self) -> BatchMeta | None:
        """Get current batch processing status.

        Returns:
            BatchMeta if meta.json exists, None otherwise.

        """
        if self._meta is None and self._meta_path.exists():
            self._load_meta()
        return self._meta

    def reset(self, video_names: list[str] | None = None) -> None:
        """Reset processing status for videos.

        Args:
            video_names: List of video names to reset. If None, reset all.

        """
        if self._meta is None:
            self._load_meta()

        if video_names is None:
            video_names = list(self._meta.videos.keys())

        for name in video_names:
            if name in self._meta.videos:
                self._meta.videos[name].status = "pending"
                self._meta.videos[name].num_clips = None
                self._meta.videos[name].processed_at = None
                self._meta.videos[name].error_message = None

        self._save_meta()

    def _scan_videos(
        self,
        video_dir: Path,
        extensions: tuple[str, ...],
    ) -> list[Path]:
        """Scan directory for video files."""
        videos = []
        for ext in extensions:
            videos.extend(video_dir.glob(f"*{ext}"))
            videos.extend(video_dir.glob(f"*{ext.upper()}"))
        return sorted(set(videos))

    def _compute_file_hash(self, path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA256 hash of file (first 1MB for speed)."""
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
        """Load meta.json."""
        with self._meta_path.open("r") as f:
            data = json.load(f)
        self._meta = BatchMeta.from_dict(data)

    def _save_meta(self) -> None:
        """Save meta.json."""
        self._meta.updated_at = datetime.now().isoformat()
        with self._meta_path.open("w") as f:
            json.dump(self._meta.to_dict(), f, indent=2)

    def _create_meta(self) -> None:
        """Create new meta."""
        # Determine starting game ID
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
        """Detect next available game ID from existing directories."""
        max_id = 10  # Default games are game1-game10
        for path in self.output_dir.iterdir():
            if path.is_dir() and path.name.startswith("game"):
                try:
                    game_id = int(path.name[4:])
                    max_id = max(max_id, game_id)
                except ValueError:
                    pass
        return max_id + 1

    def _update_video_status(self, videos: list[Path]) -> list[Path]:
        """Update meta with new videos and detect changes.

        Returns:
            List of newly detected video paths.

        """
        new_videos = []

        for video_path in videos:
            video_name = video_path.name
            file_hash = self._compute_file_hash(video_path)

            if video_name not in self._meta.videos:
                # New video
                game_name = f"game{self._meta.next_game_id}"
                self._meta.videos[video_name] = VideoStatus(
                    status="pending",
                    output_game=game_name,
                    file_hash=file_hash,
                )
                self._meta.next_game_id += 1
                new_videos.append(video_path)

            elif self._meta.videos[video_name].file_hash != file_hash:
                # Video file changed - reset status
                status = self._meta.videos[video_name]
                status.file_hash = file_hash
                status.status = "pending"
                status.num_clips = None
                status.processed_at = None
                new_videos.append(video_path)

        self._save_meta()
        return new_videos

    def _build_queue_from_dir(self, video_dir: Path) -> list[Path]:
        """Build processing queue from video directory."""
        queue = []

        for video_name, status in self._meta.videos.items():
            if status.status in ("pending", "in_progress"):
                video_path = video_dir / video_name
                if video_path.exists():
                    queue.append(video_path)

        return queue

    def _print_summary(self, result: BatchResult) -> None:
        """Print batch processing summary."""
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

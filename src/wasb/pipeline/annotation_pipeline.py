"""End-to-end annotation pipeline for tennis dataset generation.

This module orchestrates the full workflow:
1. Extract frames from video
2. Run ball detection (WASB/HRCNet)
3. Segment into clips
4. Export to tennis dataset format

Supports two modes:
- Standard mode: Loads all frames into memory (for short videos)
- Streaming mode: Processes frames in batches (for long videos)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

from src.wasb.data.streaming_loader import StreamingVideoLoader
from src.wasb.data.video_extractor import VideoExtractor
from src.wasb.models.clip_segmenter import ClipSegment, RuleBasedClipSegmenter
from src.wasb.models.trajectory_completer import (
    CompletionResult,
    HybridCompleter,
    PhysicsInterpolator,
    TrajectoryCompleter,
    create_completer,
)
from src.wasb.tennis_format import TennisLabelRow, row_from_visibility, save_label_csv

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.wasb.inference import HRCNetWASBPredictor, WASBPredictor


@dataclass
class PipelineConfig:
    """Configuration for the annotation pipeline.

    Attributes:
        score_threshold: Detection score threshold for visibility=1.
        min_clip_length: Minimum frames per clip.
        min_detection_rate: Minimum detection rate per clip.
        max_gap: Maximum gap to bridge in segmentation.
        clip_padding: Extra frames at clip boundaries.
        batch_size: Frames per batch for detection.
        frame_format: Format string for frame filenames.
        jpeg_quality: JPEG quality for saved frames.
        streaming_batch_size: Batch size for streaming mode (default: 64).
        streaming_queue_size: Queue size for streaming loader (default: 4).
        use_streaming: Enable streaming mode for long videos (default: True).
        streaming_threshold: Frame count threshold to auto-enable streaming.

    """

    score_threshold: float = 0.5
    min_clip_length: int = 30
    min_detection_rate: float = 0.5
    max_gap: int = 10
    clip_padding: int = 5
    batch_size: int = 500
    frame_format: str = "frame_{:04d}.jpg"
    jpeg_quality: int = 95

    # Streaming mode settings
    streaming_batch_size: int = 64
    streaming_queue_size: int = 4
    use_streaming: bool = True
    streaming_threshold: int = 3000  # Auto-enable streaming above this frame count

    # Trajectory completion settings
    use_completion: bool = True
    completion_method: str = "hybrid"  # "physics", "bilstm", "hybrid"
    completion_checkpoint: str | None = None  # Path to learned model checkpoint
    physics_gap_threshold: int = 5  # Max gap for physics interpolation
    max_completion_gap: int = 15  # Max gap to attempt completion


@dataclass
class PipelineResult:
    """Result from running the annotation pipeline.

    Attributes:
        game_name: Name of the generated game (e.g., "game11").
        output_dir: Path to the output game directory.
        clips: List of generated clip information.
        total_frames: Total frames processed.
        total_detections: Total frames with ball detection.
        total_completed: Total frames completed by trajectory completer.
        total_outliers_removed: Total outliers detected and removed.

    """

    game_name: str
    output_dir: Path
    clips: list[ClipInfo] = field(default_factory=list)
    total_frames: int = 0
    total_detections: int = 0
    total_completed: int = 0
    total_outliers_removed: int = 0


@dataclass
class ClipInfo:
    """Information about a generated clip.

    Attributes:
        clip_name: Name of the clip (e.g., "Clip1").
        clip_dir: Path to the clip directory.
        num_frames: Number of frames in clip.
        num_detections: Number of frames with detection (visibility=1).
        num_completed: Number of frames completed (visibility=2).
        detection_rate: Fraction of frames with detection.
        completion_rate: Fraction of frames completed.
        avg_score: Average detection score.

    """

    clip_name: str
    clip_dir: Path
    num_frames: int
    num_detections: int
    num_completed: int = 0
    detection_rate: float = 0.0
    completion_rate: float = 0.0
    avg_score: float = 0.0


class AnnotationPipeline:
    """End-to-end pipeline for generating tennis dataset from video.

    This pipeline:
    1. Loads video and extracts frames
    2. Runs WASB/HRCNet ball detection
    3. Segments video into rally clips
    4. Exports frames and labels in tennis format

    Example:
        >>> from src.wasb.inference import WASBPredictor
        >>> predictor = WASBPredictor.load_from_checkpoint("checkpoint.pth.tar")
        >>> pipeline = AnnotationPipeline(predictor)
        >>> result = pipeline.run(
        ...     video_path="match.mp4",
        ...     output_dir="data/tennis/game11",
        ... )
        >>> print(f"Generated {len(result.clips)} clips")

    """

    def __init__(
        self,
        predictor: WASBPredictor | HRCNetWASBPredictor,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize the annotation pipeline.

        Args:
            predictor: Ball detection predictor (WASB or HRCNet).
            config: Pipeline configuration. Uses defaults if None.

        """
        self.predictor = predictor
        self.config = config or PipelineConfig()

        # Initialize segmenter with config
        self.segmenter = RuleBasedClipSegmenter(
            min_clip_length=self.config.min_clip_length,
            min_detection_rate=self.config.min_detection_rate,
            max_gap=self.config.max_gap,
            score_threshold=self.config.score_threshold,
            padding_frames=self.config.clip_padding,
        )

        # Initialize trajectory completer if enabled
        self.completer: TrajectoryCompleter | None = None
        if self.config.use_completion:
            self.completer = create_completer(
                method=self.config.completion_method,  # type: ignore[arg-type]
                checkpoint_path=self.config.completion_checkpoint,
                physics_gap_threshold=self.config.physics_gap_threshold,
                max_gap=self.config.max_completion_gap,
                score_threshold=self.config.score_threshold,
            )

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        game_name: str | None = None,
        max_frames: int | None = None,
        verbose: bool = True,
    ) -> PipelineResult:
        """Run the full annotation pipeline on a video.

        Automatically selects streaming or standard mode based on video length
        and configuration.

        Args:
            video_path: Path to input video file.
            output_dir: Base output directory (e.g., "data/tennis/game11").
            game_name: Name for the game. Derived from output_dir if None.
            max_frames: Maximum frames to process (for testing).
            verbose: Show progress bars and status messages.

        Returns:
            PipelineResult with information about generated clips.

        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if game_name is None:
            game_name = output_dir.name

        if verbose:
            print(f"Starting annotation pipeline for {video_path}")
            print(f"Output directory: {output_dir}")

        # Check video metadata to decide mode
        extractor = VideoExtractor(video_path)
        frame_count = extractor.frame_count
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)

        if verbose:
            print(
                f"Video: {extractor.width}x{extractor.height}, "
                f"{frame_count} frames, {extractor.fps:.1f} fps"
            )

        # Decide whether to use streaming mode
        use_streaming = (
            self.config.use_streaming and frame_count > self.config.streaming_threshold
        )

        if use_streaming:
            if verbose:
                print(
                    f"Using streaming mode (>{self.config.streaming_threshold} frames)"
                )
            return self._run_streaming(
                video_path=video_path,
                output_dir=output_dir,
                game_name=game_name,
                max_frames=max_frames,
                verbose=verbose,
            )
        else:
            return self._run_standard(
                video_path=video_path,
                output_dir=output_dir,
                game_name=game_name,
                max_frames=max_frames,
                verbose=verbose,
            )

    def _run_standard(
        self,
        video_path: Path,
        output_dir: Path,
        game_name: str,
        max_frames: int | None = None,
        verbose: bool = True,
    ) -> PipelineResult:
        """Run pipeline in standard mode (all frames in memory)."""
        extractor = VideoExtractor(video_path)

        # Step 1: Run detection on all frames
        if verbose:
            print("Running ball detection (standard mode)...")

        frames, detection_results = self._run_detection(
            extractor, max_frames=max_frames, verbose=verbose
        )

        # Step 2: Segment into clips
        if verbose:
            print("Segmenting into clips...")

        segments = self._segment_video(detection_results)
        if verbose:
            print(f"Found {len(segments)} rally segments")

        if len(segments) == 0:
            if verbose:
                print("Warning: No valid clips found. Check detection quality.")
            return PipelineResult(
                game_name=game_name,
                output_dir=output_dir,
                total_frames=len(frames),
                total_detections=int(detection_results["visibility"].sum()),
            )

        # Step 3: Export clips
        if verbose:
            print("Exporting clips...")

        clips = self._export_clips(
            frames=frames,
            detection_results=detection_results,
            segments=segments,
            output_dir=output_dir,
            extractor=extractor,
            verbose=verbose,
        )

        # Aggregate statistics from clips
        total_detections = sum(c.num_detections for c in clips)
        total_completed = sum(c.num_completed for c in clips)

        result = PipelineResult(
            game_name=game_name,
            output_dir=output_dir,
            clips=clips,
            total_frames=len(frames),
            total_detections=total_detections,
            total_completed=total_completed,
        )

        if verbose:
            self._print_summary(result)

        return result

    def _run_streaming(
        self,
        video_path: Path,
        output_dir: Path,
        game_name: str,
        max_frames: int | None = None,
        verbose: bool = True,
    ) -> PipelineResult:
        """Run pipeline in streaming mode (memory-efficient for long videos).

        This implementation performs detection in a streaming fashion (Pass 1)
        and then re-opens the video to export clips for detected segments
        without keeping all frames in memory (Pass 2).
        """
        # Initialize streaming loader
        loader = StreamingVideoLoader(
            video_path=video_path,
            batch_size=self.config.streaming_batch_size,
            queue_size=self.config.streaming_queue_size,
            max_frames=max_frames,
        )

        total_frames = loader.metadata.total_frames
        height, width = loader.metadata.height, loader.metadata.width

        # Prepare result accumulators (per-frame detection results)
        all_xy: list[tuple[float, float]] = []
        all_visibility: list[bool] = []
        all_scores: list[float] = []
        all_frame_indices: list[int] = []

        # Reset predictor for streaming
        self.predictor.reset_tracker()

        if verbose:
            print("Running ball detection (streaming mode)...")
            pbar = tqdm(total=total_frames, desc="Processing frames")

        try:
            for batch in loader:
                # Run detection on batch
                results = self.predictor.predict_batch(
                    frames=batch.frames,
                    frame_indices=batch.frame_indices,
                )

                # Accumulate results for each frame in batch
                for i, idx in enumerate(batch.frame_indices):
                    all_frame_indices.append(idx)
                    all_xy.append(tuple(results["ball_xy_px"][i]))
                    all_visibility.append(bool(results["visibility"][i]))
                    all_scores.append(float(results["score"][i]))

                if verbose:
                    pbar.update(len(batch.frames))

        finally:
            if verbose:
                pbar.close()
            loader.stop()

        # Convert to arrays
        detection_results = {
            "ball_xy_px": np.array(all_xy, dtype=np.float32),
            "visibility": np.array(all_visibility, dtype=bool),
            "score": np.array(all_scores, dtype=np.float32),
            "frame_indices": np.array(all_frame_indices, dtype=np.int64),
            "ball_uv": np.array(all_xy, dtype=np.float32),
        }
        # Normalize UV coordinates
        if len(detection_results["ball_uv"]) > 0:
            detection_results["ball_uv"][:, 0] /= width
            detection_results["ball_uv"][:, 1] /= height

        # Segment into clips
        if verbose:
            print("Segmenting into clips...")

        segments = self._segment_video(detection_results)
        if verbose:
            print(f"Found {len(segments)} rally segments")

        total_frames_detected = int(detection_results["visibility"].shape[0])

        if len(segments) == 0:
            if verbose:
                print("Warning: No valid clips found. Check detection quality.")
            return PipelineResult(
                game_name=game_name,
                output_dir=output_dir,
                total_frames=total_frames_detected,
                total_detections=int(detection_results["visibility"].sum()),
            )

        # Export clips (Pass 2: re-open video and extract only needed segments)
        if verbose:
            print("Exporting clips...")

        extractor = VideoExtractor(video_path)

        clips = self._export_clips_streaming(
            detection_results=detection_results,
            segments=segments,
            output_dir=output_dir,
            extractor=extractor,
            verbose=verbose,
        )

        # Aggregate statistics from clips
        total_detections = sum(c.num_detections for c in clips)
        total_completed = sum(c.num_completed for c in clips)

        result = PipelineResult(
            game_name=game_name,
            output_dir=output_dir,
            clips=clips,
            total_frames=total_frames_detected,
            total_detections=total_detections,
            total_completed=total_completed,
        )

        if verbose:
            self._print_summary(result)

        return result

    def _run_detection(
        self,
        extractor: VideoExtractor,
        max_frames: int | None = None,
        verbose: bool = True,
    ) -> tuple[NDArray[np.uint8], dict[str, NDArray]]:
        """Run ball detection on video frames.

        Returns:
            Tuple of (frames, detection_results).

        """
        # Load all frames (for simplicity; could optimize with batching)
        frames = extractor.load_all_frames(max_frames=max_frames)

        # Run detection
        results = self.predictor.predict(frames)

        return frames, results

    def _segment_video(
        self,
        detection_results: dict[str, NDArray],
    ) -> list[ClipSegment]:
        """Segment video into clips based on detection results."""
        return self.segmenter.predict_segments(
            xy=detection_results["ball_xy_px"],
            score=detection_results["score"],
            visibility=detection_results["visibility"],
        )

    def _export_clips(
        self,
        frames: NDArray[np.uint8],
        detection_results: dict[str, NDArray],
        segments: list[ClipSegment],
        output_dir: Path,
        extractor: VideoExtractor,
        verbose: bool = True,
    ) -> list[ClipInfo]:
        """Export clips to tennis dataset format with trajectory completion."""
        import cv2

        clips = []
        iterator = enumerate(segments, 1)
        if verbose:
            iterator = tqdm(list(iterator), desc="Exporting clips")

        for clip_idx, segment in iterator:
            clip_name = f"Clip{clip_idx}"
            clip_dir = output_dir / clip_name
            clip_dir.mkdir(parents=True, exist_ok=True)

            # Extract segment frames and labels
            segment_frames = frames[segment.start : segment.end]
            segment_xy = detection_results["ball_xy_px"][segment.start : segment.end].copy()
            segment_vis = detection_results["visibility"][segment.start : segment.end].copy()
            segment_score = detection_results["score"][segment.start : segment.end].copy()

            # Apply trajectory completion if enabled
            num_completed = 0
            if self.completer is not None:
                completion_result = self.completer.complete(
                    xy=segment_xy.astype(np.float32),
                    visibility=segment_vis,
                    score=segment_score.astype(np.float32),
                )
                segment_xy = completion_result.xy
                completed_vis = completion_result.visibility
                num_completed = int(np.sum(completed_vis == 2))
            else:
                # No completion - mark valid detections as 1, rest as 0
                completed_vis = np.where(
                    segment_vis & (segment_score >= self.config.score_threshold), 1, 0
                ).astype(np.int32)

            # Save frames and build labels
            label_rows = []
            num_detections = 0

            for local_idx in range(len(segment_frames)):
                # Save frame
                filename = self.config.frame_format.format(local_idx)
                frame_path = clip_dir / filename
                frame_bgr = cv2.cvtColor(segment_frames[local_idx], cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(frame_path),
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                )

                # Get visibility and coordinates from completion result
                vis = int(completed_vis[local_idx])
                x, y = segment_xy[local_idx]
                score = float(segment_score[local_idx]) if vis == 1 else 0.0

                if vis == 1:
                    num_detections += 1
                elif vis == 0:
                    x, y = 0.0, 0.0

                # Use just the filename without frame_ prefix for Label.csv
                label_filename = f"{local_idx:04d}.jpg"
                label_rows.append(
                    row_from_visibility(
                        file_name=label_filename,
                        x=float(x),
                        y=float(y),
                        visibility=vis,
                        score=score,
                    )
                )

            # Save Label.csv
            label_path = clip_dir / "Label.csv"
            save_label_csv(label_path, label_rows)

            # Create clip info
            detection_rate = (
                num_detections / len(segment_frames) if len(segment_frames) > 0 else 0.0
            )
            completion_rate = (
                num_completed / len(segment_frames) if len(segment_frames) > 0 else 0.0
            )
            detected_scores = segment_score[
                segment_vis & (segment_score >= self.config.score_threshold)
            ]
            avg_score = (
                float(np.mean(detected_scores)) if len(detected_scores) > 0 else 0.0
            )

            clips.append(
                ClipInfo(
                    clip_name=clip_name,
                    clip_dir=clip_dir,
                    num_frames=len(segment_frames),
                    num_detections=num_detections,
                    num_completed=num_completed,
                    detection_rate=detection_rate,
                    completion_rate=completion_rate,
                    avg_score=avg_score,
                )
            )

        return clips

    def _export_clips_streaming(
        self,
        detection_results: dict[str, NDArray],
        segments: list[ClipSegment],
        output_dir: Path,
        extractor: VideoExtractor,
        verbose: bool = True,
    ) -> list[ClipInfo]:
        """Export clips in streaming mode with trajectory completion.

        This function re-opens the source video and uses VideoExtractor to
        extract frames for each segment directly to disk, while using the
        provided detection results to build Label.csv.
        """
        clips: list[ClipInfo] = []
        iterator = enumerate(segments, 1)
        if verbose:
            iterator = tqdm(list(iterator), desc="Exporting clips")

        for clip_idx, segment in iterator:
            clip_name = f"Clip{clip_idx}"
            clip_dir = output_dir / clip_name
            clip_dir.mkdir(parents=True, exist_ok=True)

            # Extract segment frames to disk
            saved_files = extractor.extract_segment(
                start_frame=segment.start,
                end_frame=segment.end,
                output_dir=clip_dir,
                frame_format=self.config.frame_format,
                jpeg_quality=self.config.jpeg_quality,
            )

            segment_length = len(saved_files)
            if segment_length == 0:
                continue

            start = segment.start
            end = start + segment_length

            segment_xy = detection_results["ball_xy_px"][start:end].copy()
            segment_vis = detection_results["visibility"][start:end].copy()
            segment_score = detection_results["score"][start:end].copy()

            # Apply trajectory completion if enabled
            num_completed = 0
            if self.completer is not None:
                completion_result = self.completer.complete(
                    xy=segment_xy.astype(np.float32),
                    visibility=segment_vis,
                    score=segment_score.astype(np.float32),
                )
                segment_xy = completion_result.xy
                completed_vis = completion_result.visibility
                num_completed = int(np.sum(completed_vis == 2))
            else:
                # No completion - mark valid detections as 1, rest as 0
                completed_vis = np.where(
                    segment_vis & (segment_score >= self.config.score_threshold), 1, 0
                ).astype(np.int32)

            # Save labels
            label_rows: list[TennisLabelRow] = []
            num_detections = 0

            for local_idx in range(segment_length):
                vis = int(completed_vis[local_idx])
                x, y = segment_xy[local_idx]
                score = float(segment_score[local_idx]) if vis == 1 else 0.0

                if vis == 1:
                    num_detections += 1
                elif vis == 0:
                    x, y = 0.0, 0.0

                # Use just the filename without frame_ prefix for Label.csv
                label_filename = f"{local_idx:04d}.jpg"
                label_rows.append(
                    row_from_visibility(
                        file_name=label_filename,
                        x=float(x),
                        y=float(y),
                        visibility=vis,
                        score=score,
                    )
                )

            # Save Label.csv
            label_path = clip_dir / "Label.csv"
            save_label_csv(label_path, label_rows)

            # Create clip info
            detection_rate = (
                num_detections / segment_length if segment_length > 0 else 0.0
            )
            completion_rate = (
                num_completed / segment_length if segment_length > 0 else 0.0
            )
            detected_scores = segment_score[
                segment_vis & (segment_score >= self.config.score_threshold)
            ]
            avg_score = (
                float(np.mean(detected_scores)) if len(detected_scores) > 0 else 0.0
            )

            clips.append(
                ClipInfo(
                    clip_name=clip_name,
                    clip_dir=clip_dir,
                    num_frames=segment_length,
                    num_detections=num_detections,
                    num_completed=num_completed,
                    detection_rate=detection_rate,
                    completion_rate=completion_rate,
                    avg_score=avg_score,
                )
            )

        return clips

    def _print_summary(self, result: PipelineResult) -> None:
        """Print pipeline execution summary."""
        print("\n" + "=" * 50)
        print("Pipeline Summary")
        print("=" * 50)
        print(f"Game: {result.game_name}")
        print(f"Output: {result.output_dir}")
        print(f"Total frames: {result.total_frames}")

        det_pct = 100 * result.total_detections / result.total_frames if result.total_frames > 0 else 0
        print(f"Total detections (vis=1): {result.total_detections} ({det_pct:.1f}%)")

        if result.total_completed > 0:
            comp_pct = 100 * result.total_completed / result.total_frames if result.total_frames > 0 else 0
            print(f"Total completed (vis=2): {result.total_completed} ({comp_pct:.1f}%)")

        if result.total_outliers_removed > 0:
            print(f"Outliers removed: {result.total_outliers_removed}")

        print(f"Clips generated: {len(result.clips)}")
        print()

        if result.clips:
            print("Clip details:")
            for clip in result.clips:
                completion_info = ""
                if clip.num_completed > 0:
                    completion_info = f", {clip.num_completed} completed ({100 * clip.completion_rate:.1f}%)"
                print(
                    f"  {clip.clip_name}: {clip.num_frames} frames, "
                    f"{clip.num_detections} detections ({100 * clip.detection_rate:.1f}%)"
                    f"{completion_info}, avg score: {clip.avg_score:.2f}"
                )


def run_pipeline_from_config(
    video_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    model_type: Literal["wasb", "hrcnet"] = "wasb",
    config: PipelineConfig | None = None,
    **predictor_kwargs,
) -> PipelineResult:
    """Convenience function to run pipeline with automatic predictor setup.

    Args:
        video_path: Path to input video.
        output_dir: Output directory for generated game.
        checkpoint_path: Path to model checkpoint.
        model_type: Model type ("wasb" or "hrcnet").
        config: Pipeline configuration.
        **predictor_kwargs: Additional arguments for predictor.

    Returns:
        PipelineResult with generated clips information.

    """
    from src.wasb.inference import HRCNetWASBPredictor, WASBPredictor

    # Load predictor
    if model_type == "wasb":
        predictor = WASBPredictor.load_from_checkpoint(
            checkpoint_path, **predictor_kwargs
        )
    elif model_type == "hrcnet":
        predictor = HRCNetWASBPredictor.load_from_checkpoint(
            checkpoint_path, **predictor_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run pipeline
    pipeline = AnnotationPipeline(predictor, config=config)
    return pipeline.run(video_path, output_dir)

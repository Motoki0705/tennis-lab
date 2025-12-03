"""Video frame extraction utilities for tennis dataset generation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class VideoExtractor:
    """Extract frames from video files for tennis dataset generation.

    This class handles video I/O operations including:
    - Loading video files and extracting frames
    - Saving frames to disk in the tennis dataset format
    - Iterating over frames in batches for memory efficiency

    Example:
        >>> extractor = VideoExtractor("match.mp4")
        >>> extractor.extract_to_directory("data/tennis/game11/frames")
        >>> # Or iterate over frames
        >>> for batch in extractor.iter_batches(batch_size=100):
        ...     results = predictor.predict(batch)

    """

    def __init__(self, video_path: str | Path) -> None:
        """Initialize the video extractor.

        Args:
            video_path: Path to the input video file.

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video cannot be opened.

        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Open video to get metadata
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @property
    def fps(self) -> float:
        """Video frame rate."""
        return self._fps

    @property
    def width(self) -> int:
        """Video width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Video height in pixels."""
        return self._height

    @property
    def frame_count(self) -> int:
        """Total number of frames in video."""
        return self._frame_count

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    def load_all_frames(
        self,
        max_frames: int | None = None,
    ) -> NDArray[np.uint8]:
        """Load all frames from video into memory.

        Args:
            max_frames: Maximum number of frames to load (None for all).

        Returns:
            RGB frames array with shape (T, H, W, 3).

        Warning:
            This loads all frames into memory. Use iter_batches for large videos.

        """
        cap = cv2.VideoCapture(str(self.video_path))
        frames = []

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            count += 1
            if max_frames is not None and count >= max_frames:
                break

        cap.release()
        return np.array(frames, dtype=np.uint8)

    def iter_batches(
        self,
        batch_size: int = 100,
        overlap: int = 0,
    ) -> Iterator[tuple[NDArray[np.uint8], int]]:
        """Iterate over video frames in batches.

        Args:
            batch_size: Number of frames per batch.
            overlap: Number of overlapping frames between batches.
                     Useful for models that need temporal context.

        Yields:
            Tuple of (frames, start_index) where frames is RGB array
            with shape (B, H, W, 3) and start_index is the frame index
            of the first frame in the batch.

        """
        cap = cv2.VideoCapture(str(self.video_path))
        batch = []
        batch_start = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(frame_rgb)

            if len(batch) >= batch_size:
                yield np.array(batch, dtype=np.uint8), batch_start

                # Keep overlap frames for next batch
                if overlap > 0:
                    batch = batch[-overlap:]
                    batch_start = frame_idx - overlap + 1
                else:
                    batch = []
                    batch_start = frame_idx + 1

            frame_idx += 1

        # Yield remaining frames
        if batch:
            yield np.array(batch, dtype=np.uint8), batch_start

        cap.release()

    def extract_to_directory(
        self,
        output_dir: str | Path,
        frame_format: str = "frame_{:04d}.jpg",
        start_index: int = 0,
        max_frames: int | None = None,
        jpeg_quality: int = 95,
    ) -> list[str]:
        """Extract frames and save to directory.

        Args:
            output_dir: Directory to save frames.
            frame_format: Format string for frame filenames.
            start_index: Starting index for frame numbering.
            max_frames: Maximum frames to extract (None for all).
            jpeg_quality: JPEG quality (0-100).

        Returns:
            List of saved frame filenames (without directory path).

        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video_path))
        saved_files = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            filename = frame_format.format(start_index + frame_idx)
            filepath = output_dir / filename

            cv2.imwrite(
                str(filepath),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            saved_files.append(filename)

            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

        cap.release()
        return saved_files

    def extract_segment(
        self,
        start_frame: int,
        end_frame: int,
        output_dir: str | Path,
        frame_format: str = "frame_{:04d}.jpg",
        jpeg_quality: int = 95,
    ) -> list[str]:
        """Extract a segment of frames and save to directory.

        Args:
            start_frame: Starting frame index (inclusive).
            end_frame: Ending frame index (exclusive).
            output_dir: Directory to save frames.
            frame_format: Format string for frame filenames.
            jpeg_quality: JPEG quality (0-100).

        Returns:
            List of saved frame filenames.

        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        saved_files = []
        for local_idx in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break

            filename = frame_format.format(local_idx)
            filepath = output_dir / filename

            cv2.imwrite(
                str(filepath),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            saved_files.append(filename)

        cap.release()
        return saved_files

    def get_frame(self, frame_idx: int) -> NDArray[np.uint8]:
        """Get a single frame by index.

        Args:
            frame_idx: Frame index (0-indexed).

        Returns:
            RGB frame with shape (H, W, 3).

        Raises:
            IndexError: If frame index is out of range.

        """
        if frame_idx < 0 or frame_idx >= self._frame_count:
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {self._frame_count})"
            )

        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise IndexError(f"Failed to read frame at index {frame_idx}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __repr__(self) -> str:
        return (
            f"VideoExtractor('{self.video_path}', "
            f"{self.width}x{self.height}, "
            f"{self.frame_count} frames, "
            f"{self.fps:.1f} fps)"
        )

"""Video processing utilities for BLCS demo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VideoProcessor:
    """Load and process video files for BLCS inference.

    Uses lazy loading to avoid memory issues with large videos.

    Example:
        >>> processor = VideoProcessor()
        >>> processor.open_video("video.mp4")
        >>> frame = processor.get_frame(0)

    """

    def __init__(self, max_frames: int | None = None) -> None:
        """Initialize video processor.

        Args:
            max_frames: Maximum frames to load for batch operations.

        """
        self._max_frames = max_frames
        self._video_path: Path | None = None
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0

    def open_video(self, video_path: str | Path) -> dict[str, float | int]:
        """Open video file for lazy frame access.

        Args:
            video_path: Path to video file.

        Returns:
            Video info dictionary with fps, width, height, total_frames.

        Raises:
            FileNotFoundError: If video file does not exist.
            RuntimeError: If video cannot be opened.

        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._video_path = video_path
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return {
            "fps": self._fps,
            "width": self._width,
            "height": self._height,
            "total_frames": self._total_frames,
        }

    def close(self) -> None:
        """Release video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_single_frame(self, index: int) -> NDArray[np.uint8]:
        """Get a single frame by index (lazy loading).

        Args:
            index: Frame index.

        Returns:
            Frame as RGB array, shape (H, W, 3).

        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open_video() first.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {index}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def load_video(
        self,
        video_path: str | Path,
    ) -> tuple[NDArray[np.uint8], float, int]:
        """Load video file and extract frames into memory.

        WARNING: This loads all frames into memory. For large videos,
        use open_video() and iterate_frames() instead.

        Args:
            video_path: Path to video file.

        Returns:
            Tuple of (frames, fps, total_frames):
                - frames: RGB frames array, shape (T, H, W, 3)
                - fps: Video frame rate
                - total_frames: Total number of frames in video

        """
        info = self.open_video(video_path)

        # Determine how many frames to read
        frames_to_read = self._total_frames
        if self._max_frames is not None:
            frames_to_read = min(frames_to_read, self._max_frames)

        frames = []
        for i in range(frames_to_read):
            ret, frame = self._cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frames_array = np.stack(frames, axis=0)
        return frames_array, self._fps, self._total_frames

    def iterate_frames(
        self,
        start: int = 0,
        end: int | None = None,
        step: int = 1,
        batch_size: int = 32,
    ):
        """Iterate over frames in batches (memory efficient).

        Args:
            start: Start frame index.
            end: End frame index (exclusive). None for all frames.
            step: Frame step.
            batch_size: Number of frames per batch.

        Yields:
            Tuple of (batch_frames, frame_indices) where batch_frames
            is shape (B, H, W, 3) and frame_indices is the list of
            frame indices in this batch.

        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open_video() first.")

        if end is None:
            end = self._total_frames

        frame_indices = list(range(start, end, step))

        for batch_start in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[batch_start : batch_start + batch_size]
            batch_frames = []

            for idx in batch_indices:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self._cap.read()
                if ret:
                    batch_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if batch_frames:
                yield np.stack(batch_frames, axis=0), batch_indices

    @property
    def fps(self) -> float:
        """Get video FPS."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Get total frame count."""
        return self._total_frames

    @property
    def resolution(self) -> tuple[int, int]:
        """Get video resolution (width, height)."""
        return self._width, self._height

    def extract_frame_range(
        self,
        frames: NDArray[np.uint8],
        start: int,
        end: int,
        step: int = 1,
    ) -> NDArray[np.uint8]:
        """Extract a range of frames.

        Args:
            frames: Full frames array, shape (T, H, W, 3).
            start: Start frame index (inclusive).
            end: End frame index (exclusive).
            step: Frame step size.

        Returns:
            Extracted frames, shape (N, H, W, 3).

        """
        return frames[start:end:step].copy()

    def get_frame(self, frames: NDArray[np.uint8], index: int) -> NDArray[np.uint8]:
        """Get a single frame.

        Args:
            frames: Frames array.
            index: Frame index.

        Returns:
            Single frame, shape (H, W, 3).

        """
        return frames[index].copy()

    @staticmethod
    def get_video_info(video_path: str | Path) -> dict[str, float | int]:
        """Get video metadata without loading all frames.

        Args:
            video_path: Path to video file.

        Returns:
            Dictionary with fps, width, height, total_frames.

        """
        cap = cv2.VideoCapture(str(video_path))
        try:
            info = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
        finally:
            cap.release()
        return info

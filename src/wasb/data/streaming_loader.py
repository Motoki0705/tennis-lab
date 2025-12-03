"""Streaming video frame loader with producer-consumer pattern.

This module provides memory-efficient video processing by using a bounded
queue to pipeline I/O and inference operations.

Example:
    >>> loader = StreamingVideoLoader(video_path, batch_size=16)
    >>> for batch in loader:
    ...     results = predictor.predict_batch(batch.frames)

"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class FrameBatch:
    """A batch of video frames for inference.

    Attributes:
        frames: Batch of frames with shape (B, H, W, 3) in BGR format.
        frame_indices: List of original frame indices.
        is_last: Flag indicating this is the final batch.

    """

    frames: NDArray[np.uint8]
    frame_indices: list[int]
    is_last: bool = False


@dataclass
class VideoMetadata:
    """Video file metadata.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        total_frames: Total number of frames.
        duration_sec: Duration in seconds.

    """

    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float


class StreamingVideoLoader:
    """Memory-efficient streaming video loader using producer-consumer pattern.

    This loader reads video frames in a background thread and queues them
    for consumption, allowing I/O and inference to run in parallel.

    Attributes:
        video_path: Path to the video file.
        batch_size: Number of frames per batch.
        queue_size: Maximum number of batches in queue.
        metadata: Video metadata (available after initialization).

    Example:
        >>> loader = StreamingVideoLoader("match.mp4", batch_size=16)
        >>> print(f"Processing {loader.metadata.total_frames} frames")
        >>> for batch in loader:
        ...     # Process batch.frames (shape: B, H, W, 3)
        ...     pass

    """

    def __init__(
        self,
        video_path: str | Path,
        batch_size: int = 16,
        queue_size: int = 4,
        max_frames: int | None = None,
    ) -> None:
        """Initialize streaming loader.

        Args:
            video_path: Path to video file.
            batch_size: Number of frames per batch.
            queue_size: Maximum batches to buffer (controls memory usage).
            max_frames: Maximum frames to process (None for all).

        """
        self.video_path = Path(video_path)
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.max_frames = max_frames

        # Validate video exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Load metadata
        self.metadata = self._load_metadata()

        # Internal state
        self._queue: Queue[FrameBatch | None] = Queue(maxsize=queue_size)
        self._producer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: Exception | None = None

    def _load_metadata(self) -> VideoMetadata:
        """Load video metadata without reading all frames."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Apply max_frames limit
            if self.max_frames is not None:
                total_frames = min(total_frames, self.max_frames)

            duration_sec = total_frames / fps if fps > 0 else 0

            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_sec=duration_sec,
            )
        finally:
            cap.release()

    def _producer_worker(self) -> None:
        """Background thread that reads frames and puts batches in queue."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self._error = RuntimeError(f"Failed to open video: {self.video_path}")
            self._queue.put(None)  # Signal end
            return

        try:
            frame_idx = 0
            batch_frames: list[NDArray[np.uint8]] = []
            batch_indices: list[int] = []
            frames_to_read = self.metadata.total_frames

            while frame_idx < frames_to_read and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                frame_idx += 1

                # Emit batch when full
                if len(batch_frames) >= self.batch_size:
                    batch = FrameBatch(
                        frames=np.stack(batch_frames, axis=0),
                        frame_indices=batch_indices.copy(),
                        is_last=False,
                    )
                    batch_frames.clear()
                    batch_indices.clear()

                    # Put with timeout to allow stop check
                    while not self._stop_event.is_set():
                        try:
                            self._queue.put(batch, timeout=0.1)
                            break
                        except Full:
                            continue

            # Emit final partial batch
            if batch_frames and not self._stop_event.is_set():
                batch = FrameBatch(
                    frames=np.stack(batch_frames, axis=0),
                    frame_indices=batch_indices.copy(),
                    is_last=True,
                )
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except Full:
                        continue

        except Exception as e:
            self._error = e
        finally:
            cap.release()
            # Signal end of stream
            try:
                self._queue.put(None, timeout=1.0)
            except Full:
                pass

    def __iter__(self) -> Iterator[FrameBatch]:
        """Iterate over frame batches.

        Yields:
            FrameBatch objects containing frames and metadata.

        Raises:
            RuntimeError: If producer thread encounters an error.

        """
        # Start producer thread
        self._stop_event.clear()
        self._error = None
        self._producer_thread = threading.Thread(
            target=self._producer_worker,
            daemon=True,
        )
        self._producer_thread.start()

        try:
            while True:
                # Get batch with timeout to allow interrupt
                try:
                    batch = self._queue.get(timeout=1.0)
                except Empty:
                    # Check if producer is still running
                    if not self._producer_thread.is_alive():
                        break
                    continue

                if batch is None:
                    # End of stream
                    break

                # Check for producer errors
                if self._error is not None:
                    raise self._error

                yield batch

            # Final error check
            if self._error is not None:
                raise self._error

        finally:
            # Cleanup
            self._stop_event.set()
            if self._producer_thread is not None:
                self._producer_thread.join(timeout=5.0)

    def stop(self) -> None:
        """Stop the loader and cleanup resources."""
        self._stop_event.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=5.0)


class StreamingFrameIterator:
    """Simple streaming iterator for frame-by-frame processing.

    Unlike StreamingVideoLoader which batches frames, this yields
    individual frames. Useful for frame extraction.

    """

    def __init__(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
    ) -> None:
        """Initialize iterator.

        Args:
            video_path: Path to video file.
            max_frames: Maximum frames to yield.

        """
        self.video_path = Path(video_path)
        self.max_frames = max_frames

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def __iter__(self) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        """Iterate over frames.

        Yields:
            Tuple of (frame_index, frame_array).

        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        try:
            frame_idx = 0
            while True:
                if self.max_frames is not None and frame_idx >= self.max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                yield frame_idx, frame
                frame_idx += 1
        finally:
            cap.release()

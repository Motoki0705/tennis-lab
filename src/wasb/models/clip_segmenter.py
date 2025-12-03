"""Clip segmentation for tennis video analysis.

This module provides segmentation of long tennis videos into individual
rally clips based on ball trajectory analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ClipSegment:
    """Represents a detected rally clip segment.

    Attributes:
        start: Start frame index (inclusive).
        end: End frame index (exclusive).
        detection_rate: Fraction of frames with ball detection in segment.
        avg_score: Average detection score in segment.

    """

    start: int
    end: int
    detection_rate: float
    avg_score: float

    @property
    def length(self) -> int:
        """Number of frames in segment."""
        return self.end - self.start


class ClipSegmenter(ABC):
    """Abstract base class for clip segmentation.

    Clip segmenters analyze ball trajectory data to identify
    individual rally segments within a longer video.
    """

    @abstractmethod
    def predict_segments(
        self,
        xy: NDArray[np.floating],
        score: NDArray[np.floating],
        visibility: NDArray[np.bool_],
    ) -> list[ClipSegment]:
        """Predict rally segments from ball trajectory data.

        Args:
            xy: Ball coordinates with shape (T, 2).
            score: Detection confidence scores with shape (T,).
            visibility: Visibility flags with shape (T,).

        Returns:
            List of ClipSegment objects representing detected rallies.

        """
        ...


class RuleBasedClipSegmenter(ClipSegmenter):
    """Rule-based clip segmenter using simple heuristics.

    This segmenter identifies rally segments based on:
    - Consecutive frames with ball detections
    - Minimum clip length requirements
    - Detection rate thresholds
    - Gap bridging for short detection failures

    Args:
        min_clip_length: Minimum number of frames for a valid clip.
        min_detection_rate: Minimum fraction of frames with detections.
        max_gap: Maximum gap (in frames) to bridge between detections.
        score_threshold: Minimum score to consider a detection valid.
        padding_frames: Extra frames to include at segment boundaries.

    Example:
        >>> segmenter = RuleBasedClipSegmenter(min_clip_length=30)
        >>> segments = segmenter.predict_segments(xy, score, visibility)
        >>> for seg in segments:
        ...     print(f"Clip: frames {seg.start}-{seg.end}")

    """

    def __init__(
        self,
        min_clip_length: int = 30,
        min_detection_rate: float = 0.5,
        max_gap: int = 10,
        score_threshold: float = 0.3,
        padding_frames: int = 5,
    ) -> None:
        self.min_clip_length = min_clip_length
        self.min_detection_rate = min_detection_rate
        self.max_gap = max_gap
        self.score_threshold = score_threshold
        self.padding_frames = padding_frames

    def predict_segments(
        self,
        xy: NDArray[np.floating],
        score: NDArray[np.floating],
        visibility: NDArray[np.bool_],
    ) -> list[ClipSegment]:
        """Predict rally segments using rule-based heuristics.

        The algorithm:
        1. Create binary mask of valid detections (visibility=True, score>=threshold)
        2. Bridge small gaps (<=max_gap frames) between detections
        3. Find continuous segments of detections
        4. Filter segments by minimum length and detection rate
        5. Add padding frames to segment boundaries

        Args:
            xy: Ball coordinates with shape (T, 2).
            score: Detection confidence scores with shape (T,).
            visibility: Visibility flags with shape (T,).

        Returns:
            List of ClipSegment objects sorted by start frame.

        """
        num_frames = len(visibility)
        if num_frames == 0:
            return []

        # Step 1: Create detection mask
        detection_mask = self._create_detection_mask(visibility, score)

        # Step 2: Bridge small gaps
        bridged_mask = self._bridge_gaps(detection_mask)

        # Step 3: Find continuous segments
        raw_segments = self._find_continuous_segments(bridged_mask)

        # Step 4: Filter and validate segments
        valid_segments = self._filter_segments(
            raw_segments, detection_mask, score, num_frames
        )

        return valid_segments

    def _create_detection_mask(
        self,
        visibility: NDArray[np.bool_],
        score: NDArray[np.floating],
    ) -> NDArray[np.bool_]:
        """Create binary mask of valid detections."""
        return visibility & (score >= self.score_threshold)

    def _bridge_gaps(
        self,
        mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """Bridge small gaps in detection mask.

        Uses morphological closing to connect nearby detections.
        """
        result = mask.copy()
        num_frames = len(mask)

        # Find gaps and check if they should be bridged
        gap_start = None
        for i in range(num_frames):
            if not mask[i]:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_length = i - gap_start
                    # Bridge if gap is small and there are detections on both sides
                    if gap_length <= self.max_gap:
                        if gap_start > 0 and mask[gap_start - 1]:
                            result[gap_start:i] = True
                    gap_start = None

        return result

    def _find_continuous_segments(
        self,
        mask: NDArray[np.bool_],
    ) -> list[tuple[int, int]]:
        """Find continuous segments of True values in mask.

        Returns:
            List of (start, end) tuples where end is exclusive.

        """
        segments = []
        in_segment = False
        segment_start = 0

        for i, val in enumerate(mask):
            if val and not in_segment:
                # Start of new segment
                in_segment = True
                segment_start = i
            elif not val and in_segment:
                # End of segment
                in_segment = False
                segments.append((segment_start, i))

        # Handle segment that extends to end
        if in_segment:
            segments.append((segment_start, len(mask)))

        return segments

    def _filter_segments(
        self,
        raw_segments: list[tuple[int, int]],
        detection_mask: NDArray[np.bool_],
        score: NDArray[np.floating],
        num_frames: int,
    ) -> list[ClipSegment]:
        """Filter segments by length and detection rate, add padding."""
        valid_segments = []

        for start, end in raw_segments:
            length = end - start

            # Check minimum length
            if length < self.min_clip_length:
                continue

            # Calculate detection rate in original (non-bridged) mask
            segment_detections = detection_mask[start:end]
            detection_rate = np.mean(segment_detections)

            # Check detection rate threshold
            if detection_rate < self.min_detection_rate:
                continue

            # Calculate average score for detected frames
            segment_scores = score[start:end]
            detected_scores = segment_scores[segment_detections]
            avg_score = (
                float(np.mean(detected_scores)) if len(detected_scores) > 0 else 0.0
            )

            # Add padding
            padded_start = max(0, start - self.padding_frames)
            padded_end = min(num_frames, end + self.padding_frames)

            valid_segments.append(
                ClipSegment(
                    start=padded_start,
                    end=padded_end,
                    detection_rate=detection_rate,
                    avg_score=avg_score,
                )
            )

        return valid_segments

    def __repr__(self) -> str:
        return (
            f"RuleBasedClipSegmenter("
            f"min_clip_length={self.min_clip_length}, "
            f"min_detection_rate={self.min_detection_rate}, "
            f"max_gap={self.max_gap}, "
            f"score_threshold={self.score_threshold})"
        )

"""Method-independent input and output; every emitted observation names its detection."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.appearance.parts import NativeParts


@dataclass(frozen=True)
class DetectionFeatures:
    frame: int
    rows: NDArray[np.int64]  # original, clip-global detection row IDs
    boxes: NDArray[np.float32]  # N,4 source-pixel xyxy
    scores: NDArray[np.float32]  # N
    poses: NDArray[np.float32] | None  # None explicitly means no pose was inferred
    embeddings: NDArray[np.float32]  # N,E; zero when appearance_valid is false
    appearance_valid: NDArray[np.bool_]
    parts: NativeParts | None = None

    def __post_init__(self) -> None:
        n = len(self.rows)
        if self.parts is not None and (len(self.parts.embeddings) != n or self.appearance_valid.any()):
            raise ValueError('Native parts must align to detections and cannot silently coexist with whole-image features')
        if type(self.frame) is not int or self.frame < 0 or self.rows.shape != (n,) or self.rows.dtype != np.int64 \
                or (self.rows < 0).any() or len(np.unique(self.rows)) != n:
            raise ValueError("Features require a nonnegative frame and unique int64 detection rows")
        if self.boxes.shape != (n, 4) or self.scores.shape != (n,) \
                or (self.poses is not None and self.poses.shape != (n, 17, 3)) \
                or self.embeddings.ndim != 2 or self.embeddings.shape[0] != n or self.embeddings.shape[1] < 1 \
                or self.appearance_valid.shape != (n,) or self.appearance_valid.dtype != np.bool_:
            raise ValueError("Feature arrays must share the detection axis")
        for array in (self.boxes, self.scores, self.poses, self.embeddings):
            if array is None:
                continue
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("Feature values must be finite float32")
        if (self.boxes[:, 2:] <= self.boxes[:, :2]).any() or ((self.scores < 0) | (self.scores > 1)).any():
            raise ValueError("Boxes must have positive area and detection scores must be probabilities")
        norms = np.linalg.norm(self.embeddings[self.appearance_valid], axis=1)
        if not np.allclose(norms, 1., atol=1e-4) or (self.embeddings[~self.appearance_valid] != 0).any():
            raise ValueError("Valid embeddings must have unit norm; masked embeddings must be zero")

    def require_poses(self) -> NDArray[np.float32]:
        if self.poses is None:
            raise ValueError("This tracking method requires inferred pose features")
        return self.poses


@dataclass(frozen=True)
class TrackAssignments:
    """Only real detections, never predicted/interpolated boxes. IDs are camera-local."""

    frame: int
    detection_rows: NDArray[np.int64]
    track_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.detection_rows.ndim != 1 or self.track_ids.shape != self.detection_rows.shape \
                or self.detection_rows.dtype != np.int64 or self.track_ids.dtype != np.int64:
            raise ValueError("Assignments require aligned int64 rows and IDs")
        if (self.detection_rows < 0).any() or (self.track_ids < 1).any() \
                or len(np.unique(self.detection_rows)) != len(self.detection_rows) \
                or len(np.unique(self.track_ids)) != len(self.track_ids):
            raise ValueError("A detection and a track may each occur only once per frame")


class TrackingMethod(Protocol):
    def update(self, features: DetectionFeatures) -> TrackAssignments: ...


class TrackCapacityExceeded(RuntimeError):
    """Cumulative IDs exceeded the configured cap; no recycling or forced merging."""

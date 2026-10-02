"""Hash-checked raw detection/timing archive for the far-player diagnostic."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import PersonDetectionResult
from src.utils.checksum import dual_sha256


@dataclass(frozen=True)
class DetectionArchive:
    offsets: NDArray[np.int64]
    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    milliseconds: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.offsets.ndim != 1 or self.offsets.dtype != np.int64 or len(self.offsets) < 2 \
                or self.offsets[0] != 0 or (np.diff(self.offsets) < 0).any() or self.offsets[-1] != len(self.scores):
            raise ValueError('Invalid detection archive timeline')
        if self.boxes.shape != (len(self.scores), 4) or self.boxes.dtype != np.float32 \
                or self.scores.ndim != 1 or self.scores.dtype != np.float32 \
                or self.milliseconds.shape != (len(self.offsets) - 1,) or self.milliseconds.dtype != np.float64:
            raise ValueError('Invalid detection archive axes/dtypes')
        if not all(np.isfinite(a).all() for a in (self.boxes, self.scores, self.milliseconds)) \
                or (self.boxes[:, 2:] < self.boxes[:, :2]).any() or (self.milliseconds < 0).any() \
                or ((self.scores < 0) | (self.scores > 1)).any():
            raise ValueError('Invalid detection archive values')

    def at(self, frame: int) -> PersonDetectionResult:
        start, end = self.offsets[frame:frame + 2]
        return PersonDetectionResult(self.boxes[start:end], self.scores[start:end])

    def save(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            np.savez_compressed(handle, offsets=self.offsets, boxes=self.boxes,
                                scores=self.scores, milliseconds=self.milliseconds)
        return {'path': str(path), 'sha256': dual_sha256(path), 'frames': len(self.milliseconds),
                'detections': len(self.scores), 'ms_per_frame': float(self.milliseconds.mean())}

    @classmethod
    def load(cls, record: dict[str, Any]) -> 'DetectionArchive':
        path = Path(record['path'])
        if dual_sha256(path) != record['sha256']:
            raise ValueError(f'Detection archive checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'offsets', 'boxes', 'scores', 'milliseconds'}:
                raise ValueError('Unexpected detection archive fields')
            result = cls(**{key: data[key] for key in data.files})
        if len(result.milliseconds) != record['frames']:
            raise ValueError('Detection archive frame count mismatch')
        return result

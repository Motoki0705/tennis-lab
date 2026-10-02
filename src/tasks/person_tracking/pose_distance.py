"""Shared confidence-masked pose evidence in detection-box coordinates."""

import numpy as np
from numpy.typing import NDArray


def local_pose(box: NDArray[np.float32], pose: NDArray[np.float32]) -> NDArray[np.float32]:
    result = pose.copy()
    result[:, :2] = (pose[:, :2] - box[:2]) / (box[2:] - box[:2])
    return result


def pose_distance(a: NDArray[np.float32], b: NDArray[np.float32], *, confidence: float = .3,
                  min_joints: int = 4, scale: float = .25) -> float | None:
    valid = (a[:, 2] >= confidence) & (b[:, 2] >= confidence)
    if int(valid.sum()) < min_joints:
        return None
    weights = np.minimum(a[valid, 2], b[valid, 2])
    distance = np.linalg.norm(a[valid, :2] - b[valid, :2], axis=1)
    return min(1., float(np.average(distance, weights=weights)) / scale)

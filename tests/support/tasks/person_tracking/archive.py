"""Small historical feature archive, independent of the current writer."""

import json
from pathlib import Path

import numpy as np


def write_v2_features(path: Path) -> None:
    poses: np.ndarray = np.zeros((2, 17, 3), np.float32)
    poses[0, :, 2] = np.linspace(-2, 3, 17, dtype=np.float32)
    np.savez_compressed(
        path,
        rows=np.array([7, 11], np.int64),
        boxes=np.array([[0, 0, 10, 20], [20, 0, 30, 20]], np.float32),
        scores=np.array([0, 0.75], np.float32),
        poses=poses,
        embeddings=np.array([[1, 0], [0, 0]], np.float32),
        appearance_valid=np.array([True, False]),
        offsets=np.array([0, 1, 1, 2], np.int64),
        metadata=np.asarray(json.dumps({
            "schema": "person_detection_features_v2",
            "provenance": {"detection_sha256": "a" * 64, "config": {"pose": "original"}},
        })),
    )

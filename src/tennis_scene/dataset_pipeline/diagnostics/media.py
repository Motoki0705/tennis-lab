"""Validate diagnostic video metadata against the canonical clip manifest."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.tennis_scene.generate_dataset.manifest import ClipManifest


def sample_frames(
    clip: ClipManifest, camera: str, indices: np.ndarray
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(clip.media_path(camera)))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{camera}: unreadable video or invalid FPS")
        if not math.isclose(fps, clip.fps, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(
                f"{camera}: video FPS {fps} differs from manifest {clip.fps}"
            )
        for prop, expected in (
            (cv2.CAP_PROP_FRAME_WIDTH, clip.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, clip.height),
            (cv2.CAP_PROP_FRAME_COUNT, clip.num_frames),
        ):
            actual = capture.get(prop)
            if not math.isfinite(actual) or int(round(actual)) != expected:
                raise ValueError(
                    f"{camera}: video dimensions/frame count differ from manifest"
                )
        result = []
        for index in indices:
            if index < 0 or index >= clip.num_frames:
                raise ValueError(f"{camera}: sampled frame outside clip: {index}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot read {camera}:{index}")
            result.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return result
    finally:
        capture.release()

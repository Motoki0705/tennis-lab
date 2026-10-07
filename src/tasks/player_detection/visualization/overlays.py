"""Shared bbox overlays for player evaluation and training clips."""

from __future__ import annotations

import cv2
import numpy as np

from src.tasks.player_detection.models.dino_detector import FrameDetections


def draw_overlay(
    bgr: np.ndarray, targets: np.ndarray, detections: FrameDetections, *, threshold: float, title: str
) -> np.ndarray:
    """GT in green, detections above ``threshold`` in red with scores."""
    canvas = bgr.copy()
    for x1, y1, x2, y2 in targets.round().astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
    keep = detections.scores >= threshold
    for (x1, y1, x2, y2), score in zip(
        detections.boxes_xyxy[keep].round().int().tolist(), detections.scores[keep].tolist(), strict=True
    ):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f"{score:.2f}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(canvas, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return canvas

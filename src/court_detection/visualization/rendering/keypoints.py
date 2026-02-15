"""Rendering utilities for court keypoint visualization."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.utils.schema.court import COURT_KP_NAMES, COURT_SKELETON, NUM_COURT_KP

NUM_KEYPOINTS = NUM_COURT_KP
KEYPOINT_NAMES = list(COURT_KP_NAMES)


def visualize_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Visualize court keypoints on an image."""
    config = config or {}

    point_radius = int(config.get("point_radius", 5))
    point_color = tuple(config.get("point_color", [0, 255, 0]))
    line_color = tuple(config.get("line_color", [255, 255, 0]))
    text_color = tuple(config.get("text_color", [255, 255, 255]))
    line_thickness = int(config.get("line_thickness", 2))
    show_keypoint_ids = bool(config.get("show_keypoint_ids", True))
    show_court_lines = bool(config.get("show_court_lines", True))
    visibility_threshold = float(config.get("visibility_threshold", 0.5))

    if visibility is None:
        visibility = np.ones(len(keypoints), dtype=np.float32)

    annotated = image.copy()

    if show_court_lines:
        for i, j in COURT_SKELETON:
            if visibility[i] > visibility_threshold and visibility[j] > visibility_threshold:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(annotated, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)

    for idx in range(len(keypoints)):
        if visibility[idx] > visibility_threshold:
            x, y = int(keypoints[idx, 0]), int(keypoints[idx, 1])
            cv2.circle(annotated, (x, y), point_radius, point_color, -1, cv2.LINE_AA)
            if show_keypoint_ids:
                cv2.putText(
                    annotated,
                    str(idx),
                    (x + point_radius + 2, y - point_radius),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

    return annotated


def visualize_heatmaps(
    heatmaps: np.ndarray,
    image: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Visualize heatmaps overlaid on image."""
    combined = np.sum(heatmaps, axis=0)
    combined = np.clip(combined, 0, 1)

    heatmap_color = cv2.applyColorMap(
        (combined * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    if image is not None:
        heatmap_color = cv2.resize(
            heatmap_color,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return heatmap_color


def draw_court_overlay(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray | None = None,
    pred_keypoints: np.ndarray | None = None,
    pred_visibility: np.ndarray | None = None,
) -> np.ndarray:
    """Draw GT and predicted keypoints for comparison."""
    annotated = image.copy()

    if visibility is None:
        visibility = np.ones(len(keypoints), dtype=np.float32)

    for idx in range(len(keypoints)):
        if visibility[idx] > 0.5:
            x, y = int(keypoints[idx, 0]), int(keypoints[idx, 1])
            cv2.circle(annotated, (x, y), 6, (0, 255, 0), 2, cv2.LINE_AA)

    if pred_keypoints is not None:
        if pred_visibility is None:
            pred_visibility = np.ones(len(pred_keypoints), dtype=np.float32)

        for idx in range(len(pred_keypoints)):
            if pred_visibility[idx] > 0.5:
                x, y = int(pred_keypoints[idx, 0]), int(pred_keypoints[idx, 1])
                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(annotated, "GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, "Pred", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return annotated

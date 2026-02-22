"""Reporting helpers for court detection visualization."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.court_detection.visualization.types import KeypointPrediction, RunSummary, SceneImage


def save_overlay(output_dir: Path, scene: SceneImage, overlay_bgr: np.ndarray) -> Path:
    """Save overlay image to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene.image_path.stem}_pred.png"
    cv2.imwrite(str(output_path), overlay_bgr)
    return output_path


def save_prediction_json(output_dir: Path, scene: SceneImage, pred: KeypointPrediction) -> Path:
    """Save keypoint predictions as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene.image_path.stem}_pred.json"
    payload = {
        "image": str(scene.image_path),
        "keypoints": pred.keypoints.tolist(),
        "visibility": pred.visibility.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def print_summary(summary: RunSummary) -> None:
    """Print a concise run summary."""
    print(
        f"Processed {summary.total_inputs} image(s): "
        f"success={summary.succeeded}, failed={len(summary.failed)}"
    )
    if summary.failed:
        for path in summary.failed:
            print(f"Failed: {path}")

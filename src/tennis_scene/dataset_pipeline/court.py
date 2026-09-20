"""Explicit static-camera court estimation from a learned KP checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.configuration import PathResolver
from src.utils.schema.court import CourtConfig, court_keypoints_3d


@dataclass(frozen=True)
class StaticCourtSettings:
    checkpoint: Path
    samples_per_clip: int
    min_score: float
    min_points: int
    ransac_px: float
    max_fit_error_px: float
    ball_crop_margins: dict[str, float | None] = field(default_factory=dict)
    crop_refinement_padding_px: float | None = None


def fit_static_court(
    keypoints_px: np.ndarray,
    scores: np.ndarray,
    *,
    size: tuple[int, int],
    settings: StaticCourtSettings,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a RANSAC homography to temporal median observations; reject poor fits.

    Missing channels are completed by the fitted court geometry, with their
    support explicitly recorded. They are not counted as detector observations.
    """
    if keypoints_px.ndim != 3 or keypoints_px.shape[1:] != (14, 2):
        raise ValueError("Expected sampled CourtKP14 observations (F,14,2)")
    if scores.shape != keypoints_px.shape[:-1]:
        raise ValueError("Court scores must match (F,14)")
    valid = (scores >= settings.min_score) & np.isfinite(keypoints_px).all(-1)
    counts = valid.sum(axis=0)
    supported = counts >= max(1, (len(scores) + 1) // 2)
    if int(supported.sum()) < settings.min_points:
        raise ValueError(
            f"Only {supported.sum()}/14 court channels have temporal support"
        )
    median: np.ndarray = np.zeros((14, 2), np.float32)
    for index in np.flatnonzero(supported):
        median[index] = np.median(keypoints_px[valid[:, index], index], axis=0)
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    homography, inliers = cv2.findHomography(
        physical[supported], median[supported], cv2.RANSAC, settings.ransac_px
    )
    if (
        homography is None
        or inliers is None
        or int(inliers.sum()) < settings.min_points
    ):
        raise ValueError("Static court homography has insufficient inlier support")
    projected = cv2.perspectiveTransform(physical[None], homography)[0]
    errors = np.linalg.norm(projected[supported] - median[supported], axis=-1)
    median_error = float(np.median(errors))
    if not np.isfinite(projected).all() or median_error > settings.max_fit_error_px:
        raise ValueError(f"Static court fit error too large: {median_error:.2f}px")
    normalized = projected / np.asarray(size, np.float32)
    if ((normalized < 0) | (normalized > 1)).any():
        raise ValueError("Fitted CourtKP14 is outside the source image")
    return (
        normalized.astype(np.float32),
        homography,
        {
            "method": "static_temporal_median_ransac_homography",
            "support_frames_per_channel": counts.tolist(),
            "supported_channels": np.flatnonzero(supported).tolist(),
            "inlier_channels": np.flatnonzero(supported)[
                inliers.ravel().astype(bool)
            ].tolist(),
            "fit_error_px_median": median_error,
            "fit_error_px_p95": float(np.percentile(errors, 95)),
            "is_measured_calibration": False,
        },
    )


def ball_guided_roi(
    clip: ClipManifest, camera: str, margin: float | None
) -> tuple[int, int, int, int]:
    """Use observed outsourced ball centres to identify the target-court crop."""
    if margin is None:
        return 0, 0, clip.width, clip.height
    annotation = json.loads(
        (clip.clip_dir / "outsource" / f"{camera}_annotations.json").read_text()
    )
    points = np.asarray(
        [
            [frame["center_px"]["x"], frame["center_px"]["y"]]
            for frame in annotation["frames"]
            if frame["status"] == "observed"
        ],
        np.float64,
    )
    if len(points) < 20 or not np.isfinite(points).all():
        raise ValueError(
            f"{camera}: ball-guided court ROI needs 20 finite observed centres"
        )
    low, high = np.percentile(points, [1, 99], axis=0)
    span = high - low
    if min(span) < 32:
        raise ValueError(
            f"{camera}: ball observations cover too little of the court for cropping"
        )
    start = np.maximum(0, np.floor(low - margin * span)).astype(int)
    end = np.minimum([clip.width, clip.height], np.ceil(high + margin * span)).astype(
        int
    )
    return int(start[0]), int(start[1]), int(end[0]), int(end[1])


def court_extent_roi(
    base: tuple[int, int, int, int],
    projected: np.ndarray,
    size: tuple[int, int],
    padding_px: float,
) -> tuple[int, int, int, int]:
    """Expand the union of the ball ROI and first-pass court extent, then clamp."""
    x0, y0, x1, y1 = base
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError(f"Invalid ROI {base} for image {size}")
    if not np.isfinite(padding_px) or padding_px < 0:
        raise ValueError("Court extent padding must be finite and nonnegative")
    if projected.shape != (14, 2) or not np.isfinite(projected).all():
        raise ValueError("Initial H must project 14 finite points")
    if (np.ptp(projected, axis=0) <= 0).any():
        raise ValueError("Initial H projects a degenerate court extent")
    low = np.minimum(base[:2], projected.min(0)) - padding_px
    high = np.maximum(base[2:], projected.max(0)) + padding_px
    bounds = np.concatenate(
        [np.clip(np.floor(low), 0, size), np.clip(np.ceil(high), 0, size)]
    ).astype(int)
    result = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
    if not (
        0 <= result[0] < result[2] <= size[0] and 0 <= result[1] < result[3] <= size[1]
    ):
        raise ValueError(f"Invalid refined ROI {result} for image {size}")
    return result


def observe_static_court(
    clip: ClipManifest,
    output: Path,
    *,
    resolver: PathResolver,
    settings: StaticCourtSettings,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run the configured outputs/court_detection checkpoint on sampled RGB."""
    # Court training checkpoints live under output_root, while PLCS/BLCS
    # deployment checkpoints use checkpoint_root. The input role is explicit.
    court_resolver = PathResolver(
        replace(resolver.roots, checkpoint_root=resolver.roots.output_root)
    )
    predictor = CourtKeypointPredictor.load_from_checkpoint(
        settings.checkpoint,
        resolver=court_resolver,
        device=device,
        subpixel_refine=True,
        peak_threshold=settings.min_score,
    )
    indices = np.unique(
        np.linspace(0, clip.num_frames - 1, settings.samples_per_clip)
        .round()
        .astype(int)
    )
    observations, homographies, diagnostics = [], [], []
    for camera in clip.camera_ids:
        roi = ball_guided_roi(clip, camera, settings.ball_crop_margins.get(camera))
        refine = (
            settings.crop_refinement_padding_px is not None
            and settings.ball_crop_margins.get(camera) is not None
        )
        capture = cv2.VideoCapture(str(clip.media_path(camera)))
        frames = []
        try:
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot read {camera} frame {index}")
                frames.append(frame)
        finally:
            capture.release()
        initial_h = None
        for pass_name in ("initial", "refined") if refine else ("final",):
            x0, y0, x1, y1 = roi
            keypoints, scores = [], []
            for frame in frames:
                pred = predictor.predict(
                    cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                )
                keypoints.append(pred.keypoints[:, 0].cpu().numpy() + [x0, y0])
                scores.append(pred.scores[:, 0].cpu().numpy())
            raw, confidence = np.stack(keypoints), np.stack(scores)
            sample_names = [f"{camera}_court_samples.npz"]
            if refine:
                sample_names = [f"{camera}_court_{pass_name}_samples.npz"]
                if pass_name == "refined":
                    sample_names.append(f"{camera}_court_samples.npz")
            for sample_name in sample_names:
                np.savez_compressed(
                    output / sample_name,
                    keypoints_px=raw,
                    scores=confidence,
                    frame_indices=indices,
                )
            receipt: dict[str, Any] = {
                "camera_id": camera,
                "source_roi_xyxy": roi,
                "pass": pass_name,
            }
            if initial_h is not None:
                receipt["initial_homography"] = initial_h.tolist()
            try:
                kp, h, diagnostic = fit_static_court(
                    raw, confidence, size=(clip.width, clip.height), settings=settings
                )
            except ValueError as error:
                if refine:
                    receipt.update(status="rejected", rejection=str(error))
                    (output / f"{camera}_court_{pass_name}.json").write_text(
                        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
                    )
                raise
            if refine:
                receipt.update(
                    status="fit_returned", homography=h.tolist(), **diagnostic
                )
                (output / f"{camera}_court_{pass_name}.json").write_text(
                    json.dumps(receipt, indent=2, allow_nan=False) + "\n"
                )
            if pass_name == "initial":
                initial_h = h
                physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
                assert settings.crop_refinement_padding_px is not None
                roi = court_extent_roi(
                    roi,
                    cv2.perspectiveTransform(physical[None], h)[0],
                    (clip.width, clip.height),
                    settings.crop_refinement_padding_px,
                )
        if refine:
            diagnostic = {
                **diagnostic,
                "crop_refinement_padding_px": settings.crop_refinement_padding_px,
                "initial_pass_diagnostics": f"{camera}_court_initial.json",
                "refined_pass_diagnostics": f"{camera}_court_refined.json",
            }
        observations.append(np.repeat(kp[None], clip.num_frames, axis=0))
        homographies.append(h)
        diagnostics.append({"camera_id": camera, "source_roi_xyxy": roi, **diagnostic})
        print(f"Court {camera}: {diagnostic}", flush=True)
    return np.stack(observations), np.stack(homographies), diagnostics

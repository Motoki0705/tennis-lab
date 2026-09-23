"""Explicit image-only region search for fixed-camera Court inference.

Search uses raw KP support and fitted geometry, never annotation-derived ROIs.
Raster evidence remains in the cropped prediction's own coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.model_io.contracts import CourtKeypointPrediction

if TYPE_CHECKING:
    from src.tasks.court_detection.inference.predictor import CourtPredictor

Region = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CourtRegionSearchConfig:
    enabled: bool = False
    min_inliers: int = 8
    inlier_distance_ratio: float = 0.005
    min_area_ratio: float = 0.01

    def __post_init__(self) -> None:
        if not 4 <= self.min_inliers <= 14:
            raise ValueError("region_search.min_inliers must be between 4 and 14")
        for name in ("inlier_distance_ratio", "min_area_ratio"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"region_search.{name} must be finite and between 0 and 1")


def image_content_region(image: NDArray[np.uint8]) -> Region:
    """Remove only exactly black exterior rows/columns; retain dark scene content."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Court region search requires an HWC uint8 RGB image")
    rows, columns = np.nonzero(np.any(image != 0, axis=-1))
    if not len(rows):
        raise ValueError("Court region search cannot use an entirely black image")
    return int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1


def region_proposals(image: NDArray[np.uint8]) -> tuple[Region, ...]:
    """Fixed overlapping grid, independent of checkpoint outputs and labels."""
    left, top, right, bottom = image_content_region(image)
    width, height = right - left, bottom - top
    proposals = [(0, 0, image.shape[1], image.shape[0]), (left, top, right, bottom)]
    for y in (0.0, 0.25):
        proposals.append((left, top + round(y * height), right, top + round((y + 0.75) * height)))
    for x in (0.0, 0.125, 0.25):
        for y in (0.0, 0.25, 0.5):
            proposals.append((
                left + round(x * width), top + round(y * height),
                left + round((x + 0.75) * width), top + round((y + 0.5) * height),
            ))
    return tuple(dict.fromkeys(box for box in proposals if box[2] > box[0] and box[3] > box[1]))


@dataclass(frozen=True)
class CourtRegionPrediction:
    cropped: CourtPrediction
    region: Region
    original_size_hw: tuple[int, int]

    def downstream_keypoints(self) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        geometry = self.cropped.homography
        if geometry is None:
            raise ValueError("Region geometry requires hybrid postprocessing")
        if geometry.status != "ok" or geometry.matrix is None:
            return np.zeros((14, 2), np.float32), np.zeros(14, dtype=bool)
        points = np.asarray(geometry.projected, dtype=np.float32) + np.asarray(self.region[:2], dtype=np.float32)
        height, width = self.original_size_hw
        valid = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
            & (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
        )
        return points, valid

    def geometry_diagnostics(self) -> dict[str, object]:
        diagnostic: dict[str, object] = self.cropped.geometry_diagnostics()
        raw = self.cropped.raw_heads["kp"]
        assert isinstance(raw, CourtKeypointPrediction)
        offset = np.asarray(self.region[:2])
        diagnostic["raw_keypoints_px"] = [
            [float(value) if np.isfinite(value) else None for value in point]
            for point in raw.keypoints[:, 0].numpy() + offset
        ]
        geometry = self.cropped.homography
        if geometry is not None and geometry.matrix is not None:
            translation = np.eye(3)
            translation[:2, 2] = offset
            diagnostic["homography_court_metres_to_image_pixels"] = (translation @ geometry.matrix).tolist()
        diagnostic.update(
            region_xyxy=list(self.region),
            original_size_hw=list(self.original_size_hw),
            postprocess_image_size_hw=list(self.cropped.original_size_hw),
            native_size_hw=list(self.cropped.native_size_hw),
        )
        return diagnostic


def predict_court_region(
    predictor: CourtPredictor, image: NDArray[np.uint8], region: Region,
) -> CourtRegionPrediction:
    x0, y0, x1, y1 = region
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError(f"Court region {region} is outside image {image.shape[:2]}")
    prediction = predictor.predict(
        np.ascontiguousarray(image[y0:y1, x0:x1]),
        postprocess="hybrid", heads=("kp", "line"),
    )
    return CourtRegionPrediction(prediction, region, (image.shape[0], image.shape[1]))


@dataclass(frozen=True)
class CourtRegionSelection:
    region: Region
    candidates: tuple[dict[str, object], ...]


def select_court_region(
    predictor: CourtPredictor, image: NDArray[np.uint8], config: CourtRegionSearchConfig,
) -> CourtRegionSelection:
    """Select by model inlier count, then confidence; fail if no candidate qualifies.

    This establishes a region once, not a stored homography or per-frame fallback.
    The caller reruns inference for every frame within the selected region.
    """
    left, top, right, bottom = image_content_region(image)
    height, width = image.shape[:2]
    threshold = config.inlier_distance_ratio * np.hypot(width, height)
    candidates: list[dict[str, object]] = []
    accepted: list[tuple[int, float, Region]] = []
    for region in region_proposals(image):
        prediction = predict_court_region(predictor, image, region)
        points, visible = prediction.downstream_keypoints()
        raw = prediction.cropped.raw_heads["kp"]
        assert isinstance(raw, CourtKeypointPrediction)
        raw_points = raw.keypoints[:, 0].numpy() + np.asarray(region[:2])
        inliers = (np.linalg.norm(points - raw_points, axis=-1) <= threshold) & raw.valid[:, 0].numpy()
        count = int(inliers.sum())
        score = float(raw.scores[:, 0].numpy()[inliers].sum())
        polygon = points[[0, 1, 3, 2]].astype(np.float32)
        area = abs(cv2.contourArea(polygon)) / (width * height)
        convex = bool(cv2.isContourConvex(polygon))
        inside_content = bool((
            (points[:, 0] >= left) & (points[:, 0] <= right - 1)
            & (points[:, 1] >= top) & (points[:, 1] <= bottom - 1)
        ).all())
        geometry = prediction.cropped.homography
        status = geometry.status if geometry is not None else "not_requested"
        qualifies = (
            status == "ok" and bool(visible.all()) and inside_content and convex
            and area >= config.min_area_ratio and count >= config.min_inliers
        )
        candidates.append({
            "region_xyxy": list(region), "status": status, "accepted": qualifies,
            "inliers": count, "inlier_score": score, "area_ratio": area,
            "convex": convex, "inside_content": inside_content,
        })
        if qualifies:
            accepted.append((count, score, region))
    if not accepted:
        raise ValueError(f"No Court region meets model support/geometry requirements: {candidates}")
    chosen = max(accepted, key=lambda item: (item[0], item[1]))
    return CourtRegionSelection(chosen[2], tuple(candidates))

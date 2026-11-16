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

from src.tasks.court_detection.geometry.paint_observations import PaintConfig
from src.tasks.court_detection.geometry.paint_refinement import (
    PaintResult,
    refine_paint,
)
from src.tasks.court_detection.geometry.quality import (
    DEFAULT_QUALITY_CONFIG,
    CourtQuality,
    CourtQualityConfig,
    assess_court_quality,
)
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
)

if TYPE_CHECKING:
    from src.tasks.court_detection.inference.predictor import CourtPredictor

Region = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CourtRegionSearchConfig(CourtQualityConfig):
    enabled: bool = False
    min_area_ratio: float = 0.01
    refinement_method: str = "model"
    paint: PaintConfig = PaintConfig()

    def __post_init__(self) -> None:
        CourtQualityConfig.__post_init__(self)
        if self.refinement_method not in {"model", "rgb_paint"}:
            raise ValueError("Unknown court refinement_method")
        if not isinstance(self.paint, PaintConfig):
            raise ValueError("Court paint configuration must be typed")
        if not np.isfinite(self.min_area_ratio) or not 0 < self.min_area_ratio < 1:
            raise ValueError("region_search.min_area_ratio must be finite and between 0 and 1")


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
    quality: CourtQuality
    paint: PaintResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quality, CourtQuality):
            raise ValueError("Region predictions require assessed court quality")

    def downstream_keypoints(self) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        geometry = self.cropped.homography
        if geometry is None:
            raise ValueError("Region geometry requires hybrid postprocessing")
        if geometry.status != "ok" or geometry.matrix is None or not self.quality.accepted:
            return np.zeros((14, 2), np.float32), np.zeros(14, dtype=bool)
        points = (np.asarray(geometry.projected, dtype=np.float32) + np.asarray(self.region[:2], dtype=np.float32)
                  if self.paint is None else np.asarray(self.paint.projected, dtype=np.float32))
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
        diagnostic["quality"] = self.quality.diagnostics()
        diagnostic["fit_source"] = "model_kp_line"
        if self.paint is not None:
            # The model's selected KP and residuals describe the initializer,
            # not the final RGB-only objective. Preserve both stages explicitly.
            diagnostic["model_geometry"] = {key: value for key, value in diagnostic.items() if key != "quality"}
            diagnostic["model_homography"] = diagnostic["homography_court_metres_to_image_pixels"]
            diagnostic["rgb_fitted_homography"] = None if self.paint.fitted_matrix is None else self.paint.fitted_matrix.tolist()
            diagnostic["homography_court_metres_to_image_pixels"] = None if self.paint.matrix is None else self.paint.matrix.tolist()
            diagnostic["fit_source"] = "rgb_paint_lines"
            diagnostic["selected_keypoints"] = []
            diagnostic["support_solves"] = []
            diagnostic["line_support"] = None
            diagnostic["candidate_count"] = int(self.paint.initial_matrix is not None)
            final_residuals = (np.full(14, np.nan) if self.paint.matrix is None else
                               np.linalg.norm(self.paint.projected - (raw.keypoints[:, 0].numpy() + offset), axis=1))
            diagnostic["residuals_px"] = [float(value) if np.isfinite(value) else None for value in final_residuals]
        if geometry is not None and geometry.status == "ok" and not self.quality.accepted:
            diagnostic["hybrid_status"] = "ok"
            diagnostic["status"] = "quality_rejected"
            diagnostic["rejected_homography"] = (diagnostic["homography_court_metres_to_image_pixels"] if self.paint is None
                                                 else diagnostic["rgb_fitted_homography"])
            diagnostic["homography_court_metres_to_image_pixels"] = None
        return diagnostic


def predict_court_region(
    predictor: CourtPredictor, image: NDArray[np.uint8], region: Region,
    *, quality_config: CourtQualityConfig = DEFAULT_QUALITY_CONFIG,
) -> CourtRegionPrediction:
    x0, y0, x1, y1 = region
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError(f"Court region {region} is outside image {image.shape[:2]}")
    prediction = predictor.predict(
        np.ascontiguousarray(image[y0:y1, x0:x1]),
        postprocess="hybrid", heads=("kp", "line"),
    )
    return assess_region_prediction(prediction, image, region, quality_config)


def assess_region_prediction(prediction: CourtPrediction, image: NDArray[np.uint8], region: Region,
                             config: CourtQualityConfig) -> CourtRegionPrediction:
    """Common live/frozen-raw boundary; RGB always uses the full source frame."""
    model_quality = assess_prediction_quality(prediction, config, image.shape[:2])
    paint = None
    quality = model_quality
    if isinstance(config, CourtRegionSearchConfig) and config.refinement_method == "rgb_paint":
        geometry = prediction.homography
        translation = np.eye(3)
        translation[:2, 2] = region[:2]
        initial = None if geometry is None or geometry.matrix is None else translation @ geometry.matrix
        paint = refine_paint(image, initial, config.paint)
        quality = CourtQuality(paint.quality.accepted, paint.quality.score, paint.quality.reasons,
            {"schema": "court_quality_v2", "method": "rgb_paint", "score_units": "dimensionless_paint_cost",
             "model_consistency": model_quality.diagnostics(), "rgb_paint": paint.quality.diagnostics()})
    return CourtRegionPrediction(prediction, region, (image.shape[0], image.shape[1]), quality, paint)


def assess_prediction_quality(
    prediction: CourtPrediction, config: CourtQualityConfig, evaluation_size_hw: tuple[int, int],
) -> CourtQuality:
    """One shared quality contract for full-frame and cropped inference."""
    raw, line = prediction.raw_heads["kp"], prediction.raw_heads["line"]
    if not isinstance(raw, CourtKeypointPrediction) or not isinstance(line, CourtLinePrediction):
        raise ValueError("Court quality requires typed KP and LINE evidence")
    geometry = prediction.homography
    return assess_court_quality(
        None if geometry is None else geometry.matrix, raw.keypoints[:, 0].numpy(),
        raw.valid[:, 0].numpy() & (geometry.selected if geometry is not None else False),
        line.probability.numpy(), image_size_hw=prediction.original_size_hw,
        evaluation_size_hw=evaluation_size_hw, config=config,
    )


@dataclass(frozen=True)
class CourtRegionSelection:
    region: Region
    candidates: tuple[dict[str, object], ...]


class CourtRegionUnavailable(ValueError):
    """Executed region search found no supported geometry; model errors differ."""

    def __init__(self, candidates: tuple[dict[str, object], ...]) -> None:
        self.candidates = candidates
        super().__init__(f"No Court region meets model support/geometry requirements: {list(candidates)}")


def select_court_region(
    predictor: CourtPredictor, image: NDArray[np.uint8], config: CourtRegionSearchConfig,
) -> CourtRegionSelection:
    """Select by visible-line quality and stability; fail if no candidate qualifies.

    This establishes a region once, not a stored homography or per-frame fallback.
    The caller reruns inference for every frame within the selected region.
    """
    left, top, right, bottom = image_content_region(image)
    height, width = image.shape[:2]
    candidates: list[dict[str, object]] = []
    accepted: list[tuple[float, Region]] = []
    for region in region_proposals(image):
        prediction = predict_court_region(predictor, image, region, quality_config=config)
        quality = prediction.quality
        if quality is None:
            raise ValueError("Region selection requires assessed court quality")
        geometry = prediction.cropped.homography
        status = geometry.status if geometry is not None else "not_requested"
        area, convex, inside_content = 0.0, False, False
        count, confidence_sum = 0, 0.0
        if geometry is not None and geometry.matrix is not None:
            raw = prediction.cropped.raw_heads["kp"]
            assert isinstance(raw, CourtKeypointPrediction)
            raw_inliers = raw.valid[:, 0].numpy() & (
                np.linalg.norm(geometry.projected - raw.keypoints[:, 0].numpy(), axis=1)
                <= config.line_tolerance_ratio * np.hypot(width, height)
            )
            count = int(raw_inliers.sum())
            confidence_sum = float(raw.scores[:, 0].numpy()[raw_inliers].sum())
            points = (geometry.projected + np.asarray(region[:2]) if prediction.paint is None else
                      prediction.paint.projected)
            polygon = points[[0, 1, 3, 2]].astype(np.float32)
            convex = bool(np.isfinite(polygon).all() and cv2.isContourConvex(polygon))
            if convex:
                content = np.array([[left, top], [right-1, top], [right-1, bottom-1], [left, bottom-1]], np.float32)
                intersection, _ = cv2.intersectConvexConvex(polygon, content)
                area = float(intersection) / (width * height)
            inside_content = bool(((points[:, 0] >= left) & (points[:, 0] <= right - 1)
                & (points[:, 1] >= top) & (points[:, 1] <= bottom - 1)).all())
        reasons = list(quality.reasons)
        if not convex:
            reasons.append("invalid_court_polygon")
        if area < config.min_area_ratio:
            reasons.append("insufficient_visible_area")
        qualifies = status == "ok" and quality.accepted and not reasons
        diagnostic = prediction.geometry_diagnostics()
        model_support_count = 0 if geometry is None else int(geometry.selected.sum())
        rgb_fit = None if prediction.paint is None else prediction.paint.quality.metrics.get("fine_fit")
        rgb_lines = rgb_fit.get("fitted_line_names", []) if isinstance(rgb_fit, dict) else []
        candidates.append({
            "region_xyxy": list(region), "status": diagnostic["status"], "accepted": qualifies,
            "inliers": count, "inlier_score": confidence_sum, "area_ratio": area,
            "fit_support_count": model_support_count if prediction.paint is None else 0,
            "model_fit_support_count": model_support_count, "rgb_fit_line_count": len(rgb_lines),
            "convex": convex, "inside_content": inside_content,
            "quality": quality.diagnostics(), "rejection_reasons": list(dict.fromkeys(reasons)),
            "geometry": diagnostic,
        })
        if qualifies:
            assert quality.score is not None
            accepted.append((quality.score, region))
    if not accepted:
        raise CourtRegionUnavailable(tuple(candidates))
    # Stable input order resolves equal quality. Raw confidence is not a
    # tiebreaker: channels collapsed onto one peak cannot improve this ranking.
    chosen = min(accepted, key=lambda item: item[0])
    return CourtRegionSelection(chosen[1], tuple(candidates))

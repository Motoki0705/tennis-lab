"""Raw head evidence and accepted geometry are separate inference contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.hybrid_homography import HybridHomographyResult
from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedPrediction,
    CourtKeypointPrediction,
)


@dataclass(frozen=True)
class CourtPrediction:
    raw_heads: Mapping[CourtTargetKind, CourtDecodedPrediction]
    original_size_hw: tuple[int, int]
    native_size_hw: tuple[int, int]
    homography: HybridHomographyResult | None
    keypoint_schema: str | None = None

    def downstream_keypoints(self) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Fitted14 in pixels and in-image validity, never the <=8 fit subset.

        Failed geometry has finite zero placeholders plus all-false validity.
        Successful off-image projections retain their unclipped coordinates.
        """
        if self.homography is None:
            raise ValueError(
                "Downstream geometry requires explicit hybrid postprocessing"
            )
        result = self.homography
        if result.status != "ok" or result.matrix is None:
            return np.zeros((14, 2), dtype=np.float32), np.zeros(14, dtype=bool)
        points = np.asarray(result.projected, dtype=np.float32)
        height, width = self.original_size_hw
        valid = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0)
            & (points[:, 0] <= width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= height - 1)
        )
        return points, valid

    def geometry_diagnostics(self) -> dict[str, object]:
        """Small JSON-safe per-frame provenance; dense heatmaps stay out of archives."""
        result = self.homography
        if result is None:
            return {"status": "not_requested"}
        raw = self.raw_heads["kp"]
        assert isinstance(raw, CourtKeypointPrediction)
        return {
            "status": result.status,
            "keypoint_schema": self.keypoint_schema,
            "homography_frame": "checkpoint_KP14_template_metres",
            "homography_court_metres_to_image_pixels": None
            if result.matrix is None
            else result.matrix.tolist(),
            "selected_keypoints": np.flatnonzero(result.selected).tolist(),
            "raw_keypoints_px": [
                [float(v) if np.isfinite(v) else None for v in xy]
                for xy in raw.keypoints[:, 0].numpy()
            ],
            "raw_scores": raw.scores[:, 0].tolist(),
            "raw_valid": raw.valid[:, 0].tolist(),
            "residuals_px": [
                float(x) if np.isfinite(x) else None for x in result.residuals_px
            ],
            "candidate_count": result.candidate_count,
            "line_support": None if result.best is None else dict(result.best.line),
            "original_size_hw": list(self.original_size_hw),
        }

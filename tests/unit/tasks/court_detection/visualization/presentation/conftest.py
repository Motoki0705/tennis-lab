"""Accepted synthetic geometry with distinct baseline, seeds and final fit."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.court_detection.geometry.confidence_homography import (
    ConfidenceHomographyResult,
    ProsacOrientation,
)
from src.tasks.court_detection.geometry.homography import court_template_xy
from src.tasks.court_detection.geometry.hybrid_homography import (
    HybridHomographyResult,
    HybridHypothesis,
)
from src.tasks.court_detection.geometry.line_evidence import LineDiagnostics
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
)


@pytest.fixture
def hybrid_prediction() -> tuple[CourtPrediction, NDArray[np.uint8]]:
    matrix = np.array([[10, 0, 160], [0, 6, 120], [0, 0, 1]], dtype=np.float64)
    projected = cv2.perspectiveTransform(
        court_template_xy()[None].astype(np.float64), matrix
    )[0]
    selected = np.arange(14) < 8
    diagnostics: LineDiagnostics = {
        "forward_support": 1.0,
        "reverse_support": 1.0,
        "supported_segments": 9.0,
        "segment_support": [1.0] * 9,
        "visible_segments": 9.0,
        "forward_rms_px": 0.0,
        "reverse_rms_px": 0.0,
    }
    best = HybridHypothesis(matrix, projected, selected, 0.01, diagnostics)
    right, left, baseline_matrix = (matrix.copy() for _ in range(3))
    right[0, 2] += 8
    left[0, 2] -= 8
    baseline_matrix[:2, 2] += [25, 8]
    seeds = (
        replace(best, matrix=right, projected=projected + [8, 0], score=0.2),
        replace(best, matrix=left, projected=projected - [8, 0], score=0.8),
    )
    inliers = np.arange(14) < 4
    baseline = ConfidenceHomographyResult(
        baseline_matrix,
        projected + [25, 8],
        np.arange(14),
        inliers,
        inliers,
        np.zeros(14),
        "ok",
        prosac_orientation=ProsacOrientation.IDENTITY,
    )
    result = HybridHomographyResult(
        matrix=matrix,
        projected=projected,
        selected=selected,
        residuals_px=np.zeros(14),
        status="ok",
        kp_only=baseline,
        line_selection=seeds[0],
        seeds=seeds,
        best=best,
        candidate_count=9,
        refined_count=2,
        alternative_score_gap=None,
    )
    valid = torch.ones((14, 1), dtype=torch.bool)
    valid[-1] = False
    heatmaps = torch.full((14, 60, 80), -8.0, dtype=torch.float32)
    heatmaps[0, 5, 5] = 8.0
    heatmaps[1, 6, 5] = 4.0
    probability = torch.zeros((60, 80), dtype=torch.float32)
    probability[10] = 1.0
    kp = CourtKeypointPrediction(
        torch.from_numpy(projected[:, None].copy()),
        torch.linspace(0.1, 0.95, 14)[:, None],
        valid,
        heatmaps,
    )
    line = CourtLinePrediction(probability, torch.zeros_like(probability))
    return CourtPrediction(
        {"kp": kp, "line": line}, (240, 320), (60, 80), result
    ), np.full((240, 320, 3), 32, dtype=np.uint8)

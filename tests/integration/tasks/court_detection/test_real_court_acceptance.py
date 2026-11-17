"""Cached real model heads -> hybrid fit -> quality gate -> downstream masks."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.court_detection.geometry.homography import (
    COURT_HOMOGRAPHY_EDGES,
    court_template_xy,
)
from src.tasks.court_detection.geometry.hybrid_homography import (
    HybridHomographyConfig,
    estimate_hybrid_homography,
)
from src.tasks.court_detection.geometry.quality import CourtQualityConfig
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.inference.regions import (
    CourtRegionPrediction,
    assess_prediction_quality,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
)

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parent / "fixtures" / "acceptance"


@pytest.mark.parametrize("name,accepted", [("supported", True), ("partial_unstable", False)])
def test_real_heads_keep_quality_failure_separate_from_solver_success(name, accepted):
    with np.load(FIXTURES / f"{name}.npz") as data:
        xy, scores, valid = data["raw_xy"], data["scores"], data["valid"]
        line = data["line_probability_u16"].astype(np.float32) / 65535
        size, native = tuple(data["image_size_hw"]), tuple(data["native_size_hw"])
    meta = json.loads((FIXTURES / f"{name}.json").read_text())
    observed = xy.astype(float)
    observed[~valid] = np.nan
    before = observed.copy()
    geometry = estimate_hybrid_homography(
        court_template_xy(), observed, scores, line,
        image_size_hw=size, edges=np.array(COURT_HOMOGRAPHY_EDGES),
        config=HybridHomographyConfig(support_method="candidate_subsets"),
    )
    np.testing.assert_array_equal(observed, before)
    assert geometry.status == "ok"  # A converged H alone cannot certify acceptance.
    prediction = CourtPrediction({
        "kp": CourtKeypointPrediction(torch.from_numpy(xy[:, None]), torch.from_numpy(scores[:, None]),
                                      torch.from_numpy(valid[:, None]), torch.empty((14, 0, 0))),
        "line": CourtLinePrediction(torch.from_numpy(line), torch.logit(torch.from_numpy(line).clamp(1e-6, 1-1e-6))),
    }, size, native, geometry)
    quality = assess_prediction_quality(prediction, CourtQualityConfig(), tuple(meta["source_size_hw"]))
    result = CourtRegionPrediction(prediction, tuple(meta["region"]), tuple(meta["source_size_hw"]), quality)
    points, visible = result.downstream_keypoints()
    diagnostic = result.geometry_diagnostics()
    assert quality.accepted is accepted
    quality_diagnostic = diagnostic["quality"]
    assert isinstance(quality_diagnostic, dict)
    assert quality_diagnostic["schema"] == "court_quality_v1"
    if accepted:
        assert visible.any() and np.isfinite(points).all()
        np.testing.assert_allclose(points, geometry.projected + meta["region"][:2], atol=2e-4)
        assert diagnostic["status"] == "ok"
    else:
        assert "unstable_projection" in quality.reasons
        assert geometry.observation_diagnostics is not None
        assert geometry.observation_diagnostics["boundary_censored_indices"] == [2, 5]
        assert not geometry.selected[[2, 5]].any()
        assert not visible.any() and not points.any()
        assert diagnostic["status"] == "quality_rejected"
        assert diagnostic["homography_court_metres_to_image_pixels"] is None
        assert diagnostic["rejected_homography"] is not None

"""Image-derived search, coordinate mapping and explicit rejection contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from src.tasks.court_detection.inference.regions import (
    CourtRegionPrediction,
    CourtRegionSearchConfig,
    image_content_region,
    predict_court_region,
    region_proposals,
    select_court_region,
)
from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
    geometry_prediction,
)


def test_proposals_use_only_image_extent_and_exact_black_padding() -> None:
    image: np.ndarray = np.ones((108, 192, 3), np.uint8)
    image[:, :24] = image[:, 168:] = 0
    image[25:30, 50:60] = 0
    assert image_content_region(image) == (24, 0, 168, 108)
    proposals = region_proposals(image)
    assert len(proposals) == len(set(proposals)) == 13
    assert (60, 0, 168, 54) in proposals
    # Nonzero dark pixels are content, not a heuristic crop threshold.
    image[0, 0] = 1
    assert image_content_region(image) == (0, 0, 168, 108)
    with pytest.raises(ValueError, match="entirely black"):
        region_proposals(np.zeros_like(image))


def test_crop_geometry_maps_to_full_image_without_relabeling_rasters() -> None:
    cropped = geometry_prediction()
    result = CourtRegionPrediction(cropped, (5, 4, 17, 12), (20, 30))
    points, valid = result.downstream_keypoints()
    np.testing.assert_allclose(points[:2], [[3, 7], [20, 7]])
    assert valid.all()  # The court may extend beyond the input crop.
    assert result.cropped.native_size_hw == (4, 6)
    diagnostic = result.geometry_diagnostics()
    np.testing.assert_allclose(diagnostic["homography_court_metres_to_image_pixels"], [[1, 0, 5], [0, 1, 4], [0, 0, 1]])
    np.testing.assert_allclose(diagnostic["raw_keypoints_px"], np.tile([11, 10], (14, 1)))
    assert diagnostic["postprocess_image_size_hw"] == [8, 12]
    assert diagnostic["original_size_hw"] == [20, 30]


def test_failed_crop_has_zero_placeholders_without_offset_or_fallback() -> None:
    result = CourtRegionPrediction(
        geometry_prediction(status="joint_optimization_failed"), (5, 4, 17, 12), (20, 30)
    )
    points, valid = result.downstream_keypoints()
    assert not points.any() and not valid.any()
    assert result.geometry_diagnostics()["homography_court_metres_to_image_pixels"] is None


def test_crop_outside_image_fails_before_inference() -> None:
    with pytest.raises(ValueError, match="outside image"):
        predict_court_region(cast(Any, object()), np.ones((10, 10, 3), np.uint8), (0, 0, 11, 10))


@pytest.mark.parametrize("failure", ["geometry", "support", "outside", "degenerate", None])
def test_search_requires_geometry_and_raw_model_support(monkeypatch, failure) -> None:
    import src.tasks.court_detection.inference.regions as module

    points = np.tile([20.0, 20.0], (14, 1)).astype(np.float32)
    points[:4] = [[10, 10], [50, 10], [10, 50], [50, 50]]
    if failure == "outside":
        points[3] = [150, 50]
    elif failure == "degenerate":
        points[:] = 20
    cropped = geometry_prediction(status="joint_optimization_failed" if failure == "geometry" else "ok")
    assert cropped.homography is not None
    if failure != "geometry":
        cropped = replace(cropped, homography=replace(cropped.homography, projected=points))
    raw = cropped.raw_heads["kp"]
    raw_points = points.copy()
    if failure == "support":
        raw_points[7:] += 15
    cropped = replace(cropped, raw_heads={"kp": replace(raw, keypoints=torch.tensor(raw_points[:, None]))})
    monkeypatch.setattr(module, "region_proposals", lambda _: ((0, 0, 100, 100),))
    monkeypatch.setattr(module, "predict_court_region", lambda *_: CourtRegionPrediction(cropped, (0, 0, 100, 100), (100, 100)))
    if failure is not None:
        with pytest.raises(ValueError, match="No Court region meets"):
            select_court_region(cast(Any, object()), np.ones((100, 100, 3), np.uint8), CourtRegionSearchConfig())
    else:
        selected = select_court_region(cast(Any, object()), np.ones((100, 100, 3), np.uint8), CourtRegionSearchConfig())
        assert selected.region == (0, 0, 100, 100)
        assert selected.candidates[0]["inliers"] == 14

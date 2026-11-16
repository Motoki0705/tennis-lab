"""Image-derived search, coordinate mapping and explicit rejection contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from src.tasks.court_detection.geometry.quality import CourtQuality
from src.tasks.court_detection.inference.regions import (
    CourtRegionPrediction,
    CourtRegionSearchConfig,
    assess_region_prediction,
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
    result = CourtRegionPrediction(cropped, (5, 4, 17, 12), (20, 30), CourtQuality(True, .1, (), {}))
    points, valid = result.downstream_keypoints()
    np.testing.assert_allclose(points[:2], [[3, 7], [20, 7]])
    assert valid.all()  # The court may extend beyond the input crop.
    assert result.cropped.native_size_hw == (4, 6)
    diagnostic = result.geometry_diagnostics()
    np.testing.assert_allclose(diagnostic["homography_court_metres_to_image_pixels"], [[1, 0, 5], [0, 1, 4], [0, 0, 1]])
    np.testing.assert_allclose(diagnostic["raw_keypoints_px"], np.tile([11, 10], (14, 1)))
    assert diagnostic["postprocess_image_size_hw"] == [8, 12]
    assert diagnostic["original_size_hw"] == [20, 30]
    assert diagnostic["kp_only_prosac_orientation"] == "identity"


def test_failed_crop_has_zero_placeholders_without_offset_or_fallback() -> None:
    result = CourtRegionPrediction(
        geometry_prediction(status="joint_optimization_failed"), (5, 4, 17, 12), (20, 30),
        CourtQuality(False, None, ("no_homography",), {}),
    )
    points, valid = result.downstream_keypoints()
    assert not points.any() and not valid.any()
    assert result.geometry_diagnostics()["homography_court_metres_to_image_pixels"] is None


def test_rgb_rejection_keeps_model_evidence_but_never_publishes_its_h(monkeypatch):
    import src.tasks.court_detection.inference.regions as module

    monkeypatch.setattr(module, "assess_prediction_quality", lambda *_: CourtQuality(True, .1, (), {}))
    result = assess_region_prediction(geometry_prediction(), np.full((80, 120, 3), 80, np.uint8),
                                     (5, 4, 17, 12), CourtRegionSearchConfig(refinement_method="rgb_paint"))
    points, valid = result.downstream_keypoints()
    assert not points.any() and not valid.any()
    diagnostic = result.geometry_diagnostics()
    assert diagnostic["model_homography"] is not None
    assert diagnostic["homography_court_metres_to_image_pixels"] is None
    quality = cast(dict[str, Any], diagnostic["quality"])
    assert quality["schema"] == "court_quality_v2"
    assert quality["model_consistency"]["accepted"] is True
    assert quality["rgb_paint"]["accepted"] is False
    model = cast(dict[str, Any], diagnostic["model_geometry"])
    assert model["selected_keypoints"] == [3, 4, 8, 9]
    assert diagnostic["selected_keypoints"] == []
    assert diagnostic["residuals_px"] == [None] * 14

    monkeypatch.setattr(module, "region_proposals", lambda _: ((5, 4, 17, 12),))
    monkeypatch.setattr(module, "predict_court_region", lambda *_, **__: result)
    with pytest.raises(module.CourtRegionUnavailable) as error:
        select_court_region(cast(Any, object()), np.full((80, 120, 3), 80, np.uint8), CourtRegionSearchConfig())
    candidate = error.value.candidates[0]
    assert candidate["status"] == "quality_rejected"
    assert candidate["model_fit_support_count"] == 4
    assert candidate["fit_support_count"] == candidate["rgb_fit_line_count"] == 0
    assert cast(dict[str, Any], candidate["geometry"])["model_homography"] is not None


def test_rgb_result_uses_full_frame_pixels_without_applying_crop_offset_twice(monkeypatch):
    import src.tasks.court_detection.inference.regions as module
    from tests.unit.tasks.court_detection.geometry.test_paint_refinement import (
        painted_court,
    )

    image, matrix, truth = painted_court()
    cropped = geometry_prediction()
    assert cropped.homography is not None
    translation = np.array([[1., 0, -70.], [0, 1., -40.], [0, 0, 1.]])
    cropped = replace(cropped, homography=replace(cropped.homography, matrix=translation @ matrix, projected=truth-[70, 40]))
    # Model self-consistency can reject a correct partial observation. RGB has
    # its own measured support and acceptance; the two scores are kept separate.
    monkeypatch.setattr(module, "assess_prediction_quality", lambda *_: CourtQuality(False, 3., ("unstable_projection",), {}))
    result = assess_region_prediction(cropped, image, (70, 40, 600, 470), CourtRegionSearchConfig(refinement_method="rgb_paint"))
    assert result.quality.accepted
    points, valid = result.downstream_keypoints()
    assert valid.all()
    assert np.max(np.linalg.norm(points-truth, axis=1)) < 1.5
    diagnostic = result.geometry_diagnostics()
    np.testing.assert_allclose(diagnostic["model_homography"], matrix)
    assert diagnostic["homography_court_metres_to_image_pixels"] is not None
    assert diagnostic["fit_source"] == "rgb_paint_lines"
    assert diagnostic["selected_keypoints"] == []  # RGB H has no KP equality terms.
    model = cast(dict[str, Any], diagnostic["model_geometry"])
    assert model["selected_keypoints"] == [3, 4, 8, 9]
    assert result.paint is not None
    np.testing.assert_allclose(diagnostic["residuals_px"], np.linalg.norm(result.paint.projected-np.tile([76, 46], (14, 1)), axis=1))


def test_crop_outside_image_fails_before_inference() -> None:
    with pytest.raises(ValueError, match="outside image"):
        predict_court_region(cast(Any, object()), np.ones((10, 10, 3), np.uint8), (0, 0, 11, 10))


def test_unassessed_region_cannot_bypass_quality_gate() -> None:
    with pytest.raises(ValueError, match="require assessed court quality"):
        CourtRegionPrediction(geometry_prediction(), (0, 0, 12, 8), (8, 12), cast(Any, None))


@pytest.mark.parametrize("failure", ["geometry", "support", "outside", "degenerate", None])
def test_search_requires_assessed_geometry_and_visible_area(monkeypatch, failure) -> None:
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
        raw_points[3:] += 15
    cropped = replace(cropped, raw_heads={"kp": replace(raw, keypoints=torch.tensor(raw_points[:, None]))})
    monkeypatch.setattr(module, "region_proposals", lambda _: ((0, 0, 100, 100),))
    quality = CourtQuality(failure != "support", 0.5, ("insufficient_independent_keypoints",) if failure == "support" else (),
                           {"supported_keypoints": 3 if failure == "support" else 14})
    monkeypatch.setattr(module, "predict_court_region", lambda *_, **__: CourtRegionPrediction(cropped, (0, 0, 100, 100), (100, 100), quality))
    if failure not in (None, "outside"):
        with pytest.raises(module.CourtRegionUnavailable, match="No Court region meets") as error:
            select_court_region(cast(Any, object()), np.ones((100, 100, 3), np.uint8), CourtRegionSearchConfig())
        assert len(error.value.candidates) == 1
        assert error.value.candidates[0]["accepted"] is False
    else:
        selected = select_court_region(cast(Any, object()), np.ones((100, 100, 3), np.uint8), CourtRegionSearchConfig())
        assert selected.region == (0, 0, 100, 100)
        assert selected.candidates[0]["inliers"] == 14
        assert selected.candidates[0]["inside_content"] == (failure != "outside")


def test_quality_cost_ranks_regions_instead_of_raw_support_count(monkeypatch) -> None:
    import src.tasks.court_detection.inference.regions as module

    base = geometry_prediction()
    assert base.homography is not None
    points = np.tile([30., 30.], (14, 1))
    points[:4] = [[10, 10], [60, 10], [10, 60], [60, 60]]
    base = replace(base, homography=replace(base.homography, projected=points))
    regions = ((0, 0, 100, 100), (5, 0, 100, 100))
    monkeypatch.setattr(module, "region_proposals", lambda _: regions)

    def predict(_predictor, _image, region, **_):
        score, count = (.8, 14) if region == regions[0] else (.2, 4)
        return CourtRegionPrediction(base, region, (100, 100), CourtQuality(True, score, (), {"supported_keypoints": count}))

    monkeypatch.setattr(module, "predict_court_region", predict)
    result = select_court_region(cast(Any, object()), np.ones((100, 100, 3), np.uint8), CourtRegionSearchConfig())
    assert result.region == regions[1]
    quality = result.candidates[1]["quality"]
    assert isinstance(quality, dict)
    assert quality["score"] == .2

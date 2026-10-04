"""Fractional teacher values must survive preview and web-raster rendering."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.court_detection.model_io.contracts import (
    CourtLinePrediction,
    CourtSegmentationPrediction,
)
from src.tasks.court_detection.visualization.inference.metrics import (
    categorical_metrics,
    line_metrics,
)
from src.tasks.court_detection.visualization.rendering.common import (
    COURT_SEG_PALETTE_RGB,
)
from src.tasks.court_detection.visualization.rendering.target_preview import (
    render_line_target,
    render_segmentation_target,
    render_target_only,
    summarize_targets,
)
from src.tasks.court_detection.visualization.review.rasters import categorical_raster


def test_fractional_line_alpha_is_not_thresholded() -> None:
    line = torch.tensor([[[0.0, 0.25, 1.0]]])
    rgb: NDArray[np.uint8] = np.zeros((1, 3, 3), dtype=np.uint8)
    rendered = render_line_target(rgb, line, alpha=1)
    assert rendered[0, :, 0].tolist() == [0, 64, 255]
    assert render_target_only("line", line)[0, :, 0].tolist() == [0, 64, 255]
    stats = summarize_targets({"line": line}, sigma_ratio=None)["line"]
    assert isinstance(stats, dict)
    assert stats["coverage_sum"] == 1.25 and stats["fractional_pixels"] == 1


def test_categorical_coverage_mixes_colors_instead_of_numeric_class_ids() -> None:
    target = torch.tensor([[[0.5]], [[0.25]], [[0.25]]])
    rgb: NDArray[np.uint8] = np.zeros((1, 1, 3), dtype=np.uint8)
    rendered = render_segmentation_target(rgb, target, alpha=1, max_label=2)
    expected = np.rint(
        (np.array(COURT_SEG_PALETTE_RGB[1]) + np.array(COURT_SEG_PALETTE_RGB[2])) * 0.25
    ).astype(np.uint8)
    np.testing.assert_array_equal(rendered[0, 0], expected)
    rgba = categorical_raster(target.numpy(), palette=COURT_SEG_PALETTE_RGB[:3])
    assert rgba[0, 0, 3] == 100
    expected_color = np.rint(
        (np.array(COURT_SEG_PALETTE_RGB[1]) + np.array(COURT_SEG_PALETTE_RGB[2])) * 0.5
    ).astype(np.uint8)
    np.testing.assert_array_equal(rgba[0, 0, :3], expected_color)


@pytest.mark.parametrize("value", [-0.1, float("nan"), 1.1])
def test_invalid_line_coverage_cannot_be_hidden_by_the_renderer(value: float) -> None:
    with pytest.raises(ValueError, match="coverage"):
        render_line_target(
            np.zeros((1, 1, 3), np.uint8), torch.tensor([[[value]]]), alpha=1
        )


def test_metrics_weight_fractional_ground_truth_without_thresholding_it() -> None:
    seg = CourtSegmentationPrediction(
        torch.ones(1, 1, dtype=torch.long), torch.zeros(2, 1, 1)
    )
    measured = categorical_metrics(
        seg, ground_truth=np.array([[[0.75]], [[0.25]]], dtype=np.float32), prefix="seg"
    )
    assert measured["seg_pixel_accuracy"] == pytest.approx(0.25)
    assert measured["seg_mean_iou"] == pytest.approx(0.25)
    line = CourtLinePrediction(torch.ones(1, 1), torch.zeros(1, 1))
    measured = line_metrics(
        line, ground_truth=np.array([[0.25]], np.float32), threshold=0.5
    )
    assert measured["line_iou"] == pytest.approx(0.25)
    assert measured["line_dice"] == pytest.approx(0.4)


def test_previous_hard_checkpoint_can_be_displayed_but_not_scored_as_coverage() -> None:
    from src.tasks.court_detection.target_schemas import SEGMENTATION_TARGET_SCHEMA_HARD
    from src.tasks.court_detection.visualization.inference.service import (
        DetectionService,
        _compatibility_reason,
    )
    from src.tasks.court_detection.visualization.review.datasets import GroundTruthMasks
    from tests.unit.tasks.court_detection.data.target_generation.test_online import _raw
    from tests.unit.tasks.court_detection.visualization.inference.test_inference_service import (
        _heads,
        _info,
        _layers,
    )

    info = _info(_heads(seg_schema=SEGMENTATION_TARGET_SCHEMA_HARD))
    assert _compatibility_reason(info, _layers()) is None
    service = object.__new__(DetectionService)
    seg = CourtSegmentationPrediction(
        torch.ones(1, 1, dtype=torch.long), torch.zeros(7, 1, 1)
    )
    truth: NDArray[np.float32] = np.zeros((7, 1, 1), np.float32)
    truth[0] = 0.75
    truth[1] = 0.25
    payload, metrics, warnings = service._render_prediction(
        {"seg": seg},
        masks=GroundTruthMasks(truth, None, None),
        raw=_raw(),
        threshold=0.5,
        channel_names=(),
        dense_schemas=info.dense_schemas(),
    )
    rasters = payload["rasters"]
    assert isinstance(rasters, list) and len(rasters) == 1
    assert "seg_mean_iou" not in metrics
    assert any("GT採点から除外" in warning for warning in warnings)


def test_public_preview_target_only_panel_keeps_fractional_values() -> None:
    from omegaconf import OmegaConf

    from src.tasks.court_detection.scripts.preview_augmentation import _target_panels

    cfg = OmegaConf.create(
        {"preview": {"draw": {"mask_alpha": 0.5, "heatmap_alpha": 0.5}}}
    )
    sample = {
        "image": torch.zeros(3, 1, 3),
        "targets": {"line": torch.tensor([[[0.0, 0.25, 1.0]]])},
    }
    panels, titles = _target_panels(
        sample, target_kinds=("line",), title="target", cfg=cfg, target_only=True
    )
    assert panels[1][0, :, 0].tolist() == [0, 64, 255]
    assert titles == ["target: RGB", "target: LINE coverage"]

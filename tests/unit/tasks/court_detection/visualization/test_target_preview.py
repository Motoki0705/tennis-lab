"""Tests for exact target rendering and quantitative preview metadata."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.court_detection.visualization.rendering.target_preview import (
    gaussian_pixel_geometry,
    render_heatmap_target,
    render_line_target,
    render_segmentation_target,
    summarize_targets,
)
from src.tasks.court_detection.visualization.youtube_target_preview import (
    ground_kp14_ready,
)


def test_gaussian_geometry_reports_sigma_and_fwhm_in_pixels() -> None:
    geometry = gaussian_pixel_geometry(0.01, height=256, width=256)

    assert geometry["sigma_px"] == pytest.approx(0.01 * math.hypot(256, 256))
    assert geometry["fwhm_diameter_px"] == pytest.approx(
        2.0 * math.sqrt(2.0 * math.log(2.0)) * geometry["sigma_px"]
    )


def test_renderers_preserve_rgb_shape_and_expose_each_target() -> None:
    rgb: NDArray[np.uint8] = np.full((8, 10, 3), 64, dtype=np.uint8)
    heatmaps = torch.zeros(2, 8, 10)
    heatmaps[0, 3, 4] = 1.0
    seg = torch.zeros(8, 10, dtype=torch.long)
    seg[2:6, 3:8] = 2
    line = torch.zeros(1, 8, 10)
    line[:, 4, :] = 1.0

    kp_render = render_heatmap_target(rgb, heatmaps, alpha=0.5)
    seg_render = render_segmentation_target(rgb, seg, alpha=0.5)
    line_render = render_line_target(rgb, line, alpha=0.5)

    assert kp_render.shape == seg_render.shape == line_render.shape == rgb.shape
    assert not np.array_equal(kp_render, rgb)
    assert not np.array_equal(seg_render[3, 4], rgb[3, 4])
    assert not np.array_equal(line_render[4, 4], rgb[4, 4])
    np.testing.assert_array_equal(seg_render[0, 0], rgb[0, 0])
    np.testing.assert_array_equal(line_render[0, 0], rgb[0, 0])


def test_summary_records_exact_shapes_and_foreground_counts() -> None:
    heatmap = torch.zeros(2, 8, 10)
    heatmap[:, 2:4, 3:5] = 0.75
    visible = torch.tensor([[True], [False]])
    seg = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    line = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    summary = summarize_targets(
        {
            "kp": {"heatmap": heatmap, "point_visible": visible},
            "seg": seg,
            "line": line,
        },
        sigma_ratio=0.01,
    )

    assert summary["kp"]["shape"] == [2, 8, 10]  # type: ignore[index]
    assert summary["kp"]["visible_points"] == 1  # type: ignore[index]
    assert summary["kp"]["pixels_ge_0_5"] == 8  # type: ignore[index]
    assert summary["seg"]["class_pixel_counts"] == {  # type: ignore[index]
        "0": 1,
        "1": 1,
        "2": 2,
    }
    assert summary["line"]["foreground_pixels"] == 2  # type: ignore[index]


def test_youtube_readiness_requires_all_finite_visible_ground_points() -> None:
    item = {
        "keypoints": [
            {"x": float(index), "y": float(index + 1), "visibility": 1}
            for index in range(14)
        ]
    }

    assert ground_kp14_ready(item)
    item["keypoints"][3]["visibility"] = 3
    assert not ground_kp14_ready(item)

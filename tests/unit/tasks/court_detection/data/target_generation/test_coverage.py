"""Regression tests for subpixel strokes and fractional categorical boundaries."""

from dataclasses import replace

import cv2
import numpy as np
import torch
from PIL import Image

from src.tasks.court_detection.data.target_generation.line import (
    _metric_lines,
    metric_line_quad,
)
from src.tasks.court_detection.data.target_generation.online import (
    generate_online_targets,
)
from src.tasks.court_detection.data.target_generation.rasterization import (
    CourtPlaneRasterizer,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
    SEMANTIC_LINE_TARGET_SCHEMA,
)
from tests.unit.tasks.court_detection.data.target_generation.test_online import _raw
from tests.unit.tasks.court_detection.data.target_generation.test_rasterization import (
    _projected_instance,
)


def test_subpixel_lines_survive_integer_polygon_collapse() -> None:
    matrix = np.array([[4.0, 0.0, 64.0], [0.0, -4.0, 64.0], [0.0, 0.0, 1.0]])
    instance = _projected_instance(matrix, width=128, height=128)
    raw = _raw()
    assert raw.keypoint_channels is not None
    raw = replace(
        raw,
        image=Image.new("RGB", (128, 128)),
        court_instances=(instance,),
        keypoint_channels=replace(
            raw.keypoint_channels, points_xy=instance.points_xy[:, None]
        ),
    )
    projector = CourtPlaneRasterizer.from_instance(instance, width=128, height=128)
    assert projector is not None
    segments = _metric_lines(line_width_metres=0.075, baseline_width_metres=0.15)
    assert (
        sum(
            projector.project_polygon(metric_line_quad(line)) is None
            for line in segments
        )
        >= 5
    )
    line = generate_online_targets(raw, {"line": LINE_TARGET_SCHEMA})["line"][0]
    assert bool(((line > 0) & (line < 1)).any())
    assert (
        cv2.connectedComponents((line.numpy() > 0).astype(np.uint8), connectivity=8)[0]
        == 2
    )
    # Probe every segment, including the sidelines and centre line lost by rounding.
    for segment in segments:
        points = np.linspace(segment.start, segment.end, 100)
        projected = np.c_[points, np.ones(len(points))] @ matrix.T
        for x, y in np.rint(projected[:, :2]).astype(int):
            assert line[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].sum() > 0


def test_class_coverage_is_normalized_and_line_matches_semantic_foreground() -> None:
    targets = generate_online_targets(
        _raw(),
        {
            "seg": SEGMENTATION_TARGET_SCHEMA,
            "line": LINE_TARGET_SCHEMA,
            "semantic_line": SEMANTIC_LINE_TARGET_SCHEMA,
        },
    )
    for kind, channels in (("seg", 7), ("semantic_line", 12)):
        value = targets[kind]
        assert value.shape == (channels, 256, 256)
        assert value.dtype == torch.float32
        torch.testing.assert_close(value.sum(0), torch.ones(256, 256), rtol=0, atol=0)
        assert bool(((value > 0) & (value < 1)).any())
    torch.testing.assert_close(
        targets["line"], 1 - targets["semantic_line"][:1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        targets["line"],
        generate_online_targets(_raw(), {"line": LINE_TARGET_SCHEMA})["line"],
        rtol=0,
        atol=0,
    )


def test_coverage_support_and_padding_are_background_not_zero_distributions() -> None:
    targets = generate_online_targets(
        _raw(),
        {
            "seg": SEGMENTATION_TARGET_SCHEMA,
            "line": LINE_TARGET_SCHEMA,
            "semantic_line": SEMANTIC_LINE_TARGET_SCHEMA,
        },
        source_to_output=torch.tensor(
            [[1.0, 0.0, 40.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        ),
        output_size_hw=(320, 320),
        content_size_hw=(286, 296),
    )
    for kind in ("seg", "semantic_line"):
        value = targets[kind]
        assert value[0, :30].eq(1).all()
        assert value[0, 286:].eq(1).all()
        assert not value[1:, :30].any()
        assert not value[1:, 286:].any()
        torch.testing.assert_close(value.sum(0), torch.ones(320, 320), rtol=0, atol=0)
    assert not targets["line"][:, :30].any()
    assert not targets["line"][:, 286:].any()


def test_fractional_translation_changes_boundary_coverage() -> None:
    schema = {"seg": SEGMENTATION_TARGET_SCHEMA}
    first = generate_online_targets(_raw(), schema)["seg"]
    shifted = generate_online_targets(
        _raw(),
        schema,
        source_to_output=torch.tensor(
            [[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        ),
    )["seg"]
    assert not torch.equal(first, shifted)
    # A subpixel shift redistributes coverage across neighbours without a hard jump.
    assert float((first - shifted).abs().max()) <= 0.5


def test_tiled_histogram_preserves_each_pixel_class_mass_including_final_tile() -> None:
    from src.tasks.court_detection.data.target_generation.coverage import _coverage

    labels = np.random.default_rng(42).integers(0, 7, (35 * 8, 13 * 8), dtype=np.uint8)
    actual = _coverage(labels, channels=7, height=35, width=13).numpy()
    expected = np.stack(
        [
            (labels == kind).reshape(35, 8, 13, 8).mean((1, 3), dtype=np.float32)
            for kind in range(7)
        ]
    )
    np.testing.assert_array_equal(actual, expected)

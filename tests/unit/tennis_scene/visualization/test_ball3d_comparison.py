import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.visualization.ball3d.rendering import (
    draw_candidate_overlay,
    line_segments,
    method_colors,
)
from src.tennis_scene.visualization.ball3d.report import broken_html_references


def test_trajectory_segments_omit_gap_endpoints_and_support_many_methods():
    points = np.array(
        [[1.0, 2.0], [3.0, 5.0], [900.0, 800.0], [7.0, 11.0], [13.0, 17.0]]
    )
    segments = line_segments(points, np.array([True, True, False, True, True]))
    np.testing.assert_array_equal(
        segments, [[[1.0, 2.0], [3.0, 5.0]], [[7.0, 11.0], [13.0, 17.0]]]
    )
    names = tuple(f"method_{i}" for i in range(7))
    colors = method_colors(names)
    assert list(colors) == list(names)
    assert len(set(colors.values())) == 7


def test_html_reference_audit_detects_missing_sibling_media(tmp_path):
    (tmp_path / "present.json").write_text("{}")
    page = tmp_path / "index.html"
    page.write_text(
        '<img src="data:image/png;base64,x"><a href="present.json">metrics</a><video src="missing.mp4"></video>'
    )
    assert broken_html_references(page) == ["missing.mp4"]


def test_candidate_overlay_marks_selected_raw_pixel_and_keeps_other_peaks():
    panel: NDArray[np.uint8] = np.zeros((100, 150, 3), np.uint8)
    uv = np.array([[22., 36.], [94., 74.], [0., 0.]])
    draw_candidate_overlay(panel, uv, np.array([True, True, False]),
        {"topk": 1}, {"topk": (23, 71, 239)}, scale=0.5)
    np.testing.assert_array_equal(panel[37, 47], [23, 71, 239])
    assert panel[15:22, 8:15].any()
    assert not panel[:3, :3].any()

import numpy as np

from src.tennis_scene.visualization.ball3d.rendering import line_segments, method_colors
from src.tennis_scene.visualization.ball3d.report import broken_html_references


def test_trajectory_segments_omit_gap_endpoints_and_support_many_methods():
    points = np.array([[1., 2.], [3., 5.], [900., 800.], [7., 11.], [13., 17.]])
    segments = line_segments(points, np.array([True, True, False, True, True]))
    np.testing.assert_array_equal(segments, [[[1., 2.], [3., 5.]], [[7., 11.], [13., 17.]]])
    names = tuple(f"method_{i}" for i in range(7))
    colors = method_colors(names)
    assert list(colors) == list(names)
    assert len(set(colors.values())) == 7


def test_html_reference_audit_detects_missing_sibling_media(tmp_path):
    (tmp_path / "present.json").write_text("{}")
    page = tmp_path / "index.html"
    page.write_text('<img src="data:image/png;base64,x"><a href="present.json">metrics</a><video src="missing.mp4"></video>')
    assert broken_html_references(page) == ["missing.mp4"]

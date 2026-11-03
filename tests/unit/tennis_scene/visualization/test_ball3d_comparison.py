import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.visualization.ball3d.rendering import line_segments, method_colors
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


def test_broad_covariance_ellipse_stays_inside_its_trajectory_panel():
    from dataclasses import replace

    from src.tennis_scene.evaluation.ball3d.runner import run_method
    from src.tennis_scene.visualization.ball3d.rendering import BallComparisonRenderer
    from tests.support.ball3d_evaluation import make_inputs
    from tests.unit.tennis_scene.evaluation.ball3d.test_scoring import FixedPointMethod

    inputs = make_inputs()
    run = run_method(FixedPointMethod(), inputs, bundle_sha256="fixture", code_commit="fixture", code_fingerprint="fixture")[0]
    covariance = np.tile(np.eye(3)*100, (4, 1, 1))
    covariance[~run.result.valid] = 0
    run = replace(run, result=replace(run.result, covariance=covariance))
    # No reader/video initialization: this tests only the pure court-panel layer.
    renderer = object.__new__(BallComparisonRenderer)
    renderer.inputs, renderer.runs = inputs, {"fixture": run}
    renderer.colors = {"fixture": (255, 0, 0)}
    renderer.lower, renderer.upper = np.array([-7., -13, -1]), np.array([7., 13, 5])
    canvas: NDArray[np.uint8] = np.zeros((400, 640, 3), np.uint8)
    renderer._trajectory(canvas, 0, (0, 1), (320, 0, 320, 200), "Covariance")
    assert canvas[:200, 320:].any()
    assert not canvas[:, :320].any()  # RGB panels must be untouched.
    assert not canvas[200:].any()  # Side panel must be untouched.

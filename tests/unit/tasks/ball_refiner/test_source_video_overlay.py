"""The source-video zoom translates means/labels and scales component uncertainty once."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from src.tasks.ball_refiner.visualization.overlay import component_geometry

SCRIPT = (Path(__file__).resolve().parents[4]
          / "knowledge/runs/run-i935-source-b-gate-r26-20261001/render_source_video.py")


@pytest.mark.parametrize("crop", [(0, 0, 1920, 1080), (100, 50, 384, 216)])
def test_source_crop_preserves_mean_and_covariance_geometry(crop, monkeypatch):
    module = runpy.run_path(str(SCRIPT))
    render = module["view"]
    means = np.array([[[.25, .3], [.8, .7]]])
    factors = np.array([[[[.01, 0], [.003, .02]], [[.04, 0], [0, .03]]]])
    logits = np.array([[0., 1.]])
    prediction = {"means": means, "scale_tril": factors, "mixture_logits": logits,
                  "source_size_wh": np.array([1920, 1080])}
    target = np.array([[.4, .2]])
    labels = {"target_uv": target, "target_reason": np.array([0])}
    captured = []
    points = []

    def mixture(image, uv, chol, weights, scale, *, point_summary):
        assert point_summary == "top_component"
        captured.append(component_geometry(uv, chol, weights, scale)[1])

    def marker(image, point, color, symbol):
        points.append(point)

    monkeypatch.setitem(render.__globals__, "draw_mixture", mixture)
    monkeypatch.setitem(render.__globals__, "mark", marker)
    det = np.array([520., 240.])
    image = render(np.zeros((1080, 1920, 3), np.uint8), prediction, labels, det, True, 0, crop)
    assert image.shape == (360, 640, 3)
    x, y, width, height = crop
    ratio = np.array([640 / width, 360 / height])
    origin = np.array([x, y])
    scale = np.array([1919., 1079.])
    np.testing.assert_allclose(points, [(det - origin) * ratio, (target[0] * scale - origin) * ratio])
    for i, ellipse in enumerate(captured[0]):
        np.testing.assert_allclose(ellipse.center, (means[0, i] * scale - origin) * ratio)
        angle = np.radians(ellipse.angle_degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        covariance = rotation @ np.diag((np.array(ellipse.semiaxes) / 2) ** 2) @ rotation.T
        transform = np.diag(scale * ratio)
        np.testing.assert_allclose(covariance, transform @ factors[0, i] @ factors[0, i].T @ transform)

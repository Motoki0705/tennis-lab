import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.components.camera_sampling.shapes import (
    shape_support,
    unit_shape_points,
)
from src.synthetic_data_generation.dataset.court.contracts import OrbitShape


@pytest.mark.parametrize('shape', list(OrbitShape))
def test_analytic_support_contains_entire_boundary(shape: OrbitShape) -> None:
    theta = np.linspace(0.0, 2 * np.pi, 32768, endpoint=False)
    points = unit_shape_points(shape, theta)
    angle = .73
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    shape_map = rotation @ np.diag((2.1, .7))
    normal_angles = np.linspace(0.0, 2 * np.pi, 31, endpoint=False)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    transformed_normals = normals @ shape_map
    exact = shape_support(shape, transformed_normals)
    measured = np.max(points @ transformed_normals.T, axis=0)
    assert np.all(measured <= exact + 1e-12)
    np.testing.assert_allclose(measured, exact, atol=1e-6, rtol=0)


def test_rectangle_reaches_corners_not_reached_by_ellipse() -> None:
    theta = np.arange(8) * np.pi / 4
    rectangle = unit_shape_points(OrbitShape.RECTANGLE, theta)
    ellipse = unit_shape_points(OrbitShape.ELLIPSE, theta)
    rounded = unit_shape_points(OrbitShape.SUPERELLIPSE, theta)
    np.testing.assert_allclose(rectangle[1], (1, 1))
    assert np.linalg.norm(ellipse[1]) < np.linalg.norm(rounded[1]) < np.linalg.norm(rectangle[1])
    np.testing.assert_allclose(np.sum(rounded**4, axis=1), 1.0)
    np.testing.assert_allclose(np.max(abs(rectangle), axis=1), 1.0)
    # Same axis-aligned square bound, but a diagonal cut must constrain a
    # rectangle more than an ellipse to keep its corners inside the hull.
    diagonal = np.array([[1., 1.]])
    assert shape_support(OrbitShape.RECTANGLE, diagonal)[0] == pytest.approx(2.)
    assert shape_support(OrbitShape.ELLIPSE, diagonal)[0] == pytest.approx(np.sqrt(2))

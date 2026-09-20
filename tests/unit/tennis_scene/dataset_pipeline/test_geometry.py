import numpy as np

from src.tennis_scene.dataset_pipeline.geometry import (
    TriangulationResult,
    TriangulationSettings,
    fill_triangulation_gaps,
    triangulate_ball,
)
from src.tennis_scene.dataset_pipeline.quality import project


def test_triangulation_recovers_known_points_and_rejects_single_view():
    cameras = [
        {
            "R": np.eye(3).tolist(),
            "t": [offset, 0, 10],
            "K": [[500, 0, 320], [0, 500, 180], [0, 0, 1]],
        }
        for offset in (-3, 3)
    ]
    expected = np.array([[0, 0, 1], [0.2, 0.1, 1.2], [0.4, 0.2, 1.4]], np.float32)
    uv = np.stack([project(expected, c)[0] / [640, 360] for c in cameras])
    visible: np.ndarray = np.ones((2, 3), bool)
    visible[1, 2] = False
    settings = TriangulationSettings(2.0, 100.0, (20.0, 30.0), (0.0, 15.0))
    result = triangulate_ball(
        uv, visible, cameras, size=(640, 360), fps=30.0, settings=settings
    )
    np.testing.assert_allclose(result.position[:2], expected[:2], atol=1e-5)
    np.testing.assert_array_equal(result.valid, [True, True, False])
    assert result.rejection_code[2] == 1
    assert np.isnan(result.position[2]).all()


def test_duplicate_camera_rays_do_not_produce_geometry_labels():
    camera = {
        "R": np.eye(3).tolist(),
        "t": [0, 0, 10],
        "K": [[500, 0, 320], [0, 500, 180], [0, 0, 1]],
    }
    result = triangulate_ball(
        np.full((2, 3, 2), 0.5),
        np.ones((2, 3), bool),
        [camera, camera],
        size=(640, 360),
        fps=30.0,
        settings=TriangulationSettings(2.0, 100.0, (20.0, 30.0), (0.0, 15.0)),
    )
    assert not result.valid.any()
    assert (result.rejection_code == 2).all()


def test_short_interpolation_and_long_prior_gaps_have_distinct_provenance():
    positions: np.ndarray = np.full((8, 3), np.nan, np.float32)
    positions[[1, 3, 7]] = np.array([[1, 1, 1], [3, 3, 3], [7, 7, 7]])
    valid = np.isfinite(positions).all(-1)
    result = TriangulationResult(
        positions, valid, np.zeros((2, 8)), np.zeros(8, np.uint8)
    )
    filled, source = fill_triangulation_gaps(
        result, np.full((8, 3), 9.0, np.float32), max_gap_frames=1
    )
    np.testing.assert_array_equal(source, [0, 1, 2, 1, 0, 0, 0, 1])
    np.testing.assert_allclose(filled[2], [2, 2, 2])
    np.testing.assert_allclose(filled[4], [9, 9, 9])

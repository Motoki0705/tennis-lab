"""Unit tests for fixed publication overview geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import src.synthetic_data_generation.visualization.publication.figures as figures
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.synthetic_data_generation.visualization.publication.cameras import (
    PublicationCameraCollection,
)
from src.synthetic_data_generation.visualization.publication.contracts import (
    CameraRenderingSemantics,
)
from src.synthetic_data_generation.visualization.publication.figures import (
    camera_collection_metrics,
    camera_forward_angle_differences_degrees,
    camera_render_indices,
    camera_rig_comparison_metrics,
    overview_panel_bounds,
    render_camera_comparison_figure,
    render_camera_figure,
)


def _camera_collection(
    owner: str,
    matrices: tuple[np.ndarray, ...],
    *,
    camera_ids: tuple[str, ...] | None = None,
) -> PublicationCameraCollection:
    ids = (
        tuple(f"camera-{index}" for index in range(len(matrices)))
        if camera_ids is None
        else camera_ids
    )
    cameras = tuple(
        SceneCamera(
            camera_id=camera_id,
            source_frame_index=index,
            width=64,
            height=64,
            intrinsics=(50.0, 0.0, 32.0, 0.0, 50.0, 32.0, 0.0, 0.0, 1.0),
            camera_to_scene=RigidTransform.from_matrix(matrix),
            image_path=f"images/{camera_id}.png",
        )
        for index, (camera_id, matrix) in enumerate(zip(ids, matrices, strict=True))
    )
    return PublicationCameraCollection(
        owner=owner,
        schema=f"{owner}_fixture_v1",
        scene_id="scene-0",
        logical_scene_id=None if owner == "reconstruction" else "logical-0",
        camera_ids=ids,
        cameras=cameras,
        camera_to_metric_scene=np.stack(matrices),
    )


def _layout() -> MultiCourtLayout:
    transform = RigidTransform.from_matrix(np.eye(4, dtype=np.float64))
    return MultiCourtLayout(
        courts=(
            CourtInstance(
                court_instance_id="court-0",
                candidate_id="candidate-0",
                scene_from_court=transform,
                court_from_scene=transform,
                fit_status="accepted",
                fit_metrics={},
                holdout_status="accepted",
                holdout_metrics={},
            ),
        ),
        complex_bounds_scene=(-20.0, -20.0, -1.0, 20.0, 20.0, 10.0),
        primary_court_instance_id="court-0",
    )


def _translated_matrices(count: int, *, offset: float = 0.0) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for index in range(count):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (float(index) + offset, 0.0, 4.0)
        result.append(matrix)
    return tuple(result)


def _canonical_drift_matrix() -> NDArray[np.float64]:
    angle = 1.1884684684684685
    forward = np.asarray(
        (0.6 * np.sin(angle), 0.8 * np.sin(angle), np.cos(angle)),
        dtype=np.float64,
    )
    right = np.cross(np.asarray((0.0, 1.0, 0.0)), forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix: NDArray[np.float64] = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward * (1.0 + 5.0e-8)))
    return matrix


def test_overview_panel_bounds_are_ordered_and_inside_canvas() -> None:
    width, height = 600, 400
    bounds = overview_panel_bounds((width, height))

    assert tuple(label for label, _ in bounds) == (
        "Court dataset",
        "BLCS dataset",
        "PLCS dataset",
        "Alignment evidence",
        "Captured cameras",
        "BLCS / PLCS cameras",
    )
    for _, (left, top, right, bottom) in bounds:
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height

    top_row = [rectangle for _, rectangle in bounds[:3]]
    bottom_row = [rectangle for _, rectangle in bounds[3:]]
    assert all(rectangle[1] < bottom_row[0][1] for rectangle in top_row)
    assert top_row[0][0] < top_row[1][0] < top_row[2][0]


@pytest.mark.parametrize("size", [(599, 400), (600, 399), (64, 64)])
def test_overview_panel_bounds_require_minimum_canvas(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="at least 600x400"):
        overview_panel_bounds(size)


def test_captured_render_indices_are_deterministic_bounded_and_endpoint_inclusive() -> (
    None
):
    first = camera_render_indices(491, maximum_rendered_cameras=24)
    second = camera_render_indices(491, maximum_rendered_cameras=24)

    assert first == second
    assert len(first) == 24
    assert first[0] == 0
    assert first[-1] == 490
    assert len(set(first)) == len(first)
    assert set(np.diff(first)) <= {21, 22}


def test_static_rig_has_no_temporal_metrics_or_trajectory_artist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = _camera_collection("blcs", _translated_matrices(3))

    def _unexpected_trajectory(_poses: object) -> object:
        raise AssertionError("static rig requested a temporal trajectory artist")

    monkeypatch.setattr(figures, "camera_trajectory_segments", _unexpected_trajectory)
    metrics = camera_collection_metrics(
        collection,
        rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
    )
    render_camera_figure(
        collection,
        _layout(),
        tmp_path / "static-rig.png",
        size=(320, 240),
        frustum_depth_metres=1.0,
        line_width=1.0,
        font_size=8,
        rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
        rendered_camera_indices=(0, 1, 2),
    )

    assert "trajectory_segment_count" not in metrics
    assert "trajectory_length_metres" not in metrics
    assert "maximum_adjacent_displacement_metres" not in metrics


def test_camera_rig_comparison_reports_identical_offset_and_rotated_poses() -> None:
    base_matrices = _translated_matrices(2)
    blcs = _camera_collection("blcs", base_matrices)
    identical = _camera_collection("plcs", base_matrices)
    identical_metrics = camera_rig_comparison_metrics(
        blcs,
        identical,
        centre_tolerance_metres=1.0e-6,
        forward_angle_tolerance_degrees=1.0e-6,
    )
    assert identical_metrics["coincident_camera_count"] == 2
    assert identical_metrics["coincident_camera_fraction"] == 1.0
    assert identical_metrics["maximum_centre_distance_metres"] == 0.0
    assert identical_metrics["maximum_forward_angle_difference_degrees"] == 0.0

    offset = _camera_collection("plcs", _translated_matrices(2, offset=0.01))
    offset_metrics = camera_rig_comparison_metrics(
        blcs,
        offset,
        centre_tolerance_metres=1.0e-6,
        forward_angle_tolerance_degrees=1.0e-6,
    )
    assert offset_metrics["coincident_camera_count"] == 0
    assert offset_metrics["maximum_centre_distance_metres"] == pytest.approx(0.01)

    angle_degrees = 10.0
    angle_radians = np.deg2rad(angle_degrees)
    rotation = np.asarray(
        (
            (np.cos(angle_radians), 0.0, np.sin(angle_radians)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle_radians), 0.0, np.cos(angle_radians)),
        ),
        dtype=np.float64,
    )
    rotated_matrices = tuple(matrix.copy() for matrix in base_matrices)
    for matrix in rotated_matrices:
        matrix[:3, :3] = rotation
    rotated = _camera_collection("plcs", rotated_matrices)
    rotated_metrics = camera_rig_comparison_metrics(
        blcs,
        rotated,
        centre_tolerance_metres=1.0e-6,
        forward_angle_tolerance_degrees=1.0e-6,
    )
    assert rotated_metrics["coincident_camera_count"] == 0
    assert rotated_metrics["maximum_centre_distance_metres"] == 0.0
    assert rotated_metrics["maximum_forward_angle_difference_degrees"] == pytest.approx(
        angle_degrees
    )


def test_byte_identical_canonical_drift_directions_are_exactly_coincident() -> None:
    matrices = tuple(_canonical_drift_matrix() for _ in range(6))
    for index, matrix in enumerate(matrices):
        matrix[0, 3] = float(index)
    blcs = _camera_collection("blcs", matrices)
    plcs = _camera_collection("plcs", tuple(matrix.copy() for matrix in matrices))
    forward = matrices[0][:3, 2]
    assert np.linalg.norm(forward) > 1.0
    np.testing.assert_array_equal(
        blcs.camera_to_metric_scene[:, :3, 2],
        plcs.camera_to_metric_scene[:, :3, 2],
    )

    metrics = camera_rig_comparison_metrics(
        blcs,
        plcs,
        centre_tolerance_metres=1.0e-6,
        forward_angle_tolerance_degrees=1.0e-6,
    )

    assert metrics["coincident_camera_count"] == 6
    assert metrics["coincident_camera_fraction"] == 1.0
    assert metrics["maximum_forward_angle_difference_degrees"] == 0.0


def test_nonidentical_directions_respect_angle_threshold_inside_and_outside() -> None:
    angles_degrees = (0.75e-6, 1.25e-6)
    blcs_matrices = list(_translated_matrices(2))
    plcs_matrices = [matrix.copy() for matrix in blcs_matrices]
    for matrix, angle_degrees in zip(plcs_matrices, angles_degrees, strict=True):
        angle = np.deg2rad(angle_degrees)
        matrix[:3, :3] = np.asarray(
            (
                (np.cos(angle), 0.0, np.sin(angle)),
                (0.0, 1.0, 0.0),
                (-np.sin(angle), 0.0, np.cos(angle)),
            ),
            dtype=np.float64,
        )
    metrics = camera_rig_comparison_metrics(
        _camera_collection("blcs", tuple(blcs_matrices)),
        _camera_collection("plcs", tuple(plcs_matrices)),
        centre_tolerance_metres=1.0e-6,
        forward_angle_tolerance_degrees=1.0e-6,
    )

    assert metrics["coincident_camera_count"] == 1
    assert metrics["coincident_camera_fraction"] == 0.5
    assert metrics["maximum_forward_angle_difference_degrees"] == pytest.approx(
        angles_degrees[1]
    )


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        (
            np.asarray(((np.nan, 0.0, 1.0),)),
            np.asarray(((0.0, 0.0, 1.0),)),
            "finite values",
        ),
        (
            np.asarray(((0.0, 0.0, 0.0),)),
            np.asarray(((0.0, 0.0, 1.0),)),
            "finite positive norms",
        ),
        (
            np.asarray(((np.finfo(np.float64).max, 1.0, 0.0),)),
            np.asarray(((np.finfo(np.float64).max, 1.0, 0.0),)),
            "finite positive norms",
        ),
    ],
    ids=["non-finite", "zero-norm", "overflowing-norm"],
)
def test_forward_angle_comparison_fails_closed_for_invalid_vectors(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        camera_forward_angle_differences_degrees(first, second)


def test_camera_rig_comparison_rejects_ordered_id_mismatch() -> None:
    matrices = _translated_matrices(2)
    blcs = _camera_collection("blcs", matrices)
    plcs = _camera_collection("plcs", matrices, camera_ids=("camera-1", "camera-0"))

    with pytest.raises(ValueError, match="identical ordered camera IDs"):
        camera_rig_comparison_metrics(
            blcs,
            plcs,
            centre_tolerance_metres=1.0e-6,
            forward_angle_tolerance_degrees=1.0e-6,
        )


def test_identical_six_camera_rigs_annotate_six_of_six_coincident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices = _translated_matrices(6)
    blcs = _camera_collection("blcs", matrices)
    plcs = _camera_collection("plcs", matrices)
    annotations: list[str] = []
    original_text2d = figures.Axes3D.text2D

    def _record_text(
        axis: object, _x: float, _y: float, text: str, **kwargs: object
    ) -> object:
        annotations.append(text)
        return original_text2d(axis, _x, _y, text, **kwargs)

    monkeypatch.setattr(figures.Axes3D, "text2D", _record_text)
    metrics = render_camera_comparison_figure(
        blcs,
        plcs,
        _layout(),
        tmp_path / "comparison.png",
        size=(320, 240),
        frustum_depth_metres=1.0,
        line_width=1.0,
        font_size=8,
        centre_tolerance_metres=1.0e-6,
        forward_angle_tolerance_degrees=1.0e-6,
    )

    assert metrics["coincident_camera_count"] == 6
    assert any("6/6 coincident" in text for text in annotations)
